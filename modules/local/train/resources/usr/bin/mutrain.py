#!/usr/bin/env python

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(rank, world_size):
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            rank=rank,
            world_size=world_size
        )
    except Exception as e:
        logger.error(f"初始化分布式环境失败: {str(e)}")
        raise

def cleanup():
    dist.destroy_process_group()

def train_model_distributed(rank, world_size, folder_path, epochs, device):
    try:
        setup(rank, world_size)

        # 在每个进程中加载模型和数据
        model = torch.load(os.path.join(folder_path, 'model.pt'), map_location=f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        train_data = torch.load(os.path.join(folder_path, 'train_data.pt'), map_location=f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        model = model.to(rank)
        if torch.cuda.device_count() > 1:
            model = DDP(model, device_ids=[rank])

        train_sampler = DistributedSampler(train_data.dataset,
                                         num_replicas=world_size,
                                         rank=rank)
        train_loader = torch.utils.data.DataLoader(
            train_data.dataset,
            batch_size=train_data.batch_size,
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )

        # 修改损失函数
        cross = nn.BCEWithLogitsLoss(reduction='mean')
        losses = []
        accuracies = []

        for e in range(epochs):
            model.train()
            train_sampler.set_epoch(e)
            train_loss = 0
            total_correct = 0
            total_samples = 0

            for d in train_loader:
                optimizer.zero_grad()
                data = d[0].float().to(rank)
                label = d[1].float().to(rank)
                event_label = label[:, 0]
                time_label = label[:, 1]

                pred = model(data, event_label, time_label)

                # 确保维度匹配
                if pred.dim() == 1:
                    pred = pred.unsqueeze(1)
                event_label = event_label.view_as(pred)

                loss = cross(pred, event_label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred_labels = (torch.sigmoid(pred) >= 0.5).float()
                correct = (pred_labels == event_label).sum().item()
                total_correct += correct
                total_samples += event_label.size(0)

            avg_loss = train_loss / len(train_loader)
            avg_accuracy = total_correct / total_samples

            if world_size > 1:
                avg_loss = torch.tensor(avg_loss).to(rank)
                avg_accuracy = torch.tensor(avg_accuracy).to(rank)
                dist.all_reduce(avg_loss)
                dist.all_reduce(avg_accuracy)
                avg_loss = avg_loss.item() / world_size
                avg_accuracy = avg_accuracy.item() / world_size

            losses.append(avg_loss)
            accuracies.append(avg_accuracy)

            if rank == 0:
                logger.info(f"Epoch {e + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

                if (e + 1) % 10 == 0:
                    checkpoint = {
                        'epoch': e + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                    }
                    torch.save(checkpoint, f'checkpoint_epoch_{e+1}.pt')

    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        raise
    finally:
        cleanup()

    return losses, accuracies

def plot(loss, acc, m):
    loss = [round(float(i), 2) for i in loss]
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(acc, label="acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{os.path.join(m, 'loss_acc.pdf')}")

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_model_training(folder_path, epochs, world_size):
    try:
        if not torch.cuda.is_available() and world_size > 1:
            logger.warning("没有找到GPU，将使用CPU进行训练")
            world_size = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将模型加载移到设备初始化之后
        make_dir(os.path.join('train', folder_path))

        if world_size > 1:
            mp.spawn(
                train_model_distributed,
                args=(world_size, folder_path, epochs, device),
                nprocs=world_size,
                join=True
            )
        else:
            # 单GPU或CPU训练
            model = torch.load(os.path.join(folder_path, 'model.pt'), map_location=device)
            train_data = torch.load(os.path.join(folder_path, 'train_data.pt'), map_location=device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
            loss, acc = train_model_distributed(0, 1, model, train_data, optimizer, epochs, device)
            plot(loss, acc, os.path.join('train', folder_path))

    except Exception as e:
        logger.error(f"处理文件夹 {folder_path} 时发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置多进程启动方式为spawn
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Train a neural network with multi-processing")
    parser.add_argument("--folder", type=str, help="model file")
    parser.add_argument("--m", type=int, default=2000, help="number of epochs")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="number of processes")
    args = parser.parse_args()

    set_seed(42)

    try:
        # 直接处理每个文件夹，不使用进程池
        for folder in args.folder.split(","):
            process_model_training(
                folder,
                epochs=args.m,
                world_size=min(args.world_size, torch.cuda.device_count())
            )
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise


