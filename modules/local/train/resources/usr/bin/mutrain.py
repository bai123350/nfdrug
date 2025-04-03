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
import gc
import time

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

def get_gpu_memory(device=0):
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    return 0, 0

def is_gpu_memory_available(threshold=0.9, device=0):
    """检查指定GPU内存是否有足够空间"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return allocated < threshold * total
    return False

def get_available_gpu():
    """获取当前可用的GPU"""
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        if is_gpu_memory_available(threshold=0.9, device=i):
            available_gpus.append(i)
    return available_gpus

def train_model(model, train_data, optimizer, epochs, device):
    model = model.to(device)
    cross = nn.CrossEntropyLoss()
    losses = []
    accuracies = []

    try:
        for e in range(epochs):
            model.train()
            train_loss = 0
            total_correct = 0
            total_samples = 0

            for d in train_data:
                # 检查GPU内存使用
                allocated, reserved = get_gpu_memory(device)
                if allocated > 0.9 * torch.cuda.get_device_properties(device).total_memory:
                    logger.warning("GPU内存接近上限，等待释放...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(1)  # 等待内存释放

                optimizer.zero_grad()
                data = d[0].float().to(device)
                label = d[1].float().to(device)
                event_label = label[:, 0]
                time_label = label[:, 1]

                pred = model(data, event_label, time_label)
                loss = cross(pred, event_label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pred_labels = (pred >= 0.5).float()
                correct = (pred_labels == event_label).sum().item()
                total_correct += correct
                total_samples += event_label.size(0)

            avg_loss = train_loss / len(train_data)
            avg_accuracy = total_correct / total_samples

            losses.append(avg_loss)
            accuracies.append(avg_accuracy)

            if e % 100 == 0 and e > 0:
                logger.info(f"Epoch {e + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")


    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        raise

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

def batch_process_folders(folder_list, epochs, wait_time=30):
    """多GPU批量处理文件夹"""
    pending_folders = folder_list.copy()
    active_tasks = {}  # 格式: {gpu_id: folder_name}
    results = {}

    while pending_folders or active_tasks:
        # 检查当前活跃任务的状态
        for gpu_id in list(active_tasks.keys()):
            if not active_tasks[gpu_id]:
                continue

            folder = active_tasks[gpu_id]
            device = f"cuda:{gpu_id}"

            try:
                # 加载并训练模型
                model = torch.load(os.path.join(folder, 'model.pt'), map_location=device)
                train_data = torch.load(os.path.join(folder, 'train_data.pt'), map_location='cpu')
                train_loader = torch.utils.data.DataLoader(
                    train_data.dataset,
                    batch_size=min(train_data.batch_size, 16),
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )

                optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
                model = model.to(device)

                losses, accuracies = train_model(model, train_loader, optimizer, epochs, device)

                # 保存结果
                results[folder] = {
                    'losses': losses,
                    'accuracies': accuracies
                }
                make_dir(os.path.join('train', folder))
                plot(losses, accuracies, os.path.join('train', folder))

                # 释放GPU资源
                del model
                torch.cuda.empty_cache()
                active_tasks[gpu_id] = None

                logger.info(f"完成处理文件夹 {folder} 在 GPU {gpu_id}")

            except Exception as e:
                logger.error(f"处理文件夹 {folder} 在 GPU {gpu_id} 时发生错误: {str(e)}")
                active_tasks[gpu_id] = None

        # 为空闲的GPU分配新任务
        available_gpus = get_available_gpu()
        for gpu_id in available_gpus:
            if gpu_id in active_tasks and active_tasks[gpu_id] is not None:
                continue

            if pending_folders:
                folder = pending_folders.pop(0)
                active_tasks[gpu_id] = folder
                logger.info(f"分配文件夹 {folder} 到 GPU {gpu_id}")

        time.sleep(wait_time)  # 等待一段时间再检查状态

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural networks on multiple GPUs")
    parser.add_argument("--folder", type=str, help="comma-separated list of model folders")
    parser.add_argument("--m", type=int, default=1000, help="number of epochs")
    parser.add_argument("--wait", type=int, default=3, help="waiting time between checks (seconds)")
    args = parser.parse_args()

    set_seed(22222)

    try:
        folder_list = args.folder.split(",")
        results = batch_process_folders(
            folder_list,
            args.m,
            wait_time=args.wait
        )
        logger.info("所有文件夹处理完成")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise



