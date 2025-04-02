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

def get_gpu_memory():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)
    return 0, 0

def is_gpu_memory_available(threshold=0.9):
    """检查GPU内存是否有足够空间"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0)
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated < threshold * total
    return False

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
                allocated, reserved = get_gpu_memory()
                if allocated > 0.9 * torch.cuda.get_device_properties(0).total_memory:
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

            logger.info(f"Epoch {e + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

            # 定期保存检查点
            # if (e + 1) % 10 == 0:
            #     torch.save({
            #         'epoch': e + 1,
            #         'model_state_dict': model.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #         'loss': avg_loss,
            #     }, os.path.join('train', folder_path, f'checkpoint_epoch_{e+1}.pt'))

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

def batch_process_folders(folder_list, epochs, device, memory_threshold=0.9, wait_time=30):
    """批量处理文件夹，支持等待和重试"""
    active_models = {}
    results = {}
    pending_folders = folder_list.copy()  # 待处理的文件夹

    while pending_folders:
        # 尝试加载新模型
        current_batch = []
        for folder in pending_folders[:]:  # 使用切片创建副本进行迭代
            if not is_gpu_memory_available(memory_threshold):
                logger.info("GPU内存已达到阈值，等待当前批次处理完成")
                break

            try:
                logger.info(f"尝试加载文件夹: {folder}")
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

                active_models[folder] = {
                    'model': model,
                    'train_loader': train_loader,
                    'optimizer': optimizer,
                    'losses': [],
                    'accuracies': [],
                    'epoch': 0
                }
                make_dir(os.path.join('train', folder))
                current_batch.append(folder)
                pending_folders.remove(folder)

            except Exception as e:
                logger.error(f"加载文件夹 {folder} 失败: {str(e)}")
                continue

        # 训练当前批次的模型
        if active_models:
            cross = nn.CrossEntropyLoss()

            for epoch in range(epochs):
                for folder in current_batch:
                    if folder not in active_models:
                        continue

                    data = active_models[folder]
                    model = data['model']
                    train_loader = data['train_loader']
                    optimizer = data['optimizer']

                    model.train()
                    train_loss = 0
                    total_correct = 0
                    total_samples = 0

                    for batch in train_loader:
                        optimizer.zero_grad()
                        x = batch[0].float().to(device)
                        label = batch[1].float().to(device)
                        event_label = label[:, 0]
                        time_label = label[:, 1]

                        pred = model(x, event_label, time_label)
                        loss = cross(pred, event_label)
                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        pred_labels = (pred >= 0.5).float()
                        correct = (pred_labels == event_label).sum().item()
                        total_correct += correct
                        total_samples += event_label.size(0)

                        # 释放不需要的张量
                        del x, label, pred
                        torch.cuda.empty_cache()

                    avg_loss = train_loss / len(train_loader)
                    avg_accuracy = total_correct / total_samples

                    data['losses'].append(avg_loss)
                    data['accuracies'].append(avg_accuracy)
                    data['epoch'] = epoch + 1

                    logger.info(f"文件夹: {folder}, Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

                    if (epoch + 1) % 10 == 0:
                        # 保存检查点
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': avg_loss,
                        }, os.path.join('train', folder, f'checkpoint_epoch_{epoch+1}.pt'))

                    # 保存结果并释放内存
                    if epoch == epochs - 1:
                        results[folder] = {
                            'losses': data['losses'],
                            'accuracies': data['accuracies']
                        }
                        plot(data['losses'], data['accuracies'], os.path.join('train', folder))

                        # 释放该模型的内存
                        del active_models[folder]
                        torch.cuda.empty_cache()
                        gc.collect()
                        logger.info(f"完成处理文件夹 {folder} 并释放内存")

        # 如果还有待处理的文件夹，等待一段时间后继续
        if pending_folders:
            logger.info(f"等待 {wait_time} 秒后处理剩余文件夹: {pending_folders}")
            time.sleep(wait_time)
            torch.cuda.empty_cache()
            gc.collect()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on single GPU")
    parser.add_argument("--folder", type=str, help="model file")
    parser.add_argument("--m", type=int, default=2000, help="number of epochs")
    parser.add_argument("--wait", type=int, default=30, help="waiting time between batches (seconds)")
    args = parser.parse_args()

    set_seed(22222)

    try:
        folder_list = args.folder.split(",")
        results = batch_process_folders(
            folder_list,
            args.m,
            torch.device("cuda:0"),
            wait_time=args.wait
        )

        logger.info("所有文件夹处理完成")

    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        raise


