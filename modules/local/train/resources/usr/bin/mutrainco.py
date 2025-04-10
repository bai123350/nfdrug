#!/usr/bin/env python

import argparse
import test
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json
import signal
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加超时处理
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("训练超时")

# 设置24小时超时
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(24 * 60 * 60)  # 24小时超时

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

def get_gpu_utilization(device=0):
    """获取GPU内存使用率"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device)
        total = torch.cuda.get_device_properties(device).total_memory
        return allocated / total
    return 0

def train_model(model, train_data, optimizer, epochs, device, max_time=24*3600):
    """添加时间限制的训练函数"""
    model = model.to(device)
    cross = nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    start_time = time.time()
    best_loss = float('inf')
    patience = 1000  # 早停耐心值
    no_improve = 0

    try:
        for e in range(epochs):
            # 检查是否超时
            if time.time() - start_time > max_time:
                logger.warning("训练达到时间限制")
                break

            epoch_start = time.time()
            model.train()
            train_loss = 0
            total_correct = 0
            total_samples = 0

            for d in train_data:
                # 监控GPU使用
                current_gpu_util = get_gpu_utilization(device)
                if current_gpu_util > 0.95:
                    logger.warning(f"GPU使用率过高: {current_gpu_util:.2%}")
                    torch.cuda.empty_cache()
                    time.sleep(1)

                optimizer.zero_grad()
                data = d[0].float().to(device)
                label = d[1].float().to(device)
                event_label = label[:, 0]
                time_label = label[:, 1]

                pred = model(data, event_label, time_label)
                loss = cross(pred, event_label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                optimizer.step()

                train_loss += loss.item()
                pred_labels = (pred >= 0.5).float()
                correct = (pred_labels == event_label).sum().item()
                total_correct += correct
                total_samples += event_label.size(0)

            avg_loss = train_loss / len(train_data)
            avg_accuracy = total_correct / total_samples

            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f"早停: {patience} 轮无改善")
                break

            losses.append(avg_loss)
            accuracies.append(avg_accuracy)

            epoch_time = time.time() - epoch_start
            if e % 100 == 0:  # 减少日志频率
                logger.info(f"Epoch {e + 1}/{epochs}, Loss: {avg_loss:.4f}, "
                          f"Accuracy: {avg_accuracy:.4f}, Time: {epoch_time:.2f}s")

    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        raise
    finally:
        # 清理GPU内存
        torch.cuda.empty_cache()

    return losses, accuracies

def evaluate_model(model, test_loader, device):
    """评估模型在测试集上的性能"""
    model.eval()
    cross = nn.CrossEntropyLoss()
    test_loss = 0
    total_correct = 0
    total_samples = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for d in test_loader:
            data = d[0].float().to(device)
            label = d[1].float().to(device)
            event_label = label[:, 0]
            time_label = label[:, 1]

            pred = model(data, event_label, time_label)
            loss = cross(pred, event_label)

            test_loss += loss.item()
            pred_labels = (pred >= 0.5).float()
            correct = (pred_labels == event_label).sum().item()
            total_correct += correct
            total_samples += event_label.size(0)

            predictions.extend(pred_labels.cpu().numpy())
            true_labels.extend(event_label.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy, predictions, true_labels

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

def plot_test_results(predictions, true_labels, m):
    """绘制测试结果图"""
    plt.figure(figsize=(10, 5))

    # 混淆矩阵热图
    cm = confusion_matrix(true_labels, predictions)
    # plt.subplot(1, 2, 1)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(true_labels)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # ROC曲线
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{os.path.join(m, 'test_results.pdf')}")

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def batch_process_folders(folder_list, epochs, wait_time=30):
    """优化的批处理函数"""
    num_gpus = torch.cuda.device_count()
    folders_per_gpu = min(20, len(folder_list) // num_gpus + (1 if len(folder_list) % num_gpus else 0))
    results = {}

    # 按GPU分配文件夹
    gpu_folders = {}
    for i, folder in enumerate(folder_list):
        gpu_id = i // folders_per_gpu % num_gpus
        if gpu_id not in gpu_folders:
            gpu_folders[gpu_id] = []
        gpu_folders[gpu_id].append(folder)

    # 处理每个GPU的任务
    for gpu_id, folders in gpu_folders.items():
        logger.info(f"GPU {gpu_id} 开始处理 {len(folders)} 个任务")
        device = f"cuda:{gpu_id}"

        for folder in folders:
            try:
                # 设置每个任务的超时时间
                signal.alarm(8 * 60 * 60)  # 8小时超时

                # 加载并训练模型
                model = torch.load(os.path.join(folder, 'model.pt'), map_location=device)
                train_data = torch.load(os.path.join(folder, 'train_data.pt'), map_location='cpu')
                test_data = torch.load(os.path.join(folder, 'test_data.pt'), map_location='cpu')
                train_loader = torch.utils.data.DataLoader(
                    train_data.dataset,
                    batch_size=min(train_data.batch_size, 16),
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )

                optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
                model = model.to(device)

                # 训练时记录开始时间
                start_time = time.time()
                logger.info(f"开始处理 {folder} 在 GPU {gpu_id}")

                losses, accuracies = train_model(model, train_loader, optimizer, epochs, device)

                # 保存结果
                results[folder] = {
                    'losses': losses,
                    'accuracies': accuracies
                }
                make_dir(os.path.join('train', folder))
                plot(losses, accuracies, os.path.join('train', folder))

                # 在训练完成后进行测试评估
                test_loader = torch.utils.data.DataLoader(
                    test_data.dataset,
                    batch_size=min(test_data.batch_size, 16),
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )

                test_loss, test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)
                logger.info(f"测试结果 - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

                # 保存测试结果
                results[folder].update({
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                })

                # 保存结果到文件
                results_file = os.path.join('train', folder, 'train_res.json')
                with open(results_file, 'w') as f:
                    json.dump(results[folder], f, indent=4)

                torch.save(train_data, os.path.join('train', folder, "train_data.pt"))
                torch.save(test_data, os.path.join('train', folder, "test_data.pt"))

                # 绘制测试结果图
                plot_test_results(predictions, true_labels, os.path.join('train', folder))

                # 保存最优模型
                torch.save(model.state_dict(), os.path.join('train', folder, 'best_model.pt'))

                # 记录完成时间
                end_time = time.time()
                logger.info(f"完成处理 {folder}, 用时: {(end_time-start_time)/3600:.2f}小时")

            except TimeoutError:
                logger.error(f"任务 {folder} 超时")
                continue
            except Exception as e:
                logger.error(f"处理任务 {folder} 失败: {str(e)}")
                continue
            finally:
                # 重置报警
                signal.alarm(0)
                # 清理GPU内存
                torch.cuda.empty_cache()
                gc.collect()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural networks on multiple GPUs")
    parser.add_argument("--folder", type=str, help="comma-separated list of model folders")
    parser.add_argument("--m", type=int, default=1000, help="number of epochs")
    parser.add_argument("--wait", type=int, default=1, help="waiting time between checks (seconds)")
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



