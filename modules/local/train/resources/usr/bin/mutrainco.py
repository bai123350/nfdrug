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

# import argparse
# import test
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import os
# import logging
# import gc
# import time
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# import json

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def get_gpu_memory(device=0):
#     """获取GPU内存使用情况"""
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
#     return 0, 0

# def is_gpu_memory_available(threshold=0.9, device=0):
#     """检查指定GPU内存是否有足够空间"""
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated(device)
#         total = torch.cuda.get_device_properties(device).total_memory
#         return allocated < threshold * total
#     return False

# def get_available_gpu():
#     """获取当前可用的GPU"""
#     available_gpus = []
#     for i in range(torch.cuda.device_count()):
#         if is_gpu_memory_available(threshold=0.9, device=i):
#             available_gpus.append(i)
#     return available_gpus

# def train_model(model, train_data, optimizer, epochs, device):
#     model = model.to(device)
#     cross = nn.CrossEntropyLoss()
#     losses = []
#     accuracies = []

#     try:
#         for e in range(epochs):
#             model.train()
#             train_loss = 0
#             total_correct = 0
#             total_samples = 0

#             for d in train_data:
#                 # 检查GPU内存使用
#                 allocated, reserved = get_gpu_memory(device)
#                 if allocated > 0.9 * torch.cuda.get_device_properties(device).total_memory:
#                     logger.warning("GPU内存接近上限，等待释放...")
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     time.sleep(1)  # 等待内存释放

#                 optimizer.zero_grad()
#                 data = d[0].float().to(device)
#                 label = d[1].float().to(device)
#                 event_label = label[:, 0]
#                 time_label = label[:, 1]

#                 pred = model(data, event_label, time_label)
#                 loss = cross(pred, event_label)
#                 loss.backward()
#                 optimizer.step()

#                 train_loss += loss.item()
#                 pred_labels = (pred >= 0.5).float()
#                 correct = (pred_labels == event_label).sum().item()
#                 total_correct += correct
#                 total_samples += event_label.size(0)

#             avg_loss = train_loss / len(train_data)
#             avg_accuracy = total_correct / total_samples

#             losses.append(avg_loss)
#             accuracies.append(avg_accuracy)

#             if e % 100 == 0 and e > 0:
#                 logger.info(f"Epoch {e + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

#     except Exception as e:
#         logger.error(f"训练过程发生错误: {str(e)}")
#         raise

#     return losses, accuracies

# def evaluate_model(model, test_loader, device):
#     """评估模型在测试集上的性能"""
#     model.eval()
#     cross = nn.CrossEntropyLoss()
#     test_loss = 0
#     total_correct = 0
#     total_samples = 0
#     predictions = []
#     true_labels = []

#     with torch.no_grad():
#         for d in test_loader:
#             data = d[0].float().to(device)
#             label = d[1].float().to(device)
#             event_label = label[:, 0]
#             time_label = label[:, 1]

#             pred = model(data, event_label, time_label)
#             loss = cross(pred, event_label)

#             test_loss += loss.item()
#             pred_labels = (pred >= 0.5).float()
#             correct = (pred_labels == event_label).sum().item()
#             total_correct += correct
#             total_samples += event_label.size(0)

#             predictions.extend(pred_labels.cpu().numpy())
#             true_labels.extend(event_label.cpu().numpy())

#     avg_loss = test_loss / len(test_loader)
#     accuracy = total_correct / total_samples
#     return avg_loss, accuracy, predictions, true_labels

# def plot(loss, acc, m):
#     loss = [round(float(i), 2) for i in loss]
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(loss, label="Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title("Training Loss")
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(acc, label="acc")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.title("Training Accuracy")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{os.path.join(m, 'loss_acc.pdf')}")

# def plot_test_results(predictions, true_labels, m):
#     """绘制测试结果图"""
#     plt.figure(figsize=(10, 5))

#     # 混淆矩阵热图
#     cm = confusion_matrix(true_labels, predictions)
#     # plt.subplot(1, 2, 1)
#     # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     # plt.title('Confusion Matrix')
#     # plt.xlabel('Predicted')
#     # plt.ylabel('True')
#     plt.subplot(1, 2, 1)
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(np.unique(true_labels)))
#     plt.xticks(tick_marks, tick_marks)
#     plt.yticks(tick_marks, tick_marks)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')

#     # ROC曲线
#     fpr, tpr, _ = roc_curve(true_labels, predictions)
#     roc_auc = auc(fpr, tpr)
#     plt.subplot(1, 2, 2)
#     plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], 'k--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig(f"{os.path.join(m, 'test_results.pdf')}")

# def make_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)

# def batch_process_folders(folder_list, epochs, wait_time=30):
#     """多GPU批量处理文件夹"""
#     pending_folders = folder_list.copy()
#     active_tasks = {}  # 格式: {gpu_id: folder_name}
#     results = {}

#     while pending_folders or active_tasks:
#         # 检查当前活跃任务的状态
#         for gpu_id in list(active_tasks.keys()):
#             if not active_tasks[gpu_id]:
#                 continue

#             folder = active_tasks[gpu_id]
#             device = f"cuda:{gpu_id}"

#             try:
#                 # 加载并训练模型
#                 model = torch.load(os.path.join(folder, 'model.pt'), map_location=device)
#                 train_data = torch.load(os.path.join(folder, 'train_data.pt'), map_location='cpu')
#                 test_data = torch.load(os.path.join(folder, 'test_data.pt'), map_location='cpu')
#                 train_loader = torch.utils.data.DataLoader(
#                     train_data.dataset,
#                     batch_size=min(train_data.batch_size, 16),
#                     shuffle=True,
#                     num_workers=2,
#                     pin_memory=True
#                 )

#                 optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
#                 model = model.to(device)

#                 losses, accuracies = train_model(model, train_loader, optimizer, epochs, device)

#                 # 保存结果
#                 results[folder] = {
#                     'losses': losses,
#                     'accuracies': accuracies
#                 }
#                 make_dir(os.path.join('train', folder))
#                 plot(losses, accuracies, os.path.join('train', folder))

#                 # 在训练完成后进行测试评估
#                 test_loader = torch.utils.data.DataLoader(
#                     test_data.dataset,
#                     batch_size=min(test_data.batch_size, 16),
#                     shuffle=False,
#                     num_workers=2,
#                     pin_memory=True
#                 )

#                 test_loss, test_accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)
#                 logger.info(f"测试结果 - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

#                 # 保存测试结果
#                 results[folder].update({
#                     'test_loss': test_loss,
#                     'test_accuracy': test_accuracy
#                     # 'test_predictions': predictions,
#                     # 'test_true_labels': true_labels
#                 })

#                 # 保存结果到文件
#                 results_file = os.path.join('train', folder, 'train_res.json')
#                 with open(results_file, 'w') as f:
#                     json.dump(results[folder], f, indent=4)

#                 torch.save(train_data, os.path.join('train', folder, "train_data.pt"))
#                 torch.save(test_data, os.path.join('train', folder, "test_data.pt"))

#                 # 绘制测试结果图
#                 plot_test_results(predictions, true_labels, os.path.join('train', folder))

#                 # 保存最优模型
#                 torch.save(model.state_dict(), os.path.join('train', folder, 'best_model.pt'))

#                 # 释放GPU资源
#                 del model
#                 torch.cuda.empty_cache()
#                 active_tasks[gpu_id] = None

#                 logger.info(f"完成处理文件夹 {folder} 在 GPU {gpu_id}")

#             except Exception as e:
#                 logger.error(f"处理文件夹 {folder} 在 GPU {gpu_id} 时发生错误: {str(e)}")
#                 active_tasks[gpu_id] = None

#         # 为空闲的GPU分配新任务
#         available_gpus = get_available_gpu()
#         for gpu_id in available_gpus:
#             if gpu_id in active_tasks and active_tasks[gpu_id] is not None:
#                 continue

#             if pending_folders:
#                 folder = pending_folders.pop(0)
#                 active_tasks[gpu_id] = folder
#                 logger.info(f"分配文件夹 {folder} 到 GPU {gpu_id}")

#         time.sleep(wait_time)  # 等待一段时间再检查状态

#     return results

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train neural networks on multiple GPUs")
#     parser.add_argument("--folder", type=str, help="comma-separated list of model folders")
#     parser.add_argument("--m", type=int, default=1000, help="number of epochs")
#     parser.add_argument("--wait", type=int, default=3, help="waiting time between checks (seconds)")
#     args = parser.parse_args()

#     set_seed(22222)

#     try:
#         folder_list = args.folder.split(",")
#         results = batch_process_folders(
#             folder_list,
#             args.m,
#             wait_time=args.wait
#         )
#         logger.info("所有文件夹处理完成")
#     except Exception as e:
#         logger.error(f"程序执行失败: {str(e)}")
#         raise



