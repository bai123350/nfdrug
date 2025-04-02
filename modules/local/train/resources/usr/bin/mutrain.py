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

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_data, optimizer, epoch, device):

    losses = []
    accuracies = []

    for e in range(epoch):
        train_loss = 0
        total_correct = 0
        total_samples = 0

        for d in train_data:
            optimizer.zero_grad()
            data = d[0].float().to(device)
            label = d[1].float().to(device)
            event_label = label[:, 0]
            time_label = label[:, 1]

            pred = model(data, event_label, time_label)

            # 计算损失
            loss = cross(pred, event_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算准确率
            pred_labels = (pred >= 0.5).float()  # 将预测值转换为 0 或 1
            correct = (pred_labels == event_label).sum().item()
            total_correct += correct
            total_samples += event_label.size(0)

        # 计算平均损失和准确率
        avg_loss = train_loss / len(train_data)
        avg_accuracy = total_correct / total_samples

        # 保存当前 epoch 的损失和准确率
        losses.append(avg_loss)
        accuracies.append(avg_accuracy)

        print(f"Epoch {e + 1}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    return losses, accuracies


def plot(loss, acc,m):
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
    # plt.show()
    plt.savefig(f"{os.path.join(m, 'loss_acc.pdf')}")


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("--folder", type=str, help="model file")
    parser.add_argument("--m", type=str, default=2000, help="number of epochs")
    args = parser.parse_args()
    set_seed(42)

    for p in args.folder.split(","):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = torch.load(f"{os.path.join( p, 'model.pt')}")
        train_data = torch.load(f"{os.path.join(p, 'train_data.pt')}")
        test_data = torch.load(f"{os.path.join(p, 'test_data.pt')}")

        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        cross = nn.CrossEntropyLoss()

        make_dir(f"{os.path.join('train', p)}")

        loss,acc = train_model(model, train_data, optimizer, args.m, device)
        plot(loss,acc,os.path.join('train', p))


