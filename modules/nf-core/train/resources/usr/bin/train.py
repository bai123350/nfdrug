#!/usr/bin/env python

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_data, optimizer, epoch, device):
    patience = 500
    patience_count = 0
    global_loss = 0
    global_con = 0

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("--model", type=str, help="model file")
    args = parser.parse_args()

    model = torch.load(args.model)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    cross = nn.CrossEntropyLoss()
