#!/usr/bin/env python

import argparse
import json
import logging
import os
from sympy import im
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from net import *
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Process:
    def __init__(self, args):
        self.file_list = args.dir.split(",")

    def top10(self):
        all_test_acc = {}
        for path in self.file_list:
            with open(os.path.join(path ,'train_res.json'), 'r') as f:
                data = json.load(f)
                all_test_acc[path] = data['test_accuracy']
        all_test_acc = dict(sorted(all_test_acc.items(), key=lambda item: item[1], reverse=True))
        top_10_acc = {}
        for index, (key, value) in enumerate(all_test_acc.items()):
            if index == 0 and int(value) != 1:
                top_10_acc = dict(list(all_test_acc.items())[:10])
                break
            if int(value) == 1: continue
            else:
                top_10_acc = dict(list(all_test_acc.items())[:(index + 1)])
                break
        return (top_10_acc,all_test_acc)

class DataUtil(Process):
    def read_data(self):
        results = {}
        for p in self.file_list:
            if p in self.top10()[0].keys():
                train_data = torch.load(os.path.join(p, 'train_data.pt'), map_location='cpu')
                test_data = torch.load(os.path.join(p, 'test_data.pt'), map_location='cpu')

                train_loader = DataLoader(train_data.dataset, batch_size=len(train_data.dataset), shuffle=False)
                test_loader = DataLoader(test_data.dataset, batch_size=len(test_data.dataset), shuffle=False)

                train_features, train_labels = next(iter(train_loader))
                test_features, test_labels = next(iter(test_loader))

                # 确保标签是PyTorch张量
                train_event_labels = torch.tensor(train_labels[:, 0])
                test_event_labels = torch.tensor(test_labels[:, 0])

                logger.info(f"Processing {train_event_labels}...")
                logger.info(f"Processing {train_event_labels.shape}...")


                lr_model = LogisticRegression(max_iter=100)
                lr_model.fit(train_features.numpy(), train_event_labels.numpy())
                lr_predictions = lr_model.predict(test_features.numpy())
                lr_accuracy = accuracy_score(test_event_labels.numpy(), lr_predictions)


                rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
                rf_model.fit(train_features.numpy(), train_event_labels.numpy())
                rf_predictions = rf_model.predict(test_features.numpy())
                rf_accuracy = accuracy_score(test_event_labels.numpy(), rf_predictions)


                nn_model = nn.Sequential(
                    nn.Linear(train_features.shape[1], 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
                criterion = nn.BCELoss()

                for epoch in range(100):
                    nn_model.train()
                    optimizer.zero_grad()
                    outputs = nn_model(train_features.float())
                    loss = criterion(outputs.squeeze(), train_event_labels.float())
                    loss.backward()
                    optimizer.step()

                nn_model.eval()
                with torch.no_grad():
                    nn_predictions = nn_model(test_features.float()).squeeze()
                    nn_predictions = (nn_predictions >= 0.5).float()
                nn_accuracy = accuracy_score(test_event_labels.numpy(), nn_predictions.numpy())

                # 保存结果
                results[p] = {
                    'Logistic Regression': lr_accuracy,
                    'Random Forest': rf_accuracy,
                    'Neural Network': nn_accuracy,
                    'BFReg NN' : self.top10()[0][p]
                }

        return results,self.top10()[1]

    def plot_accuracies(self, results):
        """绘制3种模型的预测准确率小提琴图和箱线图，并显示p值"""
        accuracies = {
            'Model': [],
            'Accuracy': []
        }

        for model_results in results.values():
            for model, acc in model_results.items():
                accuracies['Model'].append(model)
                accuracies['Accuracy'].append(acc)

        # 转换为DataFrame以便使用Seaborn绘图
        df = pd.DataFrame(accuracies)

        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Model', y='Accuracy', data=df, inner=None, palette='Set2', linewidth=1.5)
        sns.boxplot(x='Model', y='Accuracy', data=df, width=0.2, palette='Set2', showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "8"})

        # 添加准确率标注
        for i, model in enumerate(df['Model'].unique()):
            model_data = df[df['Model'] == model]['Accuracy']
            mean_acc = model_data.mean()
            plt.text(i, mean_acc + 0.01, f"{mean_acc:.3f}", ha='center', va='bottom', fontsize=10, color='black')

        # 计算p值并显示
        model_pairs = list(itertools.combinations(df['Model'].unique(), 2))
        y_max = df['Accuracy'].max() + 0.05
        for i, (model1, model2) in enumerate(model_pairs):
            data1 = df[df['Model'] == model1]['Accuracy']
            data2 = df[df['Model'] == model2]['Accuracy']
            stat, p_value = ttest_ind(data1, data2)

            # 显示p值
            x1, x2 = df['Model'].unique().tolist().index(model1), df['Model'].unique().tolist().index(model2)
            y = y_max + i * 0.02
            plt.plot([x1, x1, x2, x2], [y, y + 0.01, y + 0.01, y], lw=1.5, color='black')
            plt.text((x1 + x2) * 0.5, y + 0.01, f"p={p_value:.3e}", ha='center', va='bottom', fontsize=10, color='black')

        plt.title('Model Accuracy Comparison', fontsize=14)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig('model_accuracy_violinplot_with_pvalues.pdf')
        # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default="")
    args = parser.parse_args()

    data_util = DataUtil(args)
    results,all_acc = data_util.read_data()
    df_all_acc = pd.DataFrame(list(all_acc.items()), columns=['drug', 'Accuracy'])
    df_all_acc.to_csv('all_accuracies.csv', index=False)
    data_util.plot_accuracies(results)







