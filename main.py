import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.semi_supervised import LabelPropagation
import seaborn as sns
import pandas as pd
from train import train_supervised_model, pseudo_label_training

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 或者使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False 

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")


def create_labeled_unlabeled_splits(dataset, labels_per_class=20):
    """
    创建标注和无标签数据划分
    labels_per_class: 每类的标注样本数量 (20, 50, 100)
    """
    # 按类别组织索引
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # 选择标注样本
    labeled_indices = []
    unlabeled_indices = []
    
    for class_id, indices in class_indices.items():
        # 随机选择标注样本
        selected = np.random.choice(indices, size=labels_per_class, replace=False)
        labeled_indices.extend(selected)
        
        # 剩余作为无标签
        unlabeled = list(set(indices) - set(selected))
        unlabeled_indices.extend(unlabeled)
    
    return labeled_indices, unlabeled_indices

def evaluate_model(model, test_loader):
    """全面评估模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    return accuracy, f1, cm, all_preds, all_targets

def run_complete_experiment():
    """运行完整的实验"""
    results = {
        'scenario': [],
        'method': [],
        'accuracy': [],
        'f1_score': [],
        'labeled_samples': []
    }
    
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    for scenario_name, labels_per_class in label_scenarios.items():
        print(f"\n{'='*50}")
        print(f"实验场景: {scenario_name} (每类{labels_per_class}个标注样本)")
        print(f"{'='*50}")
        
        labeled_indices = splits[scenario_name]['labeled']
        unlabeled_indices = splits[scenario_name]['unlabeled']
        
        # 1. 监督基线
        print("\n1. 训练监督基线模型...")
        supervised_model, _, _ = train_supervised_model(labeled_indices, epochs=20,device=device,
                                                        train_dataset=train_dataset, test_dataset=test_dataset)
        sup_accuracy, sup_f1, sup_cm, _, _ = evaluate_model(supervised_model, test_loader)
        
        results['scenario'].append(scenario_name)
        results['method'].append('监督基线')
        results['accuracy'].append(sup_accuracy)
        results['f1_score'].append(sup_f1)
        results['labeled_samples'].append(len(labeled_indices))
        
        print(f"监督基线 - 准确率: {sup_accuracy:.4f}, F1分数: {sup_f1:.4f}")
        
        # 2. 伪标签方法
        print("\n2. 训练伪标签模型...")
        pseudo_model, _, _ = pseudo_label_training(labeled_indices, unlabeled_indices, epochs=30, device=device,
                                                    train_dataset=train_dataset, test_dataset=test_dataset)
        pseudo_accuracy, pseudo_f1, pseudo_cm, _, _ = evaluate_model(pseudo_model, test_loader)
        
        results['scenario'].append(scenario_name)
        results['method'].append('伪标签')
        results['accuracy'].append(pseudo_accuracy)
        results['f1_score'].append(pseudo_f1)
        results['labeled_samples'].append(len(labeled_indices))
        
        print(f"伪标签方法 - 准确率: {pseudo_accuracy:.4f}, F1分数: {pseudo_f1:.4f}")
        
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    return results_df

def visualize_results(results_df):
    """可视化实验结果"""
    
    # 1. 准确率对比图
    plt.figure(figsize=(12, 8))
    
    # 准确率对比
    plt.subplot(2, 2, 1)
    scenarios = results_df['scenario'].unique()
    methods = results_df['method'].unique()
    
    bar_width = 0.25
    x = np.arange(len(scenarios))
    
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        accuracies = [method_data[method_data['scenario'] == s]['accuracy'].values[0] 
                     if not method_data[method_data['scenario'] == s].empty else 0 
                     for s in scenarios]
        plt.bar(x + i * bar_width, accuracies, bar_width, label=method)
    
    plt.xlabel('标注场景')
    plt.ylabel('准确率')
    plt.title('不同方法在各场景下的准确率')
    plt.xticks(x + bar_width, scenarios, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. F1分数对比
    plt.subplot(2, 2, 2)
    for i, method in enumerate(methods):
        method_data = results_df[results_df['method'] == method]
        f1_scores = [method_data[method_data['scenario'] == s]['f1_score'].values[0] 
                    if not method_data[method_data['scenario'] == s].empty else 0 
                    for s in scenarios]
        plt.bar(x + i * bar_width, f1_scores, bar_width, label=method)
    
    plt.xlabel('标注场景')
    plt.ylabel('Macro F1分数')
    plt.title('不同方法在各场景下的F1分数')
    plt.xticks(x + bar_width, scenarios, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 性能提升分析
    """ plt.subplot(2, 2, 3)
    baseline_acc = results_df[results_df['method'] == '监督基线']['accuracy'].values
    pseudo_acc = results_df[results_df['method'] == '伪标签']['accuracy'].values
    
    improvement = (pseudo_acc - baseline_acc) / baseline_acc * 100
    
    plt.bar(scenarios, improvement, color='green', alpha=0.7)
    plt.xlabel('标注场景')
    plt.ylabel('性能提升 (%)')
    plt.title('伪标签相比基线的性能提升')
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(improvement):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center') """
    
    # 4. 标注样本数量与准确率关系
    
    plt.subplot(2, 2, 3)
    for method in methods:
        method_data = results_df[results_df['method'] == method]
        if not method_data.empty:
            plt.plot(method_data['labeled_samples'], method_data['accuracy'], 
                    marker='o', label=method, linewidth=2)
    
    plt.xlabel('标注样本数量')
    plt.ylabel('准确率')
    plt.title('标注数据量与性能关系')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('semi_supervised_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch_size', type=int, default=64, help='输入批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--method', type=str, default='supervised', help='选择的方法: supervised, pseudo_label')
    
    args = parser.parse_args()
    
    # 创建不同标注量的数据划分
    label_scenarios = {
        '20_per_class': 20,
        '50_per_class': 50, 
        '100_per_class': 100
    }

    splits = {}
    for scenario, labels_per_class in label_scenarios.items():
        labeled_idx, unlabeled_idx = create_labeled_unlabeled_splits(train_dataset, labels_per_class)
        splits[scenario] = {
            'labeled': labeled_idx,
            'unlabeled': unlabeled_idx
        }
        print(f"{scenario}: 标注样本 {len(labeled_idx)}, 无标签样本 {len(unlabeled_idx)}")
    
    # 运行实验
    print("开始半监督学习实验...")
    results_df = run_complete_experiment()
    
    # 可视化结果
    final_results = visualize_results(results_df)

    # 显示详细结果表
    print("\n" + "="*60)
    print("实验最终结果")
    print("="*60)
    print(final_results.to_string(index=False))