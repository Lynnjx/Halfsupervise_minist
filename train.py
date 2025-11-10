from torch.utils.data import DataLoader, Subset
from model import SimpleCNN
import torch
import torch.nn as nn
import torch.optim as optim
import os

def train_supervised_model(labeled_indices, epochs=20, device=None, train_dataset=None, test_dataset=None):
    """训练监督基线模型"""
    
    # 创建数据加载器
    labeled_dataset = Subset(train_dataset, labeled_indices)
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 初始化模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练记录
    train_losses = []
    test_accuracies = []
    
    best_accuracy = 0.0
    os.makedirs('pth', exist_ok=True)
    save_path = os.path.join('pth', f'best_supervise_label_model_{len(labeled_indices)}.pth')
    
    print(f"训练监督模型 (标注样本: {len(labeled_indices)})")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(labeled_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 评估阶段
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(labeled_loader)
        
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
            
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f'Epoch {epoch+1}: 新的最优模型已保存，准确率: {accuracy:.2f}%')
    
    return model, train_losses, test_accuracies

def pseudo_label_training(labeled_indices, unlabeled_indices, epochs=30, threshold=0.9, device=None, train_dataset=None, test_dataset=None):
    """伪标签半监督学习"""
    
    # 创建数据加载器
    labeled_dataset = Subset(train_dataset, labeled_indices)
    unlabeled_dataset = Subset(train_dataset, unlabeled_indices)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # 初始化模型
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    os.makedirs('pth', exist_ok=True)
    save_path = os.path.join('pth', f'best_pseudo_label_model_{len(labeled_indices)}.pth')
    
    print(f"伪标签训练 (标注: {len(labeled_indices)}, 无标签: {len(unlabeled_indices)})")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 同时迭代标注和无标签数据
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        num_batches = max(len(labeled_loader), len(unlabeled_loader))
        
        for batch_idx in range(num_batches):
            # 处理标注数据
            try:
                labeled_data, labeled_target = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_data, labeled_target = next(labeled_iter)
            
            # 处理无标签数据并生成伪标签
            try:
                unlabeled_data, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_data, _ = next(unlabeled_iter)
            
            # 合并数据
            data = torch.cat([labeled_data, unlabeled_data], 0)
            data = data.to(device)
            labeled_target = labeled_target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # 分割输出：标注部分和无标签部分
            labeled_outputs = outputs[:len(labeled_data)]
            unlabeled_outputs = outputs[len(labeled_data):]
            
            # 计算标注数据的损失
            supervised_loss = criterion(labeled_outputs, labeled_target)
            
            # 为无标签数据生成伪标签
            pseudo_probs = torch.softmax(unlabeled_outputs, dim=1)
            pseudo_conf, pseudo_labels = torch.max(pseudo_probs, dim=1)
            
            # 选择高置信度样本
            high_conf_mask = pseudo_conf > threshold
            if high_conf_mask.sum() > 0:
                high_conf_outputs = unlabeled_outputs[high_conf_mask]
                high_conf_pseudo_labels = pseudo_labels[high_conf_mask]
                unsupervised_loss = criterion(high_conf_outputs, high_conf_pseudo_labels)
            else:
                unsupervised_loss = 0
            
            # 总损失 = 监督损失 + 无监督损失
            total_loss = supervised_loss + unsupervised_loss
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = running_loss / num_batches
        
        train_losses.append(avg_loss)
        test_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
            
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f'Epoch {epoch+1}: 新的最优模型已保存，准确率: {accuracy:.2f}%')
        
    
    return model, train_losses, test_accuracies