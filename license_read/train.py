import os

import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset.make_data import licenseDataset,train_transform, val_transform
from license_read.cnn_read_nn import Net

#构建数据集
data_path = "../make_license/plate_images/single_yellow"
transform_train = train_transform()
#数据集标签列表
label_list = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "军", "使",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V","W", "X", "Y", "Z"]
data = licenseDataset(root=data_path, transform=transform_train , label_list=label_list)
#数据集批量处理
train_datas= DataLoader(data, batch_size=32, shuffle=True ,num_workers= 0 )
#构建模型
model = Net()
#如果存在模型数据则加载模型
model_save_path = "./model/plate_model/plate_model_1.pth"
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cuda')))
    print("Model loaded successfully.")
#使用GPU
if torch.cuda.is_available():
    model = model.cuda()
#定义损失函数
loss_fn = nn.CrossEntropyLoss()
#定义优化器
opt = optim.Adam(model.parameters(), lr=0.001)
#训练模型


def train(model, train_loader, loss_fn, optimizer, epochs=100):
    model.train()  # 设置模型为训练模式
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = loss_fn(outputs.permute(0,2,1), labels)
            # 清零梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:  # 每10个batch打印一次损失
                print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(train_loader):.4f}')

if __name__ == "__main__":
    train(model, train_datas, loss_fn, opt, epochs=1000)
    # 保存模型
    model_save_path = "../model/plate_model"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(model.state_dict(),os.path.join(model_save_path, f"plate_model.pth"))
    print("Model saved successfully.")