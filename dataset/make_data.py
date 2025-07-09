import torch
from torchvision import transforms , datasets
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder

datasets.MNIST
class licenseDataset(Dataset):
    def __init__(self, root, transform=None,label_list = [] ):#root为数据集路径，transform为数据预处理方式,并且需要传入标签列表用作数据编辑
        super(licenseDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.label_list = label_list
        self.imgfile_labels = self.create_img_for_labels()


    def __getitem__(self, idx):
        image ,labels = self.imgfile_labels[idx]#取出对应的图片和标签
        image_path = os.path.join(self.root, image)  # 拼接图片的完整路径
        image = Image.open(image_path).convert('RGB')  # 打开图片并转换为RGB格式
        # 如果需要对图片进行预处理，可以在这里添加
        if self.transform:
            image = self.transform(image)
        else:
            raise ValueError("未指定 transform，无法处理图片。")
        # 将标签转换为tensor格式
        labels = torch.tensor(labels, dtype=torch.long)
        return image, labels  # 返回图片和标签

    def __len__(self):
        return len(os.listdir(self.root))

    def create_img_for_labels(self):#将路径和标签对应存储并转换为tensor格式
        labels = []
        for file in os.listdir(self.root):
            if file.endswith('.jpg') or file.endswith('.png'):
                #取出标签的字符串部分
                label = file.split('_')[1].split(".")[0]
                #如果标签长度为7，需要扩到8位，最后一位为0、
                if len(label) == 7:
                    label += '0'
                elif len(label) != 8:
                    raise ValueError(f"标签长度错误: {label}，应为8位。")
                #将标签转换为数字以及tensor格式
                label = [file,[self.label_list.index(char) for char in label if char in self.label_list ]]
                labels.append(label)
        return labels

def train_transform():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 先统一缩放
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪到224x224
        transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色扰动
        transforms.RandomRotation(degrees=10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform

def val_transform():
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图片大小
        transforms.ToTensor(),  # 将图片转换为Tensor格式
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化处理
    ])
    return val_transform
# # 直接对数据目录进行划分
# def split_dataset(root , train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
#     all_files = [f for f in os.listdir(root) if f.endswith('.jpg') or f.endswith('.png')]
#     total_size = len(all_files)
#     train_size = int(total_size * train_ratio)
#     val_size = int(total_size * val_ratio)
#     test_size = int(total_size * test_ratio) # 剩余部分作为测试集
#     train_dataset, val_dataset, test_dataset = random_split(all_files, [train_size, val_size, test_size])
#     return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    label_list = ["京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                  "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "军", "使",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V","W", "X", "Y", "Z"]
    transform = train_transform()  # 获取训练集的预处理方式
    data = licenseDataset(root=r"D:\github\china_car_license\make_license\plate_images\single_yellow",label_list=label_list, transform=transform)
    # for idx, item in enumerate(data):
    #     print(item[1].shape)
    #     if item[1].shape[0] == 6:
    #         #需要错误标签对应列表的值
    #         list1=[]
    #         for i in range(6):
    #             list1.append(label_list[item[1][i].item()])
    #         print(list1)
    # 获取第一个样本
    # 创建数据加载器
    dataloader = DataLoader(data, batch_size=10, shuffle=True, num_workers=4)
    for images, labels in dataloader:
        print("Batch of images shape:", images.shape)  # 打印批次图像的形状
        print("Batch of labels shape:", labels.shape)  # 打印批次标签的形状

