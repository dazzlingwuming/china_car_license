#使用vgg模型来做图像识别
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sympy.simplify.cse_opts import sub_pre
from torchvision.models import vgg16


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = models.vgg16_bn(pretrained=True)
        #分类个数不同，需要修改最后一层
        # 冻结卷积层参数
        for param in self.vgg.features.parameters():
            param.requires_grad = False
        # 修改最后一层全连接层
        self.vgg.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 8*67)  # 8个字符，每个字符35个类别,
        )

    def forward(self, img):
        x = self.vgg(img)
        x = x.view(x.size(0), 8, 67)
        return x

if __name__ == "__main__":
    # 测试模型
    model = Net()
    model.eval()
    # 创建一个随机输入张量，模拟一张图片
    input_tensor = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, height=224, width=224
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 应该是 (1, 8, 65)