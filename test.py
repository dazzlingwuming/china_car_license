import torch

from license_read import cnn_read_nn

if __name__ == "__main__":
    # 数据集标签列表
    label_list = ["京", "沪", "粤", "苏", "浙", "川", "鲁", "豫", "冀", "晋", "辽", "吉", "黑", "皖", "闽", "赣", "贵",
                  "云", "陕", "甘", "青", "琼", "渝", "津", "蒙", "宁", "新", "藏", "港", "湘", "桂",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                  "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    # 测试模型加载
    model = cnn_read_nn.Net()
    model.load_state_dict(torch.load("./model/plate_model/plate_model.pth" , map_location=torch.device('cpu')))
    model1 = cnn_read_nn.Net()
    model1.load_state_dict(torch.load("./model/plate_model/plate_mode_1.pth" , map_location=torch.device('cpu')))
    model.eval()
    model1.eval()
    #加载一张图片
    from PIL import Image
    from dataset.make_data import val_transform
    imagee = Image.open(r"D:\github\china_car_license\make_license\plate_images\single_blue\001_京B2Q975.jpg").convert("RGB")
    transform = val_transform()
    image = transform(imagee).unsqueeze(0)  # 添加batch维度
    result = model(image)
    result1 = model1(image)
    #需要将结果转换为标签
    result = result.squeeze(0)  # 去掉batch维度
    result1 = result1.squeeze(0)  # 去掉batch维度
    #每一个字符的预测结果
    predicted_labels = []
    for i in range(result.size(0)):
        char_scores = result[i]  # 获取每个字符的预测分数
        predicted_index = char_scores.argmax().item()  # 获取最大分数的索引
        predicted_label = label_list[predicted_index]  # 转换为标签
        predicted_labels.append(predicted_label)
    for i in range(result1.size(0)):
        char_scores = result1[i]
        predicted_index = char_scores.argmax().item()
        predicted_label1 = label_list[predicted_index]
        predicted_labels.append(predicted_label1)
    # 打印预测结果
    print("Predicted labels:", predicted_labels)
