import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from utils.datasets import *
import argparse
import sys

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def get_model(model_name,nc):
    if model_name == 'resnet50':
        from net.resnet import resnet50
        model = resnet50(nc)

    elif model_name == 'resnet18':
        from net.resnet import resnet18
        model = resnet18(nc)

    elif model_name == 'resnet34':
        from net.resnet import resnet34
        model = resnet34(nc)

    elif model_name == 'resnet101':
        from net.resnet import resnet101
        model = resnet101(nc)

    elif model_name == 'resnet152':
        from net.resnet import resnet152
        model = resnet152(nc)

    elif model_name == 'vgg19':
        from net.VGG19 import vgg19
        model = vgg19(nc)

    elif model_name == 'SimpleCNN':
        from net.SimpleCNN import simplecnn
        model = simplecnn(nc)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return model

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train VGG19 model on custom dataset')
    parser.add_argument('--net', type=str, default='resnet18', metavar='NAME',help='model name')
    parser.add_argument('--epoch', type=int, default=100, metavar='epoch', help='epoch')
    parser.add_argument('--batchsize', type=int, default=128, metavar='batchsize', help='batchsize')
    parser.add_argument('--lr', type=float, default=0.1, metavar='nc', help='learnrate')
    parser.add_argument('--nc', type=int, default=2, metavar='nc', help='name class')
    parser.add_argument('--datapath', type=str, default='D:\\Resnet\\data\\catANDdog', metavar='PATH',help='path to the training dataset')
    #parser.add_argument('--test_path', type=str, default='D:\\pytorch\\data\\catANDdog\\test', metavar='PATH',help='path to the testing dataset')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()

    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转（在 -15 度到 +15 度之间）
        transforms.ToTensor(),           # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    train_data = customDataset(args.datapath+'\\train', transform=transform)
    test_data = customDataset(args.datapath+'\\test', transform=transform)

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True)
    #建立模型
    model = get_model(args.net,args.nc)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    #优化器
    #learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
    #设置参数
    total_train_step = 0
    total_test_step = 0
    #epoch = args.epoch

    writer = SummaryWriter("logs")

    for i in range(args.epoch):
        print("------第{}轮训练开始------".format(i+1))
        for data in train_dataloader:
            (imgs, targets)= data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("训练次数：{}，loss：{}".format(total_train_step, loss))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        scheduler.step()
        total_test_loss = 0
        total_acc = 0
        with torch.no_grad():
            for data in test_dataloader:
                (imgs, targets) = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss = total_test_loss + loss.item()
                acc = (outputs.argmax(1) == targets).sum()
                total_acc = total_acc + acc
        print("整体测试集上的正确率：{}".format(total_acc/test_data_size))
        print("测试数据集上的loss：{}".format(total_test_loss/test_data_size))
        writer.add_scalar("test_loss", total_test_loss/test_data_size, total_test_step)
        writer.add_scalar("test_acc", total_acc/test_data_size, total_test_step)
        total_test_step = total_test_step + 1
        is_best = 0
        if is_best < total_acc/test_data_size:
            is_best = total_acc/test_data_size
            print(is_best)
            torch.save(model, "weights/best.pth")

    torch.save(model, "weights/last.pth")
    writer.close()

if __name__ == "__main__":
    main()
