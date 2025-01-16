import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from utils.datasets import *
import argparse
import sys
from torch.optim.lr_scheduler import StepLR

def get_model(model_name, nc):
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
    parser = argparse.ArgumentParser(description='Train VGG19 model on custom dataset')
    parser.add_argument('--net', type=str, default='resnet18', metavar='NAME', help='model name')
    parser.add_argument('--epoch', type=int, default=100, metavar='epoch', help='epoch')
    parser.add_argument('--batchsize', type=int, default=128, metavar='batchsize', help='batchsize')
    parser.add_argument('--lr', type=float, default=0.1, metavar='nc', help='learnrate')
    parser.add_argument('--nc', type=int, default=2, metavar='nc', help='name class')
    parser.add_argument('--datapath', type=str, default='D:\\Resnet\\data\\catANDdog', metavar='PATH', help='path to the training dataset')
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = customDataset(args.datapath+'\\train', transform=transform)
    test_data = customDataset(args.datapath+'\\test', transform=transform)

    train_data_size = len(train_data)
    test_data_size = len(test_data)

    print("训练数据集的长度为：{}".format(train_data_size))
    print("测试数据集的长度为：{}".format(test_data_size))

    train_dataloader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batchsize, shuffle=True)

    model = get_model(args.net, args.nc)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    total_train_step = 0
    total_test_step = 0

    writer = SummaryWriter("logs")

    for epoch in range(args.epoch):
        print("------第{}轮训练开始------".format(epoch + 1))
        model.train()
        total_train_loss = 0
        correct_train_predictions = 0
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_step += 1

            total_train_loss += loss.item()
            correct_train_predictions += (outputs.argmax(1) == targets).sum().item()

            if total_train_step % 10 == 0:
                print("训练次数：{}，loss：{}".format(total_train_step, loss))
                writer.add_scalar("Loss/train", loss.item(), total_train_step)

        scheduler.step()

        train_accuracy = correct_train_predictions / train_data_size
        avg_train_loss = total_train_loss / len(train_dataloader)
        print("训练集上的正确率：{}".format(train_accuracy))
        writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

        model.eval()
        total_test_loss = 0
        correct_test_predictions = 0
        with torch.no_grad():
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                correct_test_predictions += (outputs.argmax(1) == targets).sum().item()

        test_accuracy = correct_test_predictions / test_data_size
        avg_test_loss = total_test_loss / len(test_dataloader)
        print("整体测试集上的正确率：{}".format(test_accuracy))
        print("测试数据集上的loss：{}".format(avg_test_loss))
        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        is_best = 0
        if is_best < test_accuracy:
            is_best = test_accuracy
            print("保存最佳模型，准确率：{}".format(is_best))
            torch.save(model.state_dict(), "weights/best.pth")

    torch.save(model.state_dict(), "weights/last.pth")
    writer.close()

if __name__ == "__main__":
    main()
