import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class customDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        print("class_data",self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        print("class_data", self.class_to_idx)
        self.images = self.get_images()

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_images(self):
        images = []
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                images.append((img_path, cls_idx))
        #print("class_data", images)
        return images


if __name__ == '__main__':
    # 数据集根目录
    root_dir = 'D:\\Resnet\\data\\catANDdog\\test'

    # 定义图像预处理和增强操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    # 创建数据集实例
    dataset = customDataset(root_dir, transform=transform)
    # 创建数据加载器
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for images, labels in dataloader:
        # 这里执行模型训练或推理等操作
        print("Batch size:", images.size(0))
        print("Image shape:", images.size()[1:])
        print("Label shape:", labels.size())
        break  # 打印一个 batch 后停止循环
