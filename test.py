import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import numpy as np
from torch.nn import DataParallel
from testdata import TestCustomDataset
from resnet import CustomNet
import argparse
from tqdm import tqdm



parser = argparse.ArgumentParser(description='predict tile and copy outcome')
parser.add_argument('--df', type=str, default='2700_600_test.csv', help='dataset csv')
args = parser.parse_args()
class_to_folder = {
    0: "ADI",
    1: "BAC",
    2: "DEB",
    3: "LYM",
    4: "MUC",
    5: "MUS",
    6: "NOR",
    7: "STR",
    8: "TUM",
    # 添加更多类别和文件夹映射
}
print(f'{class_to_folder} The tags are mapped')
# df = 'test300.csv'
df = args.df
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print('transform complete! ')
test_dataset = TestCustomDataset(df, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=2700, shuffle=False, num_workers=0)
# import ipdb;ipdb.set_trace()
print('The data is loaded')
resnet = CustomNet(pretrained=False)
resnet.load_state_dict(torch.load('resnet_fold_1_best.pth'))
import ipdb;ipdb.set_trace()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
resnet = DataParallel(resnet)  # 使用 DataParallel 在多个 GPU 上并行训练
resnet.to(device)
resnet.eval()
print('The model is loaded')
result_dir = 'result_tcga'
os.makedirs(result_dir, exist_ok=True)
try:
    with torch.no_grad():
        for inputs, paths in tqdm(test_loader, colour = 'MAGENTA'):
            inputs = inputs.to(device)
            outputs = resnet(inputs)
            _, predicted_classes = torch.max(outputs, 1)
            # import ipdb;ipdb.set_trace()
            # Iterate through each sample in each batch
            for predicted_class, path in zip(predicted_classes, paths):
                predicted_class = predicted_class.item()
                # import ipdb;ipdb.set_trace()
                print(f'{path} 预测结果是: {predicted_class}')
                patient_id = path.split('/')[1]
                filename = path.split('/')[2]
                target_folder = os.path.join(result_dir, patient_id, class_to_folder[predicted_class])
                os.makedirs(target_folder, exist_ok=True)
                shutil.copy(path, os.path.join(target_folder, filename))
                print(f'{path}复制到{target_folder}完成。')
    print('Prediction complete!')
except Exception as e:
    print(e)
    print('Prediction error!')