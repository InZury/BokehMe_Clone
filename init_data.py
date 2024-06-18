import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


class InputImageDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.root_dir = root_path
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        data = []

        sub_folders = sorted(os.listdir(self.root_dir))

        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(self.root_dir, sub_folder)

            exr_files = sorted(glob.glob(os.path.join(sub_folder_path, '*.exr')))

            for exr_file in exr_files:
                base_name = os.path.splitext(os.path.basename(exr_file))[0]

                image_path = os.path.join(sub_folder_path, base_name + '.jpg')
                depth_path = os.path.join(sub_folder_path, base_name + '.exr')
                disparity_path = os.path.join(sub_folder_path, base_name + '.jpg')

                # 없는 파일 방지
                if os.path.exists(image_path) and os.path.exists(depth_path) and os.path.exists(disparity_path):
                    data.append({
                        'image': image_path,
                        'depth': depth_path,
                        'disparity': disparity_path
                    })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.data[idx]['image']
        depth_path = self.data[idx]['depth']
        disparity_path = self.data[idx]['disparity']

        # 이미지 불러오기
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = cv2.cvtColor(depth, cv2.IMREAD_GRAYSCALE)

        disparity = cv2.imread(disparity_path, cv2.IMREAD_ANYDEPTH)
        disparity = cv2.cvtColor(disparity, cv2.IMREAD_GRAYSCALE)

        image = transforms.ToTensor()(np.array(image))
        depth = transforms.ToTensor()(np.array(depth))
        disparity = transforms.ToTensor()(np.array(disparity))

        return {'image': image, 'depth': depth, 'disparity': disparity}


# 데이터 세트 및 DataLoader 생성
root_dir = 'data'
dataset = InputImageDataset(root_path=root_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

for batch in dataloader:
    print("image shape:", batch['image'].shape)
    print("depth shape:", batch['depth'].shape)
    print("disparity shape:", batch['disparity'].shape)

    images = batch['image']
    depths = batch['depth']
    disparities = batch['disparity']

    for i in range(len(images)):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(np.transpose(images[i], (1, 2, 0)))
        plt.title('Image')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(np.transpose(depths[i], (1, 2, 0)))
        plt.title('Depth')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(np.transpose(disparities[i], (1, 2, 0)))
        plt.title('Disparity')
        plt.axis('off')

        plt.show()

    break  # 첫 번째 배치만 확인
