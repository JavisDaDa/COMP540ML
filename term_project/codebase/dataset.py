import os
import pandas as pd
from PIL import Image
from utils import createclass2label
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import class_list_path, BATCH_SIZE, train_label_dir, valid_label_dir, TRAIN_SAMPLE, VALID_SAMPLE,\
    test_label_dir, norm_mean, norm_std, train_dir, valid_dir, test_dir


class FoodDataset(Dataset):
    def __init__(self, data_dir, data_df, transform=None, class_list_dir=class_list_path):
        self.food_label = createclass2label(class_list_dir)
        self.data_dir = data_dir
        self.data_info = data_df
        self.transform = transform

    def __getitem__(self, index):
        path_img = os.path.join(self.data_dir, self.data_info.iloc[index]['img_name'])
        label = self.data_info.iloc[index]['label']
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)


def get_dataset():
    train_df = pd.read_csv(train_label_dir)
    valid_df = pd.read_csv(valid_label_dir)
    train_data_df = train_df.sample(TRAIN_SAMPLE)
    valid_data_df = valid_df.sample(VALID_SAMPLE)
    test_data_df = pd.read_csv(test_label_dir)
    print(f'Found {train_data_df.shape[0]} train images in {len(train_data_df.label.unique())} classes')
    print(f'Found {valid_data_df.shape[0]} valid images in {len(valid_data_df.label.unique())} classes')
    print(f'Found {test_data_df.shape[0]} test images in {len(test_data_df.label.unique())} classes')

    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomApply(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomVerticalFlip(),
             transforms.RandomAffine(30)]),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    train_data = FoodDataset(data_dir=train_dir, data_df=train_data_df, transform=train_transform)
    valid_data = FoodDataset(data_dir=valid_dir, data_df=valid_data_df, transform=valid_transform)
    test_data = FoodDataset(data_dir=test_dir, data_df=test_data_df, transform=test_transform)

    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

    return train_loader, valid_loader, test_loader