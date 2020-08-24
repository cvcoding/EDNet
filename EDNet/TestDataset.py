from torch.utils.data import Dataset
from PIL import Image

def default_loader(path):
    # return Image.open(path)
    return Image.open(path).convert('RGB')

class TestDataset(Dataset):
    def __init__(self, label_list, transform=None, target_transform=None, loader=default_loader):
        imgs = []
        for index, row in label_list.iterrows():
            imgs.append((row['img_path'], row['label']))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        filename, name = self.imgs[index]
        img = self.loader(filename)
        if self.transform is not None:
            img = self.transform(img)
        return img, name

    def __len__(self):
        return len(self.imgs)