from torchvision import transforms as T
from torch.utils.data import Dataset
from PIL import Image

class SSDGDataset(Dataset):
    def __init__(self, data, transforms=None, train=True):
        self.train = train
        self.features = data[0]
        self.labels = data[1]

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img = self.features[idx]
        img = Image.fromarray(img)
        img = self.transforms(img)
        label = self.labels[idx]
        return img, label