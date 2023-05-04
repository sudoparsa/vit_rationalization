import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
import numpy as np
import h5py
import einops
import os


class CBMNIST(Dataset):
    def __init__(self, phase, args, transform = Compose ([Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])):
        path = args.data_path
        if phase == 'train':
            self.images = torch.Tensor(np.load(os.path.join(path, 'X_train.npy')))
            labels = torch.Tensor(np.load(os.path.join(path, 'y_train.npy')))
            self.labels = nn.functional.one_hot(labels.long(), num_classes=args.num_classes).type(torch.FloatTensor)
            self.envs = torch.Tensor(np.load(os.path.join(path, 'env_train.npy')))

        elif phase == 'val':
            self.images = torch.Tensor(np.load(os.path.join(path, 'X_val.npy')))
            labels = torch.Tensor(np.load(os.path.join(path, 'y_val.npy')))
            self.labels = nn.functional.one_hot(labels.long(), num_classes=args.num_classes).type(torch.FloatTensor)
            self.envs = torch.Tensor(np.load(os.path.join(path, 'env_val.npy')))

        elif phase == 'test':
            self.images = torch.Tensor(np.load(os.path.join(path, 'X_test.npy')))
            labels = torch.Tensor(np.load(os.path.join(path, 'y_test.npy')))
            self.labels = nn.functional.one_hot(labels.long(), num_classes=args.num_classes).type(torch.FloatTensor)
            self.envs = torch.Tensor(np.load(os.path.join(path, 'env_test.npy')))

        self.transform = transform
        self.images = self.images/255
        self.images = einops.rearrange(self.images, 'b w h c -> b c w h')


    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index], self.envs[index]

    def __len__(self):
        return len(self.images)


class COCO_ON_COLOURS(Dataset):
    def __init__(self, phase, args, transform=Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])):
        path = args.data_path
        if phase == 'train':
            h5p = h5py.File(os.path.join(path, 'train.h5py'), 'r')
            self.images, labels, envs = torch.from_numpy(h5p['images'][:]), torch.from_numpy(h5p['y'][:]).type(
                torch.LongTensor), torch.from_numpy(h5p['e'][:]).type(torch.LongTensor)

            self.labels = nn.functional.one_hot(labels, num_classes=args.num_classes).type(torch.FloatTensor)
            self.envs = nn.functional.one_hot(envs)

        elif phase == 'val':
            h5p = h5py.File(os.path.join(path, 'val.h5py'), 'r')
            self.images, labels = torch.from_numpy(h5p['images'][:]), torch.from_numpy(h5p['y'][:]).type(
                torch.LongTensor)

            self.labels = nn.functional.one_hot(labels, num_classes=args.num_classes).type(torch.FloatTensor)
            self.envs = nn.functional.one_hot(torch.zeros(self.images.shape[0]).type(torch.LongTensor))


        elif phase == 'test':
            h5p = h5py.File(os.path.join(path, 'test.h5py'), 'r')
            self.images, labels = torch.from_numpy(h5p['images'][:]), torch.from_numpy(h5p['y'][:]).type(
                torch.LongTensor)

            self.labels = nn.functional.one_hot(labels, num_classes=args.num_classes).type(torch.FloatTensor)
            self.envs = nn.functional.one_hot(torch.zeros(self.images.shape[0]).type(torch.LongTensor))


        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index], self.envs[index]

    def __len__(self):
        return len(self.images)


class WATERBIRDS(Dataset):
    def __init__(self, phase, args, transform = Compose([Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])):
        path = args.data_path
        if phase == 'train':
            self.images = torch.Tensor(np.load(os.path.joinpath+'X_train.npy'))
            labels = torch.Tensor(np.load(os.path.joinpath+'y_train.npy'))
            self.labels = nn.functional.one_hot(labels, num_classes=args.num_classes).type(torch.FloatTensor)
            envs = torch.Tensor(np.load(os.path.joinpath+'env_train.npy'))
            self.envs = nn.functional.one_hot(envs.long())

        elif phase == 'val':
            self.images = torch.Tensor(np.load(os.path.join(path, 'X_val.npy')))
            labels = torch.Tensor(np.load(os.path.join(path, 'y_val.npy')))
            self.labels = nn.functional.one_hot(labels, num_classes=args.num_classes).type(torch.FloatTensor)
            envs = torch.Tensor(np.load(os.path.join(path, 'env_val.npy')))
            self.envs = nn.functional.one_hot(envs.long())

        elif phase == 'test':
            self.images = torch.Tensor(np.load(os.path.join(path, 'X_test.npy')))
            labels = torch.Tensor(np.load(os.path.join(path, 'y_test.npy')))
            self.labels = nn.functional.one_hot(labels, num_classes=args.num_classes).type(torch.FloatTensor)
            envs = torch.Tensor(np.load(os.path.join(path, 'env_test.npy')))
            self.envs = nn.functional.one_hot(envs.long())

        self.transform = transform


    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index], self.envs[index]

    def __len__(self):
        return len(self.images)


def get_dataset (args, phase, transform = Compose([Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])):
    print(args.img_size)
    if args.dataset_name=='CBMNIST':
        dataset = CBMNIST(args= args, phase=phase, transform=transform)
    elif args.dataset_name=='COCOCOLOURS':
        dataset = COCO_ON_COLOURS(args= args, phase=phase, transform=transform)
    elif args.dataset_name=='WATERBIRDS':
        dataset = WATERBIRDS(args= args, phase=phase, transform=transform)
    return dataset

def get_loader (args, phase, transform = Compose([Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))])):
    dataset = get_dataset (args, phase, transform = transform)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)