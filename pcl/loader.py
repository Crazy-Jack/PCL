from PIL import ImageFilter
import random
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ImageFolderInstanceClass(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index, target

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index



class DynamicLabelDataset(Dataset):
    '''
    torch dataset
    '''
    def __init__(self, df, data_path, gran_lvl, transform=None):
        """
        note: clean version of dynamic label dataset
        param:
            df: a panda DataFrame which contains the path of the imagenet data path and label
            data_path: root name.
                Consider the data is stored in xxxx/imagenet_unzip/n01582220/n01582220_4784.JPEG
                data_path should be 'xxxx/imagenet_unzip/'
                each row of df['path'] will be 'n01582220/n01582220_4784.JPEG'
                each row of df['class'] will be the numerical representation of true class for each image, in this case 86 (out of 100, rank mean nothing)
            gran_lvl: choose from ['class', 'simclr', f'label_gran_{gran_lvl}'], they should be in the header of df
            transform: instance of torchvision.transform
        """

        super(DynamicLabelDataset, self).__init__()
        self.data_path = data_path
        self.gran_lvl = gran_lvl
        # assign latent class column name
        if self.gran_lvl == 'class':
            self.gran_lvl_label_name = 'class'
        elif self.gran_lvl == 'simclr':
            self.gran_lvl_label_name = 'path'
        else:
            self.gran_lvl_label_name = 'label_gran_{}'.format(gran_lvl)

        # clean up df
        if -1 in df.index:
            df = df.drop([-1])
            print("drop -1 row")
        if '-1' in df.index:
            df = df.drop(['-1'])
            print("drop '-1' row")
        print(df)

        # store df
        self.df = df
        # store transform
        self.transform = transform

        # processing latent class to continuous unique mapping
        unique_class = np.unique(self.df[self.gran_lvl_label_name].to_numpy())
        print(f"unique class {unique_class}")
        # check if the unique_class have any discontiouity
        print(f"unique class {len(unique_class)}")
        assert len(unique_class) == max(unique_class) + 1, f"max unique_class should be len(unique_class)-1={len(unique_class)-1} but get {max(unique_class)}"
        for i in range(len(unique_class)):
            assert i in unique_class, f"{i} not in unique_class: {unique_class}"

        self.num_class = unique_class.shape[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        '''
        return:
            img: PIL object
            index: scalar
            lbl: scalar
        '''
        img = Image.open(os.path.join(self.data_path, self.df.iloc[index]['path']))
        if not img.mode == 'RGB':
            img = img.convert("RGB")

        # labels
        if self.gran_lvl == '-1':
            lbl = -1
        else:
            lbl = int(self.df.iloc[index][self.gran_lvl_label_name])

        if self.transform:
            img = self.transform(img)

        return img, index, lbl

