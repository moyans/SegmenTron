"""DocUnet Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from .seg_data_base import SegmentationDataset


class DocSegmentation(SegmentationDataset):
    NUM_CLASS = 2

    def __init__(self, root='datasets/docUnet', split='train', mode=None, transform=None, **kwargs):
        super(DocSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        self.root = root

        if split == 'train':
            _split_f = os.path.join(self.root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(self.root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                name = os.path.splitext(line)[0]
                _image = os.path.join(self.root, split, line.rstrip('\n'))
                # print("_image: ", _image)
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(self.root, split, name + "_dst.npy")
                # print("_mask:", _mask)
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), self.root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = np.load(self.masks[index])

        img = self._img_transform(img)
        # toTensor
        if self.transform is not None:
            img = self.transform(img)

        mask = transforms.ToTensor()(mask)
        return img, mask, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _img_transform(self, img):
        return np.array(img)

    def _docunet_mask_transform(self, img):
        return transforms.ToTensor()

    @property
    def classes(self):
        """Category names."""
        return ('Default',)


if __name__ == '__main__':
    dataset = DocSegmentation()