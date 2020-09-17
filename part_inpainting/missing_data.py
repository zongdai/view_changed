import torch
import torchvision
import torchvision.datasets as datasets
from PIL import Image
import os
import numpy as np

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class MissingData(datasets.DatasetFolder):
    def __init__(self, root, class_name, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, train_flag=True):
        super(MissingData, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = []
        self.loader = loader
        self.class_name = class_name
        self.tran_flag = train_flag
        # print(self.samples)
        # print(self.class_to_idx)
        for item in self.samples:
            if item[1] == self.class_to_idx[class_name]:
                self.imgs.append(item)
                if len(self.imgs) > 600:
                    break

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, mask) where target is class_index of the target class.
        """
        trans = torchvision.transforms.Compose([torchvision.transforms.Resize((256, 512))])
        trans_com = transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256, 512)),torchvision.transforms.ToTensor()])
    
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.tran_flag:
            complete = self.loader(os.path.join('part_complete', self.class_name, path.split('/')[-1]))
            sample_mask = np.array(trans(sample))
            sample_mask = torch.tensor(sample_mask)
            sample_mask = ((sample_mask[:,:,0] == 255) & (sample_mask[:,:,1] == 255) & (sample_mask[:,:,2] == 255)).int()
            # print(path , torch.sum(sample_mask))
            if self.transform is not None:
                sample = self.transform(sample)
                complete = trans_com(complete)
            info = {}
            info['path'] = path
            return sample_mask, sample, complete, info
        else:
            info = {}
            info['path'] = path
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, info


if __name__ == "__main__":
    missing_data = MissingData('part_train_data', '4', transform=torchvision.transforms.Compose([torchvision.transforms.Resize((1024, 256)),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    sample_mask, sample, complete = missing_data[0]
    print(len(missing_data))
    '''
    from torchvision import transforms
    unloader = transforms.ToPILImage()
    image = complete.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save('sample.png')
    
    import cv2
    print(sample_mask)
    cv2.imwrite('miss.png', sample_mask.numpy() * 255)
    '''



