"""
Implementation of Subset-ImageNet for continual learning tasks. Please note that this dataset
consists of 100 randomly chosen classes with taken 1993 seed.

"""

import torch
import numpy as np
import random  # may be useful for validation data split
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageOps

from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import ImageFolder


class SubsetImageNet(Dataset):


    def __init__(self, path: str, 
                use_one_hot: bool = False,
                use_data_augmentation: bool = False, 
                validation_size: int = 100,
                task_id: int = 0,
                setting: int = 4,
                batch_size: int = 16):
        
        assert validation_size <= 250

        super().__init__()

        path = f"{path}/seed_1993_subset_100_imagenet/data"

        self._data = dict()
        self._path = path
        self._train_path = f"{path}/train"
        self._test_path = f"{path}/val"
        self._labels = np.arange(100)
        self._use_one_hot = use_one_hot
        self._use_data_augmentation = use_data_augmentation
        self._validation_size = validation_size
        self._setting = setting
        self._batch_size = batch_size
        self._data["in_shape"] = [64, 64, 3]

        self._test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Lambda(lambda x: torch.permute(x, (1, 2, 0))),
            ]
        )
        
        if self._use_data_augmentation:
            self._train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ImageNetPolicy(),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.Lambda(lambda x: torch.permute(x, (1, 2, 0))),
                ]
            )
        else:
            self._train_transform = self._test_transform


        if self._setting == 1:
            self._data["num_classes"] = 50
            self._data["num_incr_classes"] = 10
        elif self._setting == 2:
            self._data["num_classes"] = 50
            self._data["num_incr_classes"] = 5
        elif self._setting == 3:
            self._data["num_classes"] = 40
            self._data["num_incr_classes"] = 3
        elif self._setting == 4:
            self._data["num_classes"] = 20
            self._data["num_incr_classes"] = 20
        else:
            raise(NotImplemented)

        self.train_data, self.val_data = self._get_task(task_id = task_id, mode = 'train')
        self.test_data                 = self._get_task(task_id = task_id, mode = 'test')

        self.num_train_samples = len(self.train_data)
        self.num_val_samples   = len(self.val_data)
        self.num_test_samples  = len(self.test_data)


    def _get_task(self, task_id: int = 0, mode: str = 'train'):
        assert task_id >= 0
        assert mode in ['train', 'test']
        MAX_NUM_CLASSES = self._data["num_classes"]
        INCR_NUM_CLASSES = self._data["num_incr_classes"]

        if self._use_one_hot:
            target_transform = lambda x: self._target_to_one_hot(x, task_id=task_id)
        else:
            target_transform = lambda x: self._calculate_modulo(x, task_id=task_id)

        if mode == 'train':
            dataset = ImageFolder(root=self._train_path, transform=self._train_transform, target_transform=target_transform)
        elif mode == 'test':
            dataset = ImageFolder(root=self._test_path, transform=self._test_transform, target_transform=target_transform)

        if task_id == 0:
            curr_task_labels = self._labels[task_id*MAX_NUM_CLASSES: (task_id+1)*MAX_NUM_CLASSES]
        else:
            curr_task_labels = self._labels[MAX_NUM_CLASSES + (task_id-1)*INCR_NUM_CLASSES : MAX_NUM_CLASSES + task_id*INCR_NUM_CLASSES]

        if mode == 'train': 
            print(f'Order of classes in the current task: {curr_task_labels}')
            print('Preparing task images and labels...')

            val_map = {target: [] for target in curr_task_labels}
            val_map['all'] = []

            # per each label in task, gather first self._validation_size number of instances
            for j, label in enumerate(dataset.targets):
                if label in curr_task_labels and len(val_map[label]) < self._validation_size:
                    val_map[label] += [j]
                    val_map['all'] += [j]  # gather all validation indices in one list

            indices = [j for j, label in enumerate(dataset.targets) if label in curr_task_labels and j not in val_map[label]]
            val_ds = Subset(dataset, val_map['all'])
            train_ds = Subset(dataset, indices)

            self._val_data_loader   = DataLoader(val_ds, batch_size=self._batch_size, shuffle=False)
            self._train_data_loader = DataLoader(train_ds, batch_size=self._batch_size, shuffle=True)

            del val_map
            print('Done!')
            return train_ds, val_ds
        else:
            indices = [j for j, label in enumerate(dataset.targets) if label in curr_task_labels]
            test_ds = Subset(dataset, indices)

            self._test_data_loader = DataLoader(test_ds, batch_size=self._batch_size, shuffle=False)

            return test_ds
        
        # for saving task specific images and labels in memory:
        # indices = [j for j, label in enumerate(dataset.targets) if label in curr_task_labels]
        # ds = Subset(dataset, indices)
        ## images = np.array([elem[0] for elem in ds])
        ## labels = np.array([elem[1] for elem in ds])
        # return images, labels
    
    def get_val_inputs(self):
        images = torch.stack([elem[0] for elem in self.val_data], dim=0)
        return images
    
    def get_train_inputs(self):
        images = torch.stack([elem[0] for elem in self.train_data], dim=0)
        return images
    
    def get_test_inputs(self):
        images = torch.stack([elem[0] for elem in self.test_data], dim=0)
        return images

    def get_val_outputs(self):
        labels = torch.stack([elem[1] for elem in self.val_data], dim=0)
        return labels
    
    def get_train_outputs(self):
        labels = torch.stack([elem[1] for elem in self.train_data], dim=0)
        return labels
    
    def get_test_outputs(self):
        labels = torch.stack([elem[1] for elem in self.test_data], dim=0)
        return labels
    
    def input_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):

        return x.to(device)
    
    def output_to_torch_tensor(self, x, device, mode='inference',
                              force_no_preprocessing=False, sample_ids=None):

        return x.to(device)

    def next_train_batch(self, batch_size: int):
        """Returns the next train batch of data"""

        return next(iter(self._train_data_loader))
    
    def next_test_batch(self, batch_size: int):
        """Returns the next test batch of data"""

        return next(iter(self._test_data_loader))
    
    def next_test_batch(self, batch_size: int):
        """Returns the next val batch of data"""

        return next(iter(self._val_data_loader))

    def _get_train_val_split(self):
        pass


    def get_identifier(self) -> str:
        """Returns the name of the dataset."""
        return "SubsetImageNet"
    
    
    def _calculate_modulo(self, target: int, task_id: int) -> int:
        MAX_NUM_CLASSES = self._data["num_classes"]
        INCR_NUM_CLASSES = self._data["num_incr_classes"]
        if task_id == 0:
            target = target % MAX_NUM_CLASSES
        else:
            target = target % INCR_NUM_CLASSES
        return target


    def _target_to_one_hot(self, target: int, task_id: int) -> torch.Tensor:
        target = self._calculate_modulo(target, task_id=task_id)
        one_hot = torch.eye(self._data["num_classes"])[target]
        return one_hot
    

    @staticmethod
    def plot_sample(image, label):
        plt.imshow(image.numpy())
        if isinstance(label, torch.Tensor):
            plt.title(f'Class: {label.argmax().item()}')
        elif isinstance(label, int):
            plt.title(f'Class: {label}')
        plt.axis("off")
        plt.show()

def prepare_subset_imagenet_tasks(
    datasets_folder: str = './',
    validation_size: int = 50, setting: int = 1
    ):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the SubsetImageNet dataset according to the WSN setup.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which Subset ImageNet
                         is stored / will be downloaded 
      *validation_size*: (optional int) defines the number of validation
                         samples in each task, by default it is 250 like
                         in the case of WSN
      *setting*: (optional int) defines the number and type of continual
                            learning tasks
    
    Returns a list of SubsetImageNet objects.
    """

    if setting == 1:
        number_of_tasks = 6
    elif setting == 2:
        number_of_tasks = 11
    elif setting == 3:
        number_of_tasks = 21
    elif setting == 4:
        number_of_tasks = 5
    else:
        raise(NotImplementedError)

    handlers = []
    for i in range(number_of_tasks):
        handlers.append(
            SubsetImageNet(
                path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=True,
                use_data_augmentation=True,
                task_id = i,
                setting = setting
            )
        )

    return handlers

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC, fillcolor=self.fillcolor)


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor)


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor)


class Rotate(object):
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(x.mode)


class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)

class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform = transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        func = {
            "shearX": ShearX(fillcolor=fillcolor),
            "shearY": ShearY(fillcolor=fillcolor),
            "translateX": TranslateX(fillcolor=fillcolor),
            "translateY": TranslateY(fillcolor=fillcolor),
            "rotate": Rotate(),
            "color": Color(),
            "posterize": Posterize(),
            "solarize": Solarize(),
            "contrast": Contrast(),
            "sharpness": Sharpness(),
            "brightness": Brightness(),
            "autocontrast": AutoContrast(),
            "equalize": Equalize(),
            "invert": Invert()
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


if __name__ == "__main__":

    pass