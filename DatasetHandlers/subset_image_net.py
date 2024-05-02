"""
Implementation of Subset-ImageNet for continual learning tasks. Please note that this dataset
consists of 100 randomly chosen classes with taken 1993 seed.

"""

import torch
import numpy as np
import random  # may be useful for validation data split
import matplotlib.pyplot as plt

from torchvision import transforms
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision.datasets import ImageFolder


class SubsetImageNet(Dataset):


    def __init__(self, path: str = "./seed_1993_subset_100_imagenet/data", 
                use_one_hot: bool = False,
                use_data_augmentation: bool = False, 
                validation_size: int = 100,
                task_id: int = 0,
                setting: int = 1):
        
        assert validation_size <= 250

        super().__init__()

        self._data = dict()
        self._path = path
        self._train_path = f"{path}/train"
        self._test_path = f"{path}/val"
        self._labels = np.arange(100)
        self._use_one_hot = use_one_hot
        self._use_data_augmentation = use_data_augmentation
        self._validation_size = validation_size
        self._setting = setting

        self._test_transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: torch.permute(x, (1, 2, 0)))
            ]
        )
        
        if self._use_data_augmentation:
            self._train_transform = transforms.Compose(
                [
                    # transforms.RandomCrop(64, padding=4),  # ERROR: input image size (62, 80)
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: torch.permute(x, (1, 2, 0)))
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
            del val_map
            print('Done!')
            return train_ds, val_ds
        else:
            indices = [j for j, label in enumerate(dataset.targets) if label in curr_task_labels]
            test_ds = Subset(dataset, indices)
            return test_ds
        
        # for saving task specific images and labels in memory:
        # indices = [j for j, label in enumerate(dataset.targets) if label in curr_task_labels]
        # ds = Subset(dataset, indices)
        ## images = np.array([elem[0] for elem in ds])
        ## labels = np.array([elem[1] for elem in ds])
        # return images, labels
    

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
if __name__ == "__main__":

    path = './Data/imagenet100'
    data = prepare_subset_imagenet_tasks(datasets_folder=path, setting = 4)

    for task in data:
        train_dl = DataLoader(task.train_data, batch_size=10, shuffle=False)
        valid_dl = DataLoader(task.val_data, batch_size=10, shuffle=False)
        test_dl = DataLoader(task.test_data, batch_size=10, shuffle=False)
        break

    for images, labels in test_dl:
        break

    for img, lab in valid_dl:
        break

    for _img, _lab in train_dl:
        break


    print(labels[-1])

    SubsetImageNet.plot_sample(images[0], labels[0])
    SubsetImageNet.plot_sample(img[0], lab[0])
    SubsetImageNet.plot_sample(_img[0], _lab[0])