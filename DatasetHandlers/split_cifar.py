#!/usr/bin/env python3
# Copyright 2019 Johannes von Oswald
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title           :split_cifar.py
# @author          :jvo
# @contact         :oswald@ini.ethz.ch
# @created         :05/13/2019
# @version         :1.0
# @python_version  :3.7.3
"""
Split CIFAR-10/100 Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.split_cifar` contains a wrapper for data handlers
for the Split-CIFAR10/CIFAR100 task.
"""
# FIXME The code in this module is mostly a copy of the code in the
# corresponding `split_mnist` module.
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
import torch

from hypnettorch.data.special.split_cifar import _transform_split_outputs
from DatasetHandlers.cifar10_data import CIFAR10Data
from DatasetHandlers.cifar100_data import CIFAR100Data

def get_split_cifar_handlers(data_path, use_one_hot=True, validation_size=0,
                             use_data_augmentation=False, use_cutout=False,
                             num_classes_per_task=10, num_tasks=6):
    """This method will combine 1 object of the class
    :class:`data.cifar10_data.CIFAR10Data` and 5 objects of the class
    :class:`SplitCIFAR100Data`.

    The SplitCIFAR benchmark consists of 6 tasks, corresponding to the images
    in CIFAR-10 and 5 tasks from CIFAR-100 corresponding to the images with
    labels [0-10], [10-20], [20-30], [30-40], [40-50].

    Args:
        data_path: Where should the CIFAR-10 and CIFAR-100 datasets
            be read from? If not existing, the datasets will be downloaded
            into this folder.
        use_one_hot (bool): Whether the class labels should be represented in a
            one-hot encoding.
        validation_size: The size of the validation set of each individual
            data handler.
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor`
            (hence, **only available for PyTorch**).
        use_cutout (bool): See docstring of class
            :class:`data.cifar10_data.CIFAR10Data`.
        num_classes_per_task (int): Number of classes to put into one data
            handler. For example, if ``2``, then every data handler will include
            2 digits.

            If ``10``, then the first dataset will simply be CIFAR-10.
        num_tasks (int): A number between 1 and 11 (assuming
            ``num_classes_per_task == 10``), specifying the number of data
            handlers to be returned. If ``num_tasks=6``, then there will be
            the CIFAR-10 data handler and the first 5 splits of the CIFAR-100
            dataset (as in the usual CIFAR benchmark for CL).

    Returns:
        (list) A list of data handlers. The first being an instance of class
        :class:`data.cifar10_data.CIFAR10Data` and the remaining ones being an
        instance of class :class:`SplitCIFAR100Data`.
    """
    if not (num_tasks >= 1 and (num_tasks * num_classes_per_task) <= 110):
        raise ValueError('Cannot create SplitCIFAR datasets for %d tasks ' \
                         % (num_tasks) + 'with %d classes per task.' \
                         % (num_classes_per_task))

    print('Creating data handlers for SplitCIFAR tasks ...')

    handlers = []
    ### CIFAR-10
    if num_classes_per_task == 10:
        # First task is CIFAR-10.
        handlers.append(CIFAR10Data(data_path, use_one_hot=use_one_hot,
                validation_size=validation_size,
                use_data_augmentation=use_data_augmentation,
                use_cutout=use_cutout))
    else:
        if (num_tasks * num_classes_per_task) > 10 and \
                10 % num_classes_per_task != 0:
            # Our implementation doesn't allow to create datasets where CIFAR-10
            # and CIFAR-100 data is mixed.
            raise ValueError('Argument "num_classes_per_task" must be in ' +
                             '[1, 2, 5, 10].')

        steps = num_classes_per_task
        for i in range(0, 10, steps):
            handlers.append(SplitCIFAR10Data(data_path,
                use_one_hot=use_one_hot, validation_size=validation_size,
                use_data_augmentation=use_data_augmentation,
                use_cutout=use_cutout, labels=range(i, i+steps)))

            if len(handlers) == num_tasks:
                break

    ### CIFAR-100
    if len(handlers) < num_tasks:
        steps = num_classes_per_task
        for i in range(0, 100, steps):
            handlers.append(SplitCIFAR100Data(data_path,
                use_one_hot=use_one_hot, validation_size=validation_size,
                use_data_augmentation=use_data_augmentation,
                use_cutout=use_cutout, labels=range(i, i+steps)))

            if len(handlers) == num_tasks:
                break

    print('Creating data handlers for SplitCIFAR tasks ... Done')

    return handlers

class SplitCIFAR100Data(CIFAR100Data):
    """An instance of the class shall represent a single SplitCIFAR-100 task.

    Args:
        data_path: Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size: The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor`
            (hence, **only available for PyTorch**).
            Note, we are using the same data augmentation pipeline as for
            CIFAR-10.
        use_cutout (bool): See docstring of class
            :class:`data.cifar10_data.CIFAR10Data`.
        labels: The labels that should be part of this task.
        full_out_dim: Choose the original CIFAR instead of the the new
            task output dimension. This option will affect the attributes
            :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
    """
    # Note, we build the validation set below!
    def __init__(self, data_path, use_one_hot=False, validation_size=1000,
                 use_data_augmentation=False, use_cutout=False,
                 labels=range(0, 10), full_out_dim=False):
        super().__init__(data_path, use_one_hot=use_one_hot, validation_size=0,
                         use_data_augmentation=use_data_augmentation,
                         use_cutout=use_cutout)

        _split_cifar_object(self, data_path, use_one_hot, validation_size,
                            use_data_augmentation, labels, full_out_dim)

    def transform_outputs(self, outputs):
        """Transform the outputs from the 100D CIFAR100 dataset into proper
        labels based on the constructor argument ``labels``.

        See :meth:`data.special.split_mnist.SplitMNIST.transform_outputs` for
        more information.

        Args:
            outputs: 2D numpy array of outputs.

        Returns:
            2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitCIFAR100'

class SplitCIFAR10Data(CIFAR10Data):
    """An instance of the class shall represent a single SplitCIFAR-10 task.

    Each instance will contain only samples of CIFAR-10 belonging to a subset
    of the labels.

    Args:
        (....): See docstring of class :class:`SplitCIFAR100Data`.
    """
    def __init__(self, data_path, use_one_hot=False, validation_size=1000,
                 use_data_augmentation=False, use_cutout=False,
                 labels=range(0, 2), full_out_dim=False):
        # Note, we build the validation set below!
        super().__init__(data_path, use_one_hot=use_one_hot, validation_size=0,
                         use_data_augmentation=use_data_augmentation,
                         use_cutout=use_cutout)

        _split_cifar_object(self, data_path, use_one_hot, validation_size,
                            use_data_augmentation, labels, full_out_dim)

    def transform_outputs(self, outputs):
        """Transform the outputs from the 10D CIFAR10 dataset into proper labels
        based on the constructor argument ``labels``.

        See :meth:`data.special.split_mnist.SplitMNIST.transform_outputs` for
        more information.

        Args:
            outputs (numpy.ndarray): 2D numpy array of outputs.

        Returns:
            (numpy.ndarray): 2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return 'SplitCIFAR10'

def _split_cifar_object(data, data_path, use_one_hot, validation_size,
                        use_data_augmentation, labels, full_out_dim):
    """Extract a subset of labels from a CIFAR-10 or CIFAR-100 dataset.

    The constructors of classes :class:`SplitCIFAR10Data` and
    :class:`SplitCIFAR100Data` are essentially identical. Therefore, the code
    is realized in this function.
    
    Args:
        data: The data handler (which is a full CIFAR-10 or CIFAR-100 dataset,
            which will be modified).
        (....): See docstring of class :class:`SplitCIFAR10Data`.
    """
    assert isinstance(data, SplitCIFAR10Data) or \
           isinstance(data, SplitCIFAR100Data)
    data._full_out_dim = full_out_dim

    if isinstance(labels, range):
        labels = list(labels)
    assert np.all(np.array(labels) >= 0) and \
           np.all(np.array(labels) < data.num_classes) and \
           len(labels) == len(np.unique(labels))
    K = len(labels)

    data._labels = labels

    train_ins = data.get_train_inputs()
    test_ins = data.get_test_inputs()

    train_outs = data.get_train_outputs()
    test_outs = data.get_test_outputs()

    # Get labels.
    if data.is_one_hot:
        train_labels = data._to_one_hot(train_outs, reverse=True)
        test_labels = data._to_one_hot(test_outs, reverse=True)
    else:
        train_labels = train_outs
        test_labels = test_outs

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    train_mask = train_labels == labels[0]
    test_mask = test_labels == labels[0]
    for k in range(1, K):
        train_mask = np.logical_or(train_mask, train_labels == labels[k])
        test_mask = np.logical_or(test_mask, test_labels == labels[k])

    train_ins = train_ins[train_mask, :]
    test_ins = test_ins[test_mask, :]

    train_outs = train_outs[train_mask, :]
    test_outs = test_outs[test_mask, :]

    if validation_size > 0:
        if validation_size >= train_outs.shape[0]:
            raise ValueError('Validation set must contain less than %d ' \
                             % (train_outs.shape[0]) + 'samples!')
        val_inds = np.arange(validation_size)
        train_inds = np.arange(validation_size, train_outs.shape[0])

    else:
        train_inds = np.arange(train_outs.shape[0])

    test_inds = np.arange(train_outs.shape[0],
                          train_outs.shape[0] + test_outs.shape[0])

    outputs = np.concatenate([train_outs, test_outs], axis=0)

    if not full_out_dim:
        # Note, the method assumes `full_out_dim` when later called by a
        # user. We just misuse the function to call it inside the
        # constructor.
        data._full_out_dim = True
        outputs = data.transform_outputs(outputs)
        data._full_out_dim = full_out_dim

        # Note, we may also have to adapt the output shape appropriately.
        if data.is_one_hot:
            data._data['out_shape'] = [len(labels)]

        # And we also correct the label names.
        if isinstance(data, SplitCIFAR10Data):
            data._data['cifar10']['label_names'] = \
                [data._data['cifar10']['label_names'][ii] for ii in labels]
        else:
            data._data['cifar100']['fine_label_names'] = \
                [data._data['cifar100']['fine_label_names'][ii] \
                 for ii in labels]
            # FIXME I just set it to `None` as I don't know what to do with it
            # right now.
            data._data['cifar100']['coarse_label_names'] = None

    images = np.concatenate([train_ins, test_ins], axis=0)
    print(np.min(images))

    ### Overwrite internal data structure. Only keep desired labels.

    # Note, we continue to pretend to be a 100 class problem, such that
    # the user has easy access to the correct labels and has the original
    # 1-hot encodings.
    if not full_out_dim:
        data._data['num_classes'] = len(labels)
    else:
        # Note, we continue to pretend to be a 10/100 class problem, such that
        # the user has easy access to the correct labels and has the
        # original 1-hot encodings.
        if isinstance(data, SplitCIFAR10Data):
            assert data._data['num_classes'] == 10
        else:
            assert data._data['num_classes'] == 100
    data._data['in_data'] = images
    data._data['out_data'] = outputs
    data._data['train_inds'] = train_inds
    data._data['test_inds'] = test_inds
    if validation_size > 0:
        data._data['val_inds'] = val_inds

    n_val = 0
    if validation_size > 0:
        n_val = val_inds.size

    print('Created SplitCIFAR-%d task with labels %s and %d train, %d test '
          % (10 if isinstance(data, SplitCIFAR10Data) else 100, str(labels),
             train_inds.size, test_inds.size) +
          'and %d val samples.' % (n_val))

def _transform_split_outputs(data, outputs):
    """Actual implementation of method ``transform_outputs`` for split dataset
    handlers.

    Args:
        data: Data handler.
        outputs (numpy.ndarray): See docstring of method
            :meth:`data.special.split_mnist.SplitMNIST.transform_outputs`
    
    Returns:
        (numpy.ndarray)
    """
    if not data._full_out_dim:
        # TODO implement reverse direction as well.
        raise NotImplementedError('This method is currently only ' +
            'implemented if constructor argument "full_out_dim" was set.')

    labels = data._labels
    if data.is_one_hot:
        assert(outputs.shape[1] == data._data['num_classes'])
        mask = np.zeros(data._data['num_classes'], dtype=np.bool)
        mask[labels] = True

        return outputs[:, mask]
    else:
        assert (outputs.shape[1] == 1)
        ret = outputs.copy()
        for i, l in enumerate(labels):
            ret[ret == l] = i
        return ret


class SplitCIFAR100Data_FeCAM(CIFAR100Data):
    """An instance of the class shall represent a single SplitCIFAR-100 task.

    Args:
        data_path: Where should the dataset be read from? If not existing,
            the dataset will be downloaded into this folder.
        use_one_hot (bool): Whether the class labels should be
            represented in a one-hot encoding.
        validation_size: The number of validation samples. Validation
            samples will be taking from the training set (the first :math:`n`
            samples).
        use_data_augmentation (optional): Note, this option currently only
            applies to input batches that are transformed using the class
            member :meth:`data.dataset.Dataset.input_to_torch_tensor`
            (hence, **only available for PyTorch**).
            Note, we are using the same data augmentation pipeline as for
            CIFAR-10.
        use_cutout (bool): See docstring of class
            :class:`data.cifar10_data.CIFAR10Data`.
        labels: The labels that should be part of this task.
        full_out_dim: Choose the original CIFAR instead of the the new
            task output dimension. This option will affect the attributes
            :attr:`data.dataset.Dataset.num_classes` and
            :attr:`data.dataset.Dataset.out_shape`.
    """

    # Note, we build the validation set below!
    def __init__(
        self,
        data_path,
        use_one_hot=False,
        validation_size=1000,
        use_data_augmentation=False,
        use_cutout=False,
        labels=range(0, 10),
        full_out_dim=False,
    ):
        super().__init__(
            data_path,
            use_one_hot=use_one_hot,
            validation_size=0,
            use_data_augmentation=use_data_augmentation,
            use_cutout=use_cutout,
        )

        _split_cifar_100_fecam_object(
            self,
            data_path,
            use_one_hot,
            validation_size,
            use_data_augmentation,
            labels,
            full_out_dim,
        )

    def transform_outputs(self, outputs):
        """Transform the outputs from the 100D CIFAR100 dataset into proper
        labels based on the constructor argument ``labels``.

        See :meth:`data.special.split_mnist.SplitMNIST.transform_outputs` for
        more information.

        Args:
            outputs: 2D numpy array of outputs.

        Returns:
            2D numpy array of transformed outputs.
        """
        return _transform_split_outputs(self, outputs)

    def get_identifier(self):
        """Returns the name of the dataset."""
        return "SplitCIFAR100"


def _split_cifar_100_fecam_object(
    data,
    data_path,
    use_one_hot,
    validation_size,
    use_data_augmentation,
    labels,
    full_out_dim,
):
    """Extract a subset of labels from a CIFAR-100 dataset.

    The constructors of classes :class:`SplitCIFAR10Data` and
    :class:`SplitCIFAR100Data_FeCAM` are essentially identical. Therefore, the code
    is realized in this function.

    Args:
        data: The data handler (which is a full CIFAR-100 dataset,
            which will be modified).
        (....): See docstring of class :class:`SplitCIFAR10Data`.
    """
    assert isinstance(data, SplitCIFAR100Data_FeCAM)
    data._full_out_dim = full_out_dim

    if isinstance(labels, range):
        labels = list(labels)
    assert (
        np.all(np.array(labels) >= 0)
        and np.all(np.array(labels) < data.num_classes)
        and len(labels) == len(np.unique(labels))
    )
    K = len(labels)

    data._labels = labels

    train_ins = data.get_train_inputs()
    test_ins = data.get_test_inputs()

    train_outs = data.get_train_outputs()
    test_outs = data.get_test_outputs()

    # Get labels.
    if data.is_one_hot:
        train_labels = data._to_one_hot(train_outs, reverse=True)
        test_labels = data._to_one_hot(test_outs, reverse=True)
    else:
        train_labels = train_outs
        test_labels = test_outs

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    train_mask = train_labels == labels[0]
    test_mask = test_labels == labels[0]
    for k in range(1, K):
        train_mask = np.logical_or(train_mask, train_labels == labels[k])
        test_mask = np.logical_or(test_mask, test_labels == labels[k])

    train_ins = train_ins[train_mask, :]
    test_ins = test_ins[test_mask, :]

    train_outs = train_outs[train_mask, :]
    test_outs = test_outs[test_mask, :]

    if validation_size > 0:
        if validation_size >= train_outs.shape[0]:
            raise ValueError(
                "Validation set must contain less than %d "
                % (train_outs.shape[0])
                + "samples!"
            )
        val_inds = np.arange(validation_size)
        train_inds = np.arange(validation_size, train_outs.shape[0])

    else:
        train_inds = np.arange(train_outs.shape[0])

    test_inds = np.arange(
        train_outs.shape[0], train_outs.shape[0] + test_outs.shape[0]
    )

    outputs = np.concatenate([train_outs, test_outs], axis=0)

    if not full_out_dim:
        # Note, the method assumes `full_out_dim` when later called by a
        # user. We just misuse the function to call it inside the
        # constructor.
        data._full_out_dim = True
        outputs = data.transform_outputs(outputs)
        data._full_out_dim = full_out_dim

        # Note, we may also have to adapt the output shape appropriately.
        if data.is_one_hot:
            data._data["out_shape"] = [len(labels)]

        data._data["cifar100"]["fine_label_names"] = [
            data._data["cifar100"]["fine_label_names"][ii] for ii in labels
        ]
        # FIXME I just set it to `None` as I don't know what to do with it
        # right now.
        data._data["cifar100"]["coarse_label_names"] = None

    images = np.concatenate([train_ins, test_ins], axis=0)

    ### Overwrite internal data structure. Only keep desired labels.

    # Note, we continue to pretend to be a 100 class problem, such that
    # the user has easy access to the correct labels and has the original
    # 1-hot encodings.
    if not full_out_dim:
        data._data["num_classes"] = len(labels)
    else:
        # Note, we continue to pretend to be a 10/100 class problem, such that
        # the user has easy access to the correct labels and has the
        # original 1-hot encodings.
        assert data._data["num_classes"] == 100
    data._data["in_data"] = images
    data._data["out_data"] = outputs
    data._data["train_inds"] = train_inds
    data._data["test_inds"] = test_inds
    if validation_size > 0:
        data._data["val_inds"] = val_inds

    n_val = 0
    if validation_size > 0:
        n_val = val_inds.size

    print(
        "Created SplitCIFAR-%d task with labels %s and %d train, %d test "
        % (100, str(labels), train_inds.size, test_inds.size)
        + "and %d val samples." % (n_val)
    )


########################################################
# Classes from FeCAM
########################################################
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

            mask[y1:y2, x1:x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ShearX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=self.fillcolor,
        )


class ShearY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
            Image.BICUBIC,
            fillcolor=self.fillcolor,
        )


class TranslateX(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
            fillcolor=self.fillcolor,
        )


class TranslateY(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor

    def __call__(self, x, magnitude):
        return x.transform(
            x.size,
            Image.AFFINE,
            (1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
            fillcolor=self.fillcolor,
        )


class Rotate(object):
    def __call__(self, x, magnitude):
        rot = x.convert("RGBA").rotate(magnitude * random.choice([-1, 1]))
        return Image.composite(
            rot, Image.new("RGBA", rot.size, (128,) * 4), rot
        ).convert(x.mode)


class Color(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Color(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class Posterize(object):
    def __call__(self, x, magnitude):
        return ImageOps.posterize(x, magnitude)


class Solarize(object):
    def __call__(self, x, magnitude):
        return ImageOps.solarize(x, magnitude)


class Contrast(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Contrast(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class Sharpness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Sharpness(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class Brightness(object):
    def __call__(self, x, magnitude):
        return ImageEnhance.Brightness(x).enhance(
            1 + magnitude * random.choice([-1, 1])
        )


class AutoContrast(object):
    def __call__(self, x, magnitude):
        return ImageOps.autocontrast(x)


class Equalize(object):
    def __call__(self, x, magnitude):
        return ImageOps.equalize(x)


class Invert(object):
    def __call__(self, x, magnitude):
        return ImageOps.invert(x)


class SubPolicy(object):
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
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
            "invert": [0] * 10,
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
            "invert": Invert(),
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


class CIFAR10Policy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.

    Example:
    >>> policy = CIFAR10Policy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     CIFAR10Policy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"

if __name__ == '__main__':
    pass