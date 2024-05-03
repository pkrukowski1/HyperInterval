"""
This file implements functions to deal with datasets and tasks in continual learning

Author: Kamil Książek
"""

import os
import numpy as np
import torch
from hypnettorch.data.special import permuted_mnist
from DatasetHandlers.split_cifar import SplitCIFAR100Data
from DatasetHandlers.split_mnist import get_split_mnist_handlers
from DatasetHandlers.subset_image_net import SubsetImageNet
from DatasetHandlers.TinyImageNet import TinyImageNet
from DatasetHandlers.cifar100_FeCAM import SplitCIFAR100Data_FeCAM

def generate_random_permutations(shape_of_data_instance,
                                 number_of_permutations):
    """
    Prepare a list of random permutations of the selected shape
    for continual learning tasks.

    Arguments:
    ----------
      *shape_of_data_instance*: a number defining shape of the dataset
      *number_of_permutations*: int, a number of permutations that will
                                be prepared; it corresponds to the total
                                number of tasks
      *seed*: int, optional argument, default: None
              if one would get deterministic results
    """
    list_of_permutations = []
    for _ in range(number_of_permutations):
        list_of_permutations.append(np.random.permutation(
            shape_of_data_instance))
    return list_of_permutations


def prepare_split_cifar100_tasks(datasets_folder,
                                 validation_size,
                                 use_augmentation,
                                 use_cutout=False):
    """
    Prepare a list of 10 tasks with 10 classes per each task.
    i-th task, where i in {0, 1, ..., 9} will store samples
    from classes {10*i, 10*i + 1, ..., 10*i + 9}.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    "apply_cutout" option form "torch_input_transforms".
    """
    handlers = []
    for i in range(0, 100, 10):
        handlers.append(SplitCIFAR100Data(
            datasets_folder,
            use_one_hot=True,
            validation_size=validation_size,
            use_data_augmentation=use_augmentation,
            use_cutout=use_cutout,
            labels=range(i, i + 10)
        ))
    return handlers

def prepare_split_cifar100_tasks_aka_FeCAM(
    datasets_folder,
    number_of_tasks,
    no_of_validation_samples_per_class,
    use_augmentation,
    use_cutout=False,
):
    """
    Prepare a list of 5, 10 or 20 incremental tasks with 20, 10 or 5 classes,
    respectively, per each task. Furthermore, the first task contains
    a higher number of classes, i.e. 50 or 40. Therefore, in these cases,
    the total number of tasks is equal to 6, 11 or 21.
    Also, there is a possibility of 5 tasks with 20 classes per each.
    The order of classes is the same like in FeCAM, also the scenarios
    are constructed in such a way to enable a fair comparison with FeCAM

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which CIFAR-100
                         is stored / will be downloaded
      *number_of_tasks* (int) Defines how many continual learning tasks
                        will be created. Possible options: 6, 11 or 21
      *no_of_validation_samples_per_class*: (int) The number of validation
                                            samples in a single class
      *use_augmentation*: (Boolean) potentially applies
                          a data augmentation method from
                          hypnettorch
      *use_cutout*: (optional Boolean) in the positive case it applies
                    "apply_cutout" option form "torch_input_transforms".
    """
    # FeCAM considered four scenarios: 5, 10 and 20 incremental tasks
    # and 5 tasks with the equal number of classes
    assert number_of_tasks in [5, 6, 11, 21]
    # The order of image classes in the case of FeCAM was not 0-10, 11-20, etc.,
    # but it was chosen randomly by the authors, and was at follows:
    class_orders = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
        94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
        84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
        69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
        17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
        1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
        38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
        40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
        98, 13, 99, 7, 34, 55, 54, 26, 35, 39
    ]
    # Incremental tasks from Table I, FeCAM
    if number_of_tasks == 6:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([10 for i in range(5)])
    elif number_of_tasks == 11:
        numbers_of_classes_per_tasks = [50]
        numbers_of_classes_per_tasks.extend([5 for i in range(10)])
    elif number_of_tasks == 21:
        numbers_of_classes_per_tasks = [40]
        numbers_of_classes_per_tasks.extend([3 for i in range(20)])
    # Tasks with the equal number of elements, Table V, FeCAM
    elif number_of_tasks == 5:
        numbers_of_classes_per_tasks = [20 for i in range(5)]

    handlers = []
    for i in range(len(numbers_of_classes_per_tasks)):
        current_number_of_tasks = numbers_of_classes_per_tasks[i]
        validation_size = (
            no_of_validation_samples_per_class * current_number_of_tasks
        )
        handlers.append(
            SplitCIFAR100Data_FeCAM(
                datasets_folder,
                use_one_hot=True,
                validation_size=validation_size,
                use_data_augmentation=use_augmentation,
                use_cutout=use_cutout,
                labels=class_orders[
                    (i * current_number_of_tasks) : (
                        (i + 1) * current_number_of_tasks
                    )
                ],
            )
        )
    return handlers


def prepare_subset_imagenet_tasks(
    datasets_folder: str = "./",
    no_of_validation_samples_per_class: int = 50, 
    setting: int = 4,
    use_augmentation = False,
    use_cutout = False,
    number_of_tasks = 5
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

    if setting != 4:
        raise ValueError("Only 5 incremental tasks are supported right now!")
    

    handlers = []
    for i in range(number_of_tasks):

        validation_size = (
            no_of_validation_samples_per_class * number_of_tasks
        )

        handlers.append(
            SubsetImageNet(
                path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=True,
                use_data_augmentation=use_augmentation,
                task_id = i,
                setting = setting
            )
        )

    return handlers


def prepare_permuted_mnist_tasks(datasets_folder,
                                 input_shape,
                                 number_of_tasks,
                                 padding,
                                 validation_size):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the PermutedMNIST dataset.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *input_shape*: (int) a number defining shape of the dataset
      *validation_size*: (int) The number of validation samples

    Returns a list of PermutedMNIST objects.
    """
    permutations = generate_random_permutations(
        input_shape,
        number_of_tasks
    )
    return permuted_mnist.PermutedMNISTList(
        permutations,
        datasets_folder,
        use_one_hot=True,
        padding=padding,
        validation_size=validation_size
    )


def prepare_split_mnist_tasks(datasets_folder,
                              validation_size,
                              use_augmentation,
                              number_of_tasks=5):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the SplitMNIST dataset. By default, it should be
    5 task containing consecutive pairs of classes:
    [0, 1], [2, 3], [4, 5], [6, 7] and [8, 9].

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which MNIST dataset
                         is stored / will be downloaded
      *validation_size*: (int) The number of validation samples
      *use_augmentation*: (bool) defines whether dataset augmentation
                          will be applied
      *number_of_tasks* (int) a number defining the number of learning
                        tasks, by default 5.

    Returns a list of SplitMNIST objects.
    """
    return get_split_mnist_handlers(
        datasets_folder,
        use_one_hot=True,
        validation_size=validation_size,
        num_classes_per_task=2,
        num_tasks=number_of_tasks,
        use_torch_augmentation=use_augmentation
    )

def prepare_tinyimagenet_tasks(
    datasets_folder, seed,
    validation_size=250, number_of_tasks=40
    ):
    """
    Prepare a list of *number_of_tasks* tasks related
    to the TinyImageNet dataset according to the WSN setup.

    Arguments:
    ----------
      *datasets_folder*: (string) Defines a path in which TinyImageNet
                         is stored / will be downloaded 
      *seed*: (int) Necessary for the preparation of random permutation
              of the order of classes in consecutive tasks.
      *validation_size*: (optional int) defines the number of validation
                         samples in each task, by default it is 250 like
                         in the case of WSN
      *number_of_tasks*: (optional int) defines the number of continual
                         learning tasks (by default: 40)
    
    Returns a list of TinyImageNet objects.
    """
    # Set randomly the order of classes
    rng = np.random.default_rng(seed)
    class_permutation = rng.permutation(200)
    # 40 classification tasks with 5 classes in each
    handlers = []
    for i in range(0, 5 * number_of_tasks, 5):
        current_labels = class_permutation[i:(i + 5)]
        print(f"Order of classes in the current task: {current_labels}")
        handlers.append(
            TinyImageNet(
                data_path=datasets_folder,
                validation_size=validation_size,
                use_one_hot=True,
                labels=current_labels
            )
        )
    return handlers


def set_hyperparameters(dataset,
                        grid_search=False):
    """
    Set hyperparameters of the experiments, both in the case of grid search
    optimization and a single network run.

    Arguments:
    ----------
      *dataset*: "PermutedMNIST", "SplitMNIST" or "CIFAR100"
      *grid_search*: (Boolean optional) defines whether a hyperparameter
                     optimization should be performed or hyperparameters
                     for just a single run have to be returned
      *part* (only for SplitMNIST or CIFAR100!) selects a subset
             of hyperparameters for optimization (by default 0)

    Returns a dictionary with necessary hyperparameters.
    """

    if dataset == "PermutedMNIST":
        if grid_search:
            hyperparams = {
                "custom_init": [True],
                "embedding_sizes": [24, 32, 64],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.01, 0.005, 0.1],
                "hypernetworks_hidden_layers": [[100, 100], [200, 200]],
                "perturbated_epsilon": [5.0, 10.0, 15.0],
                "best_model_selection_method": "val_loss",
                "dropout_rate": [-1, 0.1],
                "embd_dropout_rate": [-1],
                # not for optimization
                "seed": [1]
            }
            
            hyperparams["saving_folder"] = (
                "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/"
                f'permuted_mnist_final_grid_experiments/{hyperparams["best_model_selection_method"]}/'
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [3],
                "embedding_sizes": [24],
                "custom_init": [True],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "betas": [0.005],
                "perturbated_epsilon": [5.0],
                "hypernetworks_hidden_layers": [[100, 100]],
                "dropout_rate": [-1],
                "embd_dropout_rate": [-1],
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/"
                f"permuted_mnist_final_grid_experiments/last_model/"
            }

        # Both in the grid search and individual runs
        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = 5000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 500
        hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["target_network"] = "MLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 8
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        # Directly related to the MNIST dataset
        hyperparams["padding"] = 2
        hyperparams["shape"] = (28 + 2 * hyperparams["padding"])**2
        hyperparams["number_of_tasks"] = 10
        hyperparams["augmentation"] = False

        # Full-interval model or simpler one
        hyperparams["full_interval"] = True

    elif dataset == "CIFAR100":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "embedding_sizes": [48],
                "betas": [0.01, 0.1, 1.0],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "hypernetworks_hidden_layers": [[100]],
                "perturbated_epsilon": [10, 5, 1, 0.5],
                "dropout_rate": [-1, 0.25, 0.5],
                "embd_dropout_rate": [-1, 0.25],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": False
            }
        
            hyperparams["saving_folder"] = (
                "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/"
                f"CIFAR-100_single_seed/"
                f"ResNet/"
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [1],
                "custom_init": [False],
                "embedding_sizes": [48],
                "betas": [0.01],
                "batch_sizes": [32],
                "learning_rates": [0.001],
                "perturbated_epsilon": [1.0],
                "hypernetworks_hidden_layers": [[100]],
                "dropout_rate": [-1, 0.25],
                "embd_dropout_rate": [-1, 0.25],
                "use_batch_norm": True,
                "use_chunks": False,
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "number_of_epochs": 200,
                "target_network": "ResNet",
                "optimizer": "adam",
                "augmentation": True
            }
           
            hyperparams["saving_folder"] = (
                "./Results/grid_search_relu/"
                f"CIFAR-100_single_seed/"
                f"final_run/"
            )
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 500
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 10
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False

    elif dataset == "SplitMNIST":
        if grid_search:
            hyperparams = {
                "custom_init": [True],
                "learning_rates": [0.001],
                "batch_sizes": [64, 128],
                "betas": [0.01, 0.001],
                "hypernetworks_hidden_layers": [[25, 25], [50, 50], [75, 75]],
                "dropout_rate": [-1],
                "embd_dropout_rate": [-1],
                "perturbated_epsilon": [5.0, 10.0, 15.0],
                # seed is not for optimization but for ensuring multiple results
                "seed": [1],
                "best_model_selection_method": "val_loss",
                "embedding_sizes": [24, 72, 96, 128],
                "augmentation": True
            }

            hyperparams["saving_folder"] = (
                "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/"
                f"split_mnist/augmented/"
            )

        else:
            # single run experiment
            hyperparams = {
                "seed": [3],
                "embedding_sizes": [24],
                "learning_rates": [0.001],
                "batch_sizes": [64],
                "betas": [0.001],
                "perturbated_epsilon": [10.0],
                "dropout_rate": [-1],
                "embd_dropout_rate": [-1],
                "hypernetworks_hidden_layers": [[100]],
                "augmentation": False,
                "best_model_selection_method": "val_loss",
                "saving_folder": "./Results/SplitMNIST/"
            }
        hyperparams["lr_scheduler"] = False
        hyperparams["target_network"] = "MLP"
        hyperparams["resnet_number_of_layer_groups"] = None
        hyperparams["resnet_widening_factor"] = None
        hyperparams["optimizer"] = "adam"
        hyperparams["number_of_iterations"] = 2000
        hyperparams["number_of_epochs"] = None
        hyperparams["no_of_validation_samples"] = 1000
        hyperparams["target_hidden_layers"] = [400, 400]
        hyperparams["shape"] = 28**2
        hyperparams["number_of_tasks"] = 5
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 96
        hyperparams["use_chunks"] = False
        hyperparams["use_batch_norm"] = False
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = True
    
    elif dataset == "TinyImageNet":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "custom_init": [True],
                "perturbated_epsilon": [10, 15, 20],
                "embedding_sizes": [96],
                "learning_rates": [0.001, 0.01],
                "batch_sizes": [16, 32],
                "dropout_rate": [-1, 0.25],
                "embd_dropout_rate": [-1],
                "betas": [1.0, 0.01, 0.1],
                "hypernetworks_hidden_layers": [[100, 100], [200, 200]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": False,
                "target_network": "ZenkeNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True
            }

            hyperparams["saving_folder"] = (
                "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/"
                f"TinyImageNet/"
                f"ResNet/"
            )
        else:
            hyperparams = {
               "seed": [1],
               "custom_init": [True],
                "perturbated_epsilon": [1.0],
                "embedding_sizes": [48],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "betas": [0.01],
                "hypernetworks_hidden_layers": [[100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "embd_dropout_rate": [-1, 0.25],
                "dropout_rate": [-1, 0.25, 0.5],
                "optimizer": "adam",
                "use_batch_norm": False,
                "target_network": "ZenkeNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True,
                "saving_folder": "./Results/TinyImageNet/best_hyperparams/"
            }
        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples"] = 250
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 64
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 12288
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["number_of_tasks"] = 40
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False
    
    elif dataset == "SubsetImageNet":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "custom_init": [True],
                "perturbated_epsilon": [10, 15, 20],
                "embedding_sizes": [48, 96, 128],
                "learning_rates": [0.001, 0.01],
                "batch_sizes": [16, 32],
                "dropout_rate": [-1, 0.25],
                "embd_dropout_rate": [-1],
                "betas": [1.0, 0.01, 0.1],
                "hypernetworks_hidden_layers": [[100, 100], [200, 200]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True
            }

            hyperparams["saving_folder"] = (
                "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/"
                f"SubsetImageNet/"
                f"ResNet/"
            )
        else:
            hyperparams = {
               "seed": [1],
               "custom_init": [True],
                "perturbated_epsilon": [1.0],
                "embedding_sizes": [48],
                "learning_rates": [0.001],
                "batch_sizes": [32],
                "betas": [0.01],
                "hypernetworks_hidden_layers": [[100]],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "embd_dropout_rate": [-1, 0.25],
                "dropout_rate": [-1, 0.25, 0.5],
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 10,
                "augmentation": True,
                "saving_folder": "./Results/SubsetImageNet/best_hyperparams/"
            }
        hyperparams["lr_scheduler"] = True
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples_per_class"] = 50
        hyperparams["number_of_tasks"] = 5
        
        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 64
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 12288
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False
    
    elif dataset == "CIFAR100_FeCAM_setup":
        if grid_search:
            hyperparams = {
                "seed": [1],
                "custom_init": [True],
                "embedding_sizes": [48, 64],
                "betas": [0.01, 0.1, 1.0],
                "learning_rates": [0.0001, 0.001],
                "batch_sizes": [32],
                "hypernetworks_hidden_layers": [[200], [100]],
                "perturbated_epsilon": [10, 15, 20],
                "dropout_rate": [-1, 0.2],
                "embd_dropout_rate": [-1],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": False,
                "target_network": "ZenkeNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True,
                "best_model_selection_method": "val_loss"
            }

            hyperparams["saving_folder"] = (
                "/shared/results/pkrukowski/HyperIntervalResults/common_embedding/grid_search_relu/"
                f'CIFAR100_FeCAM_setup/{hyperparams["best_model_selection_method"]}/'
            )
        else:
            # Single experiment
            hyperparams = {
               "seed": [1],
               "custom_init": [True],
                "embedding_sizes": [48],
                "betas": [0.01],
                "learning_rates": [0.001],
                "batch_sizes": [128],
                "hypernetworks_hidden_layers": [[100]],
                "perturbated_epsilon": [10],
                "dropout_rate": [-1],
                "embd_dropout_rate": [-1],
                "resnet_number_of_layer_groups": 3,
                "resnet_widening_factor": 2,
                "optimizer": "adam",
                "use_batch_norm": True,
                "target_network": "ResNet",
                "use_chunks": False,
                "number_of_epochs": 200,
                "augmentation": True
            }
            # FeCAM considered three incremental scenarios: with 6, 11 and 21 tasks
            # ResNet - parts 0, 1 and 2
            # ZenkeNet - parts 3, 4 and 5
            # Also, one scenario with equal number of classes: ResNet - part 6
            hyperparams[
                "saving_folder"
            ] = f"./Results/CIFAR_100_FeCAM/"

        hyperparams["lr_scheduler"] = False
        hyperparams["number_of_iterations"] = None
        hyperparams["no_of_validation_samples_per_class"] = 50
        hyperparams["no_of_validation_samples"] = 2000
        hyperparams["number_of_tasks"] = 5

        if hyperparams["target_network"] in ["ResNet", "ZenkeNet"]:
            hyperparams["shape"] = 32
            hyperparams["target_hidden_layers"] = None
        elif hyperparams["target_network"] == "MLP":
            hyperparams["shape"] = 3072
            hyperparams["target_hidden_layers"] = [1000, 1000]
        hyperparams["chunk_size"] = 100
        hyperparams["chunk_emb_size"] = 32
        hyperparams["padding"] = None
        hyperparams["best_model_selection_method"] = "val_loss"

        # Full-interval model or simpler one
        hyperparams["full_interval"] = False

    else:
        raise ValueError("This dataset is not implemented!")

    # General hyperparameters
    hyperparams["activation_function"] = torch.nn.ReLU()
    hyperparams["norm"] = 1  # L1 norm
    hyperparams["use_bias"] = True
    hyperparams["dataset"] = dataset
    hyperparams["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams["kappa"] = 0.5
    os.makedirs(hyperparams["saving_folder"], exist_ok=True)
    return hyperparams


if __name__ == "__main__":
    datasets_folder = "./Data"
    #datasets_folder = "/shared/sets/datasets/"
    os.makedirs(datasets_folder, exist_ok=True)
    validation_size = 500
    use_data_augmentation = False
    use_cutout = False

    split_cifar100_list = prepare_split_cifar100_tasks(
        datasets_folder,
        validation_size,
        use_data_augmentation,
        use_cutout
    )