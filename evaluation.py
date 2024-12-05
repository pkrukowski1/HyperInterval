from Utils.prepare_non_forced_scenario_params import set_hyperparameters
from Utils.dataset_utils import *
from train_non_forced_scenario import (
    load_pickle_file,
    set_seed,
    intersection_of_embeds
)
from IntervalNets.interval_MLP import IntervalMLP
from IntervalNets.hmlp_ibp_wo_nesting import HMLP_IBP
from VanillaNets.ResNet18 import ResNetBasic
from IntervalNets.interval_ZenkeNet64 import IntervalZenkeNet
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from typing import Tuple, List, Dict

def load_dataset(dataset, path_to_datasets, hyperparameters):
    """""
    Load and prepare a specific dataset based on the given parameters.

    Parameters:
    -----------
        dataset: str
            Name of the dataset (e.g., "PermutedMNIST", "CIFAR-100", etc.).

        path_to_datasets: str
            Path to the dataset files.

        hyperparameters: dict
            Dictionary containing hyperparameters for dataset preparation:

    Returns:
    --------
        Tuple: A tuple containing the prepared dataset tasks.
    """
    if dataset == "PermutedMNIST":
        return prepare_permuted_mnist_tasks(
            path_to_datasets,
            hyperparameters["shape"],
            hyperparameters["number_of_tasks"],
            hyperparameters["padding"],
            hyperparameters["no_of_validation_samples"],
        )
    elif dataset == "CIFAR-100":
        return prepare_split_cifar100_tasks(
            path_to_datasets,
            validation_size=hyperparameters["no_of_validation_samples"],
            use_augmentation=hyperparameters["augmentation"],
        )
    elif dataset == "SplitMNIST":
        return prepare_split_mnist_tasks(
            path_to_datasets,
            validation_size=hyperparameters["no_of_validation_samples"],
            use_augmentation=hyperparameters["augmentation"],
            number_of_tasks=hyperparameters["number_of_tasks"],
        )
    elif dataset == "CIFAR100_FeCAM_setup":
        return prepare_split_cifar100_tasks_aka_FeCAM(
            path_to_datasets,
            number_of_tasks=hyperparameters["number_of_tasks"],
            no_of_validation_samples_per_class=hyperparameters[
                "no_of_validation_samples_per_class"
            ],
            use_augmentation=hyperparameters["augmentation"],
        )
    else:
        raise ValueError("This dataset is currently not handled!")


def prepare_target_network(hyperparameters, output_shape):
    """
    Prepare the target network based on specified hyperparameters.

    Parameters:
    -----------
        hyperparameters: dict
            Dictionary containing hyperparameters.

        output_shape int
            Number of output classes.

    Returns:
    --------
        torch.nn.Module: The prepared target network.
    """
    if hyperparameters["target_network"] == "MLP":
        target_network = IntervalMLP(
            n_in=hyperparameters["shape"],
            n_out=output_shape,
            hidden_layers=hyperparameters["target_hidden_layers"],
            use_bias=hyperparameters["use_bias"],
            no_weights=False,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ResNet":
        if hyperparameters["dataset"] == "TinyImageNet" or hyperparameters["dataset"] == "SubsetImageNet":
            mode = "tiny"
        elif hyperparameters["dataset"] == "CIFAR-100" or hyperparameters["dataset"] == "CIFAR100_FeCAM_setup":
            mode = "cifar"
        else:
            mode = "default"
        target_network = ResNetBasic(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
                use_bias=False,
                use_fc_bias=hyperparameters["use_bias"],
                bottleneck_blocks=False,
                num_classes=output_shape,
                num_feature_maps=[16, 16, 32, 64, 128],
                blocks_per_group=[2, 2, 2, 2],
                no_weights=False,
                use_batch_norm=hyperparameters["use_batch_norm"],
                projection_shortcut=True,
                bn_track_stats=False,
                cutout_mod=True,
                mode=mode,
        ).to(hyperparameters["device"])
    elif hyperparameters["target_network"] == "ZenkeNet":
        if hyperparameters["dataset"] in ["CIFAR-100", "CIFAR100_FeCAM_setup"]:
            architecture = "cifar"
        elif hyperparameters["dataset"] == "TinyImageNet":
            architecture = "tiny"
        else:
            raise ValueError("This dataset is currently not implemented!")
        target_network = IntervalZenkeNet(
            in_shape=(hyperparameters["shape"], hyperparameters["shape"], 3),
            num_classes=output_shape,
            arch=architecture,
            no_weights=False,
        ).to(hyperparameters["device"])
    else:
        raise NotImplementedError
    return target_network


def prepare_and_load_weights_for_models(
    path_to_stored_networks,
    path_to_datasets,
    number_of_model,
    dataset,
    seed,
):
    """
    Prepare the hypernetwork and target network, and load stored weights
    for both models. Also, load experiment hyperparameters.

    Parameters:
    -----------
        path_to_stored_networks: str
            Path for all models located in subfolders.

        number_of_model: int
            The number of the currently loaded model.
        dataset: str
            The name of the currently analyzed dataset, one of the following:
            'PermutedMNIST', 'SplitMNIST', 'CIFAR-100', 'CIFAR100_FeCAM_setup',
            'CIFAR-10', 'SubsetImageNet'
        seed: int
            Defines a seed value for deterministic calculations.

    Returns:
    --------
        dict: A dictionary with the following keys:
            - "list_of_CL_tasks": List of tasks from the dataset.
            - "hypernetwork": An instance of the HMLP class.
            - "hypernetwork_weights": Loaded weights for the hypernetwork.
            - "target_network": An instance of the MLP or ResNet class.
            - "target_network_weights": Loaded weights for the target network.
            - "hyperparameters": A dictionary with experiment hyperparameters.
    """

    assert dataset in [
        "PermutedMNIST",
        "SplitMNIST",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet"
    ]
    path_to_model = f"{path_to_stored_networks}{number_of_model}/"
    hyperparameters = set_hyperparameters(dataset, grid_search=False)
    
    set_seed(seed)
    # Load proper dataset
    dataset_tasks_list = load_dataset(
        dataset, path_to_datasets, hyperparameters
    )
    if hyperparameters["dataset"] == "SubsetImageNet":
        output_shape = dataset_tasks_list[0]._data["num_classes"]
    else:
        output_shape = list(
            dataset_tasks_list[0].get_train_outputs())[0].shape[0]

    # Build target network
    target_network = prepare_target_network(hyperparameters, output_shape)

    if not hyperparameters["use_chunks"]:
        hypernetwork = HMLP_IBP(
            target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=hyperparameters["embedding_sizes"][0],
            activation_fn=hyperparameters["activation_function"],
            layers=hyperparameters["hypernetworks_hidden_layers"][0],
            num_cond_embs=hyperparameters["number_of_tasks"],
        ).to(hyperparameters["device"])
    else:
        raise NotImplementedError
    # Load weights
    hnet_weights = load_pickle_file(
        f"{path_to_model}hypernetwork_"
        f'after_{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )
    target_weights = load_pickle_file(
        f"{path_to_model}target_network_after_"
        f'{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )

    perturbation_vectors = load_pickle_file(
        f"{path_to_model}perturbation_vectors_"
        f'after_{hyperparameters["number_of_tasks"] - 1}_task.pt'
    )

    hypernetwork._perturbated_eps_T = perturbation_vectors

    # Check whether the number of target weights is exactly the same like
    # the loaded weights
    for prepared, loaded in zip(
        [hypernetwork, target_network],
        [hnet_weights, target_weights],
    ):
        no_of_loaded_weights = 0
        for item in loaded:
            no_of_loaded_weights += item.shape.numel()
        assert prepared.num_params == no_of_loaded_weights
    return {
        "list_of_CL_tasks": dataset_tasks_list,
        "hypernetwork": hypernetwork,
        "hypernetwork_weights": hnet_weights,
        "target_network": target_network,
        "target_network_weights": target_weights,
        "hyperparameters": hyperparameters,
    }


def evaluate_target_network(
    target_network, network_input, weights, target_network_type, condition=None
):
    """
    Evaluate the target network on a single batch of samples.

    Parameters:
    -----------
        target_network: torch.nn.Module
            The target network.

        network_input: torch.Tensor
            Input data.

        weights: torch.Tensor
            Weights for the target network.

        target_network_type: str
            Type of the target network (e.g., "ResNet").

        condition: int, optional
            The number of the currently tested task for batch normalization.

    Returns:
    --------
        torch.Tensor: Logits from the target network.
    """
    if target_network_type == "ResNet":
        assert condition is not None
    if target_network_type == "ResNet":
        # Only ResNet needs information about the currently tested task
        return target_network.forward(
            network_input, weights=weights, condition=condition
        )
    else:
        return target_network.forward(network_input, weights=weights)

def plot_accuracy_curve(
        list_of_folders_path,
        save_path,
        filename,
        mode = 1,
        perturbation_sizes = [5.0, 10.0, 15.0, 20.0, 25.0],
        dataset_name = "PermutedMNIST-10",
        beta_params = [1.0, 0.1, 0.05, 0.01, 0.001],
        y_lim_max = 100.0,
        fontsize = 10,
        figsize = (6, 4)
):
    """
    Saves the accuracy curve for the specified mode.

    Parameters:
    ---------
        list_of_folders_path: List[str]
            A list with paths to stored results, one path for each seed.
        save_path: str
            The path where plots will be stored.
        filename: str
            Name of the saved plot.
        mode: int
            An integer having the following meanings:
            - 1: Experiments for different perturbation sizes in the
                 universal embedding setup.
            - 2: Experiments for different beta parameters in the
                 universal embedding setup.
            - 3: Experiments for different perturbation sizes in the
                 non-forced setup.
            - 4: Experiments for different beta parameters in the
                 non-forced setup.
        perturbation_sizes: List[float]
            A list with gamma hyperparameters.
        dataset_name: str
            A dataset name.
        beta_params: List[float]
            A list with beta hyperparameters.
        y_lim_max: float
            An upper limit for the Y-axis.
        fontsize: int
            Font size of titles and axes.
        figsize: Tuple[int]
            A tuple with width and height of the figures.

    Returns:
    --------
        None
    """

    assert mode in [1, 2, 3, 4], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    results_folder_seed_1 = os.listdir(list_of_folders_path[0])
    results_folder_seed_2 = os.listdir(list_of_folders_path[1])

    title = f'Results for {dataset_name}'

    if mode in [1,3]:
        params = perturbation_sizes
        title = f'{title} for $\gamma$ hyperparameters'
    elif mode in [2,4]:
        params = beta_params
        title = f'{title} for $\\beta$ hyperparameters'

    dataframe = {}
    fig, ax = plt.subplots(figsize=figsize)
    
    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i+1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i+1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i+1 for i in range(40)]
    

    for (results_seed_1, results_seed_2, param) in zip(
                                    results_folder_seed_1, 
                                    results_folder_seed_2, 
                                    params):
        if mode in [1, 2]:
            acc_path_1 = f'{list_of_folders_path[0]}/{results_seed_1}/results_intersection.csv'
            acc_path_2 = f'{list_of_folders_path[1]}/{results_seed_2}/results_intersection.csv'
        else:
            acc_path_1 = f'{list_of_folders_path[0]}/{results_seed_1}/results.csv'
            acc_path_2 = f'{list_of_folders_path[1]}/{results_seed_2}/results.csv'

        pd_results_1 = pd.read_csv(acc_path_1, sep=";")
        pd_results_2 = pd.read_csv(acc_path_2, sep=";")


        acc_1 = pd_results_1.loc[
                pd_results_1["after_learning_of_task"] == pd_results_1["after_learning_of_task"].max(), "accuracy"].values
        acc_2 = pd_results_2.loc[
                pd_results_2["after_learning_of_task"] == pd_results_2["after_learning_of_task"].max(), "accuracy"].values

        dataframe["accuracy"] = (acc_1 + acc_2)/2.0

        if "beta" in title:
            ax.plot(tasks_list, dataframe["accuracy"], label=f"$\\beta = {param}$")
        elif "gamma" in title:
            ax.plot(tasks_list, dataframe["accuracy"], label=f"$\gamma = {param}$")
        
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)
    ax.grid()
    ax.set_xticks(range(1, tasks_list[-1]+1))

    ax.set_ylim(top=y_lim_max)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()


def plot_accuracy_curve_for_diff_nesting_methods(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name = "PermutedMNIST-10",
        y_lim_max = 100.0,
        fontsize = 10
):
    """
    Saves the accuracy curve for different nesting methods, such as tanh or cosine.

    Parameters:
    ---------
        list_of_folders_path: List[str]
            A list with paths to stored results. The first two paths are for the tanh nesting method,
            and the last two are for the cosine nesting method.
        save_path: str
            The path where plots will be stored.
        filename: str
            Name of the saved plot.
        dataset_name: str
            A dataset name.
        y_lim_max: float
            An upper limit for the Y-axis.
        fontsize: int
            Font size for titles and axes.

    Returns:
    --------
        None
    """

    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Nesting results for {dataset_name}'

    tanh_dataframe = {}
    cos_dataframe = {}
    fig, ax = plt.subplots()
    
    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i+1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i+1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i+1 for i in range(40)]
    
    tanh_acc_path_1 = f'{list_of_folders_path[0]}/results_intersection.csv'
    tanh_acc_path_2 = f'{list_of_folders_path[1]}/results_intersection.csv'

    cos_acc_path_1 = f'{list_of_folders_path[2]}/results_intersection.csv'
    cos_acc_path_2 = f'{list_of_folders_path[3]}/results_intersection.csv'

    tanh_pd_results_1 = pd.read_csv(tanh_acc_path_1, sep=";")
    tanh_pd_results_2 = pd.read_csv(tanh_acc_path_2, sep=";")

    cos_pd_results_1 = pd.read_csv(cos_acc_path_1, sep=";")
    cos_pd_results_2 = pd.read_csv(cos_acc_path_2, sep=";")

    tanh_acc_1 = tanh_pd_results_1.loc[
                tanh_pd_results_1["after_learning_of_task"] == tanh_pd_results_1["after_learning_of_task"].max(), "accuracy"].values
    tanh_acc_2 = tanh_pd_results_2.loc[
            tanh_pd_results_2["after_learning_of_task"] == tanh_pd_results_2["after_learning_of_task"].max(), "accuracy"].values
    
    cos_acc_1 = cos_pd_results_1.loc[
                cos_pd_results_1["after_learning_of_task"] == cos_pd_results_1["after_learning_of_task"].max(), "accuracy"].values
    cos_acc_2 = cos_pd_results_2.loc[
            cos_pd_results_2["after_learning_of_task"] == cos_pd_results_2["after_learning_of_task"].max(), "accuracy"].values


    tanh_dataframe["accuracy"] = (tanh_acc_1 + tanh_acc_2)/2.0
    cos_dataframe["accuracy"] = (cos_acc_1 + cos_acc_2)/2.0

    ax.plot(tasks_list, tanh_dataframe["accuracy"], label=f"$\\tanh$")
    ax.plot(tasks_list, cos_dataframe["accuracy"], label=f"$\cos$")
        
    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)
    ax.grid()
    ax.set_xticks(range(1, tasks_list[-1]+1))

    ax.set_ylim(top=y_lim_max)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0), fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()

def plot_histogram_of_intervals(path_to_stored_networks,
                                save_path,
                                filename,
                                parameters,
                                num_bins: int = 10,
                                threshold_collapsed: float = 1e-8,
                                plot_vlines: bool = True,
                                figsize: Tuple[int] = (7,5),
                                rotation = False,
                                fontsize = 10
                                ):
    """
    Prepare a histogram plot based on specified parameters.

    Parameters:
    -----------
        path_to_stored_networks: str
            Path to the folder where the saved hypernetwork is stored.
        save_path: str
            Path to the folder where the histogram will be saved.
        filename: str
            Name of the saved histogram plot.
        parameters: dict
            A dictionary with experiment hyperparameters.
        num_bins: int
            Number of histogram bins (number of bars).
        threshold_collapsed: float
            Threshold for treating the interval as collapsed to a point.
        plot_vlines: bool
            If True, vertical lines will be drawn to separate each bar.
        figsize: tuple
            Represents the size of the plot.
        path: str
            Path to the saving directory.
        rotation: bool
            If True, OX axis ticks will be rotated by 45 degrees to the right.
        fontsize: int
            Size of font in title, OX, and OY axes.
    
    Returns:
    --------
        None
    """

    os.makedirs(save_path, exist_ok=True)
    target_network = prepare_target_network(parameters, parameters["out_shape"])

    eps = parameters["perturbated_epsilon"]
    dim_emb = parameters["embedding_size"]
    no_tasks = parameters["number_of_tasks"]
    sigma = 0.5 * eps / dim_emb

    hypernetwork = HMLP_IBP(
            perturbated_eps=eps,
            target_shapes=target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=dim_emb,
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=no_tasks)
    
    hnet_weights = load_pickle_file(
        f"{path_to_stored_networks}hypernetwork_"
        f'after_{parameters["number_of_tasks"] - 1}_task.pt'
     )
    
    embds = hnet_weights[:no_tasks]

    with torch.no_grad():
        radii = torch.stack([
            eps * F.softmax(torch.ones(dim_emb), dim=-1) for i in range(no_tasks)
        ], dim=0)

        embds_centers = torch.stack([
            sigma * torch.cos(embds[i].detach()) for i in range(no_tasks)
        ], dim=0)

        universal_embedding_lower, universal_embedding_upper = intersection_of_embeds(embds_centers - radii, embds_centers + radii)

        universal_embedding = (universal_embedding_lower + universal_embedding_upper)/2.0
        universal_radii = (universal_embedding_upper - universal_embedding_lower)/2.0
        
        W_lower, _, W_upper, _ = hypernetwork.forward(
            cond_input = universal_embedding.view(1, -1),
            return_extended_output = True,
            weights = hnet_weights,
            common_radii = universal_radii
        )

        plt.rcParams["figure.figsize"] = figsize
        epsilon = [(upper - lower).view(-1) for upper, lower in zip(W_upper, W_lower)]
        outputs = torch.cat(epsilon)
        num_zero_outputs = torch.where(outputs < threshold_collapsed, 1, 0).sum().item()
        ylabel = "Desity of intervals"

        outputs = outputs.detach().numpy()

        n, bins, patches = plt.hist(outputs,
                                    num_bins,
                                    density = False,
                                    color = "green",
                                    alpha = 0.5)

        plt.xlabel('Interval length', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)
        ticks = np.linspace(start=outputs.min(), stop=outputs.max(), num=num_bins+1)
        locs, _ = plt.yticks() 
        ax = plt.gca()
        precision_format = "{:.4f}"
        yticks = ax.get_yticks()
        ytick_labels = [precision_format.format(tick) for tick in locs/len(outputs)]
        ax.set_yticks(yticks[:-1])
        ax.set_yticklabels(ytick_labels[:-1])
        if plot_vlines:
            plt.vlines(x = ticks[:-1], ymin = 0, ymax = n, linestyle="--", color='black')
        plt.xticks(ticks)
        plt.title('Density of epsilon intervals', fontweight = "bold", fontsize=fontsize)
        if rotation:
            plt.xticks(rotation=45, ha='right')
        plt.legend([], title=f'Total number of coordinates = {len(outputs)}\nNumber of collapsed coordinates = {num_zero_outputs}', fontsize=fontsize)
        plt.tight_layout()
        plt.grid()
        plt.savefig(f'{save_path}{filename}')
        plt.close()

def plot_intervals_around_embeddings_for_trained_models(path_to_stored_networks,
                                                        save_path,
                                                        filename,
                                                        parameters,
                                                        figsize: Tuple[int] = (7,5),
                                                        fontsize = 10,
                                                        dims_to_plot = 5):
    """
    Prepare a histogram plot based on specified parameters.

    Parameters:
    -----------
        path_to_stored_networks: str
            Path to the folder where the saved hypernetwork is stored.
        save_path: str
            Path to the folder where the plot will be saved.
        filename: str
            Name of the saved plot.
        parameters: dict
            A dictionary with experiment hyperparameters.
        figsize: tuple
            Represents the size of the plot.
        path: str
            Path to the saving directory.
        fontsize: int
            Size of font in title, OX, and OY axes.
        dims_to_plot: int
            Number of the first dimensions to plot.
    
    Returns:
    --------
        None
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    no_tasks = parameters["number_of_tasks"]
    dim_emb  = parameters["embedding_size"]
    eps      = parameters["perturbated_epsilon"]
    sigma    = 0.5 * eps / dim_emb
    
    hnet_weights = load_pickle_file(
        f"{path_to_stored_networks}hypernetwork_"
        f'after_{parameters["number_of_tasks"] - 1}_task.pt'
    )

    with torch.no_grad():
        
        embds = hnet_weights[:no_tasks]

        radii = torch.stack([
            eps * F.softmax(torch.ones(dim_emb), dim=-1) for i in range(no_tasks)
        ], dim=0)

        embds_centers = torch.stack([
            sigma * torch.cos(embds[i].detach()) for i in range(no_tasks)
        ], dim=0)

        universal_embedding_lower, universal_embedding_upper = intersection_of_embeds(embds_centers - radii, embds_centers + radii)

        universal_embedding = (universal_embedding_lower + universal_embedding_upper)/2.0
        universal_radii = (universal_embedding_upper - universal_embedding_lower)/2.0

        universal_embedding = universal_embedding.cpu().detach().numpy()
        universal_radii = universal_radii.cpu().detach().numpy()

        universal_embedding = universal_embedding[:dims_to_plot]
        universal_radii = universal_radii[:dims_to_plot]
        
        # Create a plot
        fig = plt.figure(figsize=figsize)
        cm  = plt.get_cmap("gist_rainbow")

        colors = [cm(1.*i/(no_tasks + 1)) for i in range(no_tasks + 1)]

        for task_id, (tasks_embeddings, radii_per_emb) in enumerate(zip(embds_centers, radii)):
            
            tasks_embeddings = tasks_embeddings.cpu().detach().numpy()
            radii_per_emb = radii_per_emb.cpu().detach().numpy()

            tasks_embeddings = tasks_embeddings[:dims_to_plot]
            radii_per_emb = radii_per_emb[:dims_to_plot]

            # Generate an x axis
            x = [_ for _ in range(dims_to_plot)]

            # Create a scatter plot
            plt.scatter(x, tasks_embeddings, label=f"{task_id}-th task", marker="o", c=[colors[task_id]], alpha=0.3)

            for i in range(len(x)):
                plt.vlines(x[i], ymin=tasks_embeddings[i] - radii_per_emb[i],
                            ymax=tasks_embeddings[i] + radii_per_emb[i], linewidth=2, colors=[colors[task_id]], alpha=0.3)
        
        plt.scatter(x, universal_embedding, label=f"Intersection", marker="o", c=[colors[-1]], alpha=1.0)

        for i in range(len(x)):
            plt.vlines(x[i], ymin=universal_embedding[i] - universal_radii[i],
                        ymax=universal_embedding[i] + universal_radii[i], linewidth=2, colors=[colors[-1]], alpha=1.0)

        # Add labels and a legend
        plt.xlabel("Number of embedding's coordinate", fontsize=fontsize)
        plt.xticks(x, range(1, dims_to_plot+1))
        plt.ylabel("Embedding's value", fontsize=fontsize)
        plt.title(f'Intervals around embeddings', fontsize=fontsize)

        # Update legend to be at the top and split into two lines
        handles, _ = plt.gca().get_legend_handles_labels()
        ncol = (no_tasks + 1) // 2 + (no_tasks + 1) % 2

        labels = [f'{i+1}-th task' for i in range(no_tasks)]
        labels.append("Intersection")

        plt.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.132),
                   ncol=ncol, fontsize=fontsize, handletextpad=0.5, columnspacing=1.0)
        
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{save_path}/{filename}.png', dpi=300)
        plt.close()

def plot_accuracy_curve_with_confidence_intervals(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name="PermutedMNIST-10",
        mode=1,
        y_lim_max=100.0,
        fontsize=10,
        figsize=(6, 4),
        legend_loc = "upper right"
):
    """
    Saves the accuracy curve for the specified mode with 95% confidence intervals.

    Parameters:
    ---------
        list_of_folders_path: List[str]
            A list with paths to stored results, one path for each seed.
        save_path: str
            The path where plots will be stored.
        filename: str
            Name of the saved plot.
        dataset_name: str
            A dataset name.
        mode: int
            - 1: Results for the non-forced intervals method.
            - 2: Results for the universal embedding method.
        y_lim_max: float
            Upper limit of the Y-axis.
        fontsize: int
            Font size of titles and axes.
        figsize: Tuple[int]
            Tuple with width and height of the figures.
        legend_loc: str
            Location of the legend.

    Returns:
    --------
        None
    """

    assert mode in [1, 2], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "PermutedMNIST-100",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i + 1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i + 1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i + 1 for i in range(40)]
    elif dataset_name == "PermutedMNIST-100":
        tasks_list = [i+1 for i in range(100)]

    if dataset_name == "CIFAR100_FeCAM_setup":
        dataset_name = "CIFAR-100"

    if mode == 1:
        title = f'Results for {dataset_name} (task incremental learning)'
    elif mode == 2:
        
        title = f'Results for {dataset_name} (class incremental learning)'

    if mode == 1:
        file_suffix = "results.csv"
    else:
        file_suffix = "results_intersection.csv"

    results_list = []
    for folder in list_of_folders_path:
        acc_path = os.path.join(folder, file_suffix)
        results_list.append(pd.read_csv(acc_path, sep=";"))

    acc_just_after_training = []
    acc_after_all_training_sessions = []

    for pd_results in results_list:
        acc_just_after_training.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["tested_task"], "accuracy"].values)
        acc_after_all_training_sessions.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["after_learning_of_task"].max(), "accuracy"].values)

    acc_just_after_training = np.array(acc_just_after_training)
    acc_after_all_training_sessions = np.array(acc_after_all_training_sessions)

    mean_just_after_training = np.mean(acc_just_after_training, axis=0)
    mean_after_all_training_sessions = np.mean(acc_after_all_training_sessions, axis=0)

    ci_just_after_training = stats.t.interval(0.95, len(acc_just_after_training) - 1, loc=mean_just_after_training,
                                              scale=stats.sem(acc_just_after_training, axis=0))
    ci_after_all_training_sessions = stats.t.interval(0.95, len(acc_after_all_training_sessions) - 1,
                                                      loc=mean_after_all_training_sessions,
                                                      scale=stats.sem(acc_after_all_training_sessions, axis=0))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(tasks_list, mean_just_after_training, label="Just after training")
    ax.fill_between(tasks_list, ci_just_after_training[0], ci_just_after_training[1], alpha=0.2)
    ax.plot(tasks_list, mean_after_all_training_sessions, label="After training of all tasks")
    ax.fill_between(tasks_list, ci_after_all_training_sessions[0], ci_after_all_training_sessions[1], alpha=0.2)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)
    ax.grid()
    
    if dataset_name == "PermutedMNIST-100":
        ax.set_xticks(range(1, tasks_list[-1] + 1, 5))
    else:
        ax.set_xticks(range(1, tasks_list[-1] + 1))
    ax.set_ylim(top=y_lim_max)
    ax.legend(loc=legend_loc, fontsize=fontsize)
    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()

def plot_accuracy_curve_with_barplot(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name="TinyImageNet",
        mode=1,
        y_lim_max=100.0,
        fontsize=10,
        figsize=(6, 4),
        bar_width = 0.35
):
    """
    Saves the accuracy curve as a bar plot for the specified mode.

    Parameters:
    ---------
        list_of_folders_path: List[str]
            A list with paths to stored results, one path for each seed.
        save_path: str
            The path where plots will be stored.
        filename: str
            Name of the saved plot.
        dataset_name: str
            A dataset name.
        mode: int
            - 1: Results for the non-forced intervals method.
            - 2: Results for the universal embedding method.
        y_lim_max: float
            Upper limit of the Y-axis.
        fontsize: int
            Font size of titles and axes.
        figsize: Tuple[int]
            Tuple with width and height of the figures.
        bar_width: float
            Width of the bars in the bar plot.

    Returns:
    --------
        None
    """

    assert mode in [1, 2], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "PermutedMNIST-100",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Results for {dataset_name}'

    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i + 1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i + 1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i + 1 for i in range(40)]
    elif dataset_name == "PermutedMNIST-100":
        tasks_list = [i + 1 for i in range(100)]

    if mode == 1:
        file_suffix = "results.csv"
    else:
        file_suffix = "results_intersection.csv"

    results_list = []
    for folder_path in list_of_folders_path:
        acc_path = os.path.join(folder_path, file_suffix)
        results_list.append(pd.read_csv(acc_path, sep=";"))

    acc_just_after_training = []
    acc_after_all_training_sessions = []

    for pd_results in results_list:
        acc_just_after_training.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["tested_task"], "accuracy"].values)
        acc_after_all_training_sessions.append(pd_results.loc[
            pd_results["after_learning_of_task"] == pd_results["after_learning_of_task"].max(), "accuracy"].values)

    acc_just_after_training = np.array(acc_just_after_training)
    acc_after_all_training_sessions = np.array(acc_after_all_training_sessions)

    mean_just_after_training = np.mean(acc_just_after_training, axis=0)
    mean_after_all_training_sessions = np.mean(acc_after_all_training_sessions, axis=0)

    fig, ax = plt.subplots(figsize=figsize)

    bar_positions = np.arange(len(tasks_list))

    ax.bar(bar_positions - bar_width / 2, mean_just_after_training, bar_width, label="Just after training")
    ax.bar(bar_positions + bar_width / 2, mean_after_all_training_sessions, bar_width, label="After training of all tasks")

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel("Number of task", fontsize=fontsize)
    ax.set_ylabel("Accuracy [%]", fontsize=fontsize)

    # Set x-ticks at every 5 tasks
    ax.set_xticks(np.arange(0, len(tasks_list), 5))
    ax.set_xticklabels(np.arange(1, len(tasks_list) + 1, 5))

    ax.set_ylim(top=y_lim_max)
    ax.legend(loc="upper right", fontsize=fontsize)
    ax.grid(axis='y')

    plt.tight_layout()
    fig.savefig(f"{save_path}/{filename}")
    plt.close()

def plot_heatmap_for_n_runs(
        list_of_folders_path,
        save_path,
        filename,
        dataset_name="PermutedMNIST-10",
        mode=1,
        fontsize=10,
):
    """
    Saves the accuracy curve for the specified mode with 95% confidence intervals.

    Parameters:
    ---------
        list_of_folders_path: List[str]
            A list with paths to stored results, one path for each seed.
        save_path: str
            The path where plots will be stored.
        filename: str
            Name of the saved plot.
        dataset_name: str
            A dataset name.
        mode: int
            - 1: Results for the non-forced intervals method.
            - 2: Results for the universal embedding method.
        fontsize: int
            Font size of titles and axes.

    Returns:
    --------
        None
    """

    assert len(list_of_folders_path) == 5, "Please provide results on 5 runs!"
    assert mode in [1, 2], "Please provide the correct mode!"
    assert dataset_name in [
        "PermutedMNIST-10",
        "SplitMNIST",
        "CIFAR-100",
        "CIFAR100_FeCAM_setup",
        "SubsetImageNet",
        "TinyImageNet"
    ]

    os.makedirs(save_path, exist_ok=True)

    title = f'Mean accuracy for 5 runs of HyperInterval for {dataset_name}'

    if dataset_name in ["PermutedMNIST-10", "CIFAR-100"]:
        tasks_list = [i+1 for i in range(10)]
    elif dataset_name in ["SplitMNIST", "SubsetImageNet", "CIFAR100_FeCAM_setup"]:
        tasks_list = [i+1 for i in range(5)]
    elif dataset_name == "TinyImageNet":
        tasks_list = [i+1 for i in range(40)]
    
    if mode == 1:
        file_suffix = "results.csv"
    else:
        file_suffix = "results_intersection.csv"

    results_list = []
    for folder in list_of_folders_path:
        acc_path = os.path.join(folder, file_suffix)
        results_list.append(pd.read_csv(acc_path, sep=";"))
    
    dataframe = pd.read_csv(acc_path, sep=";")

    acc = []

    for pd_results in results_list:
        acc.append(pd_results["accuracy"].values)

    acc = np.mean(acc, axis=0)
    dataframe["accuracy"] = acc
    dataframe = dataframe.astype(
        {"after_learning_of_task": "int32",
         "tested_task": "int32"}
        )
    table = dataframe.pivot(
        "after_learning_of_task", "tested_task", "accuracy")
    sns.heatmap(table, annot=True, fmt=".1f")
    plt.xlabel("Number of the tested task", fontsize=fontsize)
    plt.ylabel("Number of the previously learned task", fontsize=fontsize)
    plt.xticks(ticks=np.arange(len(tasks_list))+0.5, labels=np.arange(1, len(tasks_list)+1), fontsize=fontsize)
    plt.yticks(ticks=np.arange(len(tasks_list))+0.5, labels=np.arange(1, len(tasks_list)+1), fontsize=fontsize)
    plt.tight_layout()
    plt.title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{filename}", dpi=300)
    plt.close()

def calculate_backward_transfer(dataframe):
    """
    Calculate backward transfer based on dataframe with results
    containing columns: 'after_learning_of_task', 'tested_task',
    'accuracy'.
    ---
    BWT = 1/(N-1) * sum_{i=1}^{N-1} A_{N,i} - A_{i,i}
    where N is the number of tasks, A_{i,j} is the result
    for the network trained on the i-th task and tested
    on the j-th task.

    Returns a float with backward transfer result.

    Reference: https://github.com/gmum/HyperMask/blob/main/evaluation.py

    """
    backward_transfer = 0
    number_of_last_task = int(dataframe.max()["after_learning_of_task"])
    # Indeed, number_of_last_task represents the number of tasks - 1
    # due to the numeration starting from 0
    for i in range(number_of_last_task + 1):
        trained_on_last_task = dataframe.loc[
            (dataframe["after_learning_of_task"] == number_of_last_task)
            & (dataframe["tested_task"] == i)
        ]["accuracy"].values[0]
        trained_on_the_same_task = dataframe.loc[
            (dataframe["after_learning_of_task"] == i) & (dataframe["tested_task"] == i)
        ]["accuracy"].values[0]
        backward_transfer += trained_on_last_task - trained_on_the_same_task
    backward_transfer /= number_of_last_task
    return backward_transfer

def calculate_BWT_different_files(paths, forward=True):
    """
    Calculate mean backward transfer with corresponding
    sample standard deviations based on results saved in .csv files.
    
    Reference: https://github.com/gmum/HyperMask/blob/main/evaluation.py

    Parameters :
    ---------
      paths: List
        Contains path to the results files.
      forward: Optional, Boolean
        Defines whether forward transfer will be calculated.

    Returns:
    --------
      BWTs: List[float]
        Contains consecutive backward transfer values.
    """
    BWTs = []
    for path in paths:
        dataframe = pd.read_csv(path, sep=";", index_col=0)
        BWTs.append(calculate_backward_transfer(dataframe))
    print(
        f"Mean backward transfer: {np.mean(BWTs)}, "
        f"population standard deviation: {np.std(BWTs)}"
    )
    return BWTs

def get_subdirs(path: str = "./") -> List[str]:
    """
    Find the immediate subdirectories given a path to a directory of interest.

    Parameters :
    ---------
      path: str
        A path to the directory of interest.

    Returns:
    --------
      subdirs: List[str]
        Contains names of subdirectories of the given path directory.
    """
    subdirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return subdirs

def calculate_BWT_different_datasets(datasets_folder: str = './HINT_models') -> Dict:
    """
    This function assumes that the dataset_folder contains directories
    such as "CIFAR100/known_task_id/1/results.csv".

    Parameters :
    ---------
      datasets_folder: str
        A path to the datasets folders with results.

    Returns:
    --------
      mean_results_dict: Dict
        Contains average backward transfer values with standard deviation per dataset and available scenario.
    """

    datasets = get_subdirs(datasets_folder)
    mean_results_dict = {}

    for dataset in datasets:
        temp_path = f"{datasets_folder}/{dataset}"
        scenarios = get_subdirs(temp_path)

        for scenario in scenarios:
            temp_scenario_path = f"{temp_path}/{scenario}"
            seeds = get_subdirs(temp_scenario_path)
            paths = [f"{temp_scenario_path}/{seed}/results.csv" for seed in seeds]

            bwt = calculate_BWT_different_files(paths, forward = False)
            mean_results_dict[f"{dataset}: {scenario}"] = [np.round(np.mean(bwt),3), np.round(np.std(bwt),2)]

    (pd.DataFrame.from_dict(data=mean_results_dict, orient='index', columns=['Avg', 'Std']).to_csv(f'{datasets_folder}/avg_bwt_results.csv', header=True))
    return mean_results_dict

if __name__ == "__main__":
   
   pass