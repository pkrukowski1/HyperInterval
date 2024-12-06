import torch

from IntervalNets.interval_MLP import IntervalMLP
from IntervalNets.interval_modules import parse_logits
from IntervalNets.interval_ResNet import IntervalResNetBasic
from IntervalNets.hmlp_ibp_wo_nesting import HMLP_IBP

import os
import time
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from itertools import product

from VanillaNets.ResNet18 import ResNetBasic
from VanillaNets.AlexNet import AlexNet
from VanillaNets.LeNet_300_100 import LeNet
from hypnettorch.mnets.mlp import MLP
from hypnettorch.mnets.resnet_imgnet import ResNetIN

import Utils.hnet_middle_regularizer as hreg
from LossFunctions.regression_loss_function import IntervalMSELoss
from Utils.prepare_non_forced_scenario_params import set_hyperparameters
from Utils.dataset_utils import *
from Utils.handy_functions import *

def train_single_task(hypernetwork,
                      target_network,
                      criterion,
                      parameters,
                      dataset_list_of_tasks,
                      current_no_of_task):
    """
    Train a hypernetwork that generates the weights of the target neural network.
    This module operates on a single training task with a specific number.

    Parameters:
    -----------
    hypernetwork: hypnettorch.hnets module (e.g., mlp_hnet.MLP)
        A hypernetwork that generates weights for the target network.
    target_network: hypnettorch.mnets module (e.g., mlp.MLP)
        A target network that finally performs regression.
    criterion: torch.nn module
        Implements a loss function (e.g., CrossEntropyLoss).
    parameters: dictionary
        Contains necessary hyperparameters describing an experiment.
    dataset_list_of_tasks: module
        Contains a list of tasks for the CL scenario (e.g., permuted_mnist.PermutedMNISTList).
    current_no_of_task: int
        Specifies the number of the currently solving task.

    Returns:
    --------
    hypernetwork: modified module of the hypernetwork.
    target_network: modified module of the target network.
    """
    # Optimizer cannot be located outside of this function because after
    # deep copy of the network it needs to be reinitialized
    if parameters["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            [*hypernetwork.parameters(), *target_network.parameters()],
            lr=parameters["learning_rate"]
        )
    elif parameters["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            [*hypernetwork.parameters(), *target_network.parameters()],
            lr=parameters["learning_rate"]
        )
    else:
        raise ValueError("Wrong type of the selected optimizer!")
    if parameters["best_model_selection_method"] == "val_loss":
        # Store temporary best models to keep those with the highest
        # validation MSE.
        best_hypernetwork = deepcopy(hypernetwork).to(parameters["device"])
        best_target_network = deepcopy(target_network).to(parameters["device"])
        best_val_mse_loss = np.inf
        
    elif parameters["best_model_selection_method"] != "last_model":
        raise ValueError("Wrong value of best_model_selection_method parameter!")
    # Compute targets for the regularization part of loss before starting
    # the training of a current task
    hypernetwork.train()
    target_network.train()
    print(f"task: {current_no_of_task}")
    if current_no_of_task > 0:
        previous_hnet_theta = None
        previous_hnet_embeddings = None

        # Save previous hnet weights
        hypernetwork._prev_hnet_weights = deepcopy(hypernetwork.unconditional_params)

        middle_reg_targets = hreg.get_current_targets(
                                        task_id=current_no_of_task,
                                        hnet=hypernetwork,
                                        eps=parameters["perturbated_epsilon"])

    if (parameters["target_network"] == "ResNet") and \
       parameters["use_batch_norm"]:
        use_batch_norm_memory = True
    else:
        use_batch_norm_memory = False
    current_dataset_instance = dataset_list_of_tasks[current_no_of_task]
    # If training through a given number of epochs is desired
    # the number of iterations has to be calculated
    if parameters["number_of_epochs"] is not None:
        no_of_iterations_per_epoch, parameters["number_of_iterations"] = \
            calculate_number_of_iterations(
                current_dataset_instance.num_train_samples,
                parameters["batch_size"],
                parameters["number_of_epochs"]
            )
        # Scheduler can be set only when the number of epochs is given
        if parameters["lr_scheduler"]:
            current_epoch = 0

    iterations_to_adjust = (parameters["number_of_iterations"] // 2)
    iterations_to_adjust = int(iterations_to_adjust)

    for iteration in range(parameters["number_of_iterations"]):
        current_batch = current_dataset_instance.next_train_batch(
            parameters["batch_size"]
        )
        tensor_input = current_dataset_instance.input_to_torch_tensor(
            current_batch[0], parameters["device"], mode="train"
        )
        tensor_output = current_dataset_instance.output_to_torch_tensor(
            current_batch[1], parameters["device"], mode="train"
        )

        optimizer.zero_grad()

        # Adjust kappa and epsilon
        if iteration < iterations_to_adjust:
            kappa = max(1 - 0.00005*iteration, hyperparameters["kappa"])
            eps   = (iteration / (iterations_to_adjust-1)) * parameters["perturbated_epsilon"]
        else:
            kappa = parameters["kappa"]
            eps   = parameters["perturbated_epsilon"]

        # Get weights, lower weights, upper weights and predicted radii
        # returned by the hypernetwork
        lower_weights, target_weights, upper_weights, _ = hypernetwork.forward(cond_id=current_no_of_task, 
                                                                                return_extended_output=True,
                                                                                perturbated_eps=eps)

        if parameters["full_interval"]:
            predictions = target_network.forward(x=tensor_input,
                                                upper_weights=upper_weights,
                                                middle_weights=target_weights,
                                                lower_weights=lower_weights)
            
            lower_pred, middle_pred, upper_pred = parse_logits(predictions)
        else:
            lower_pred, middle_pred, upper_pred = reverse_predictions(target_network,
                                                                      tensor_input,
                                                                      lower_weights,
                                                                      target_weights,
                                                                      upper_weights)
        
        
        # We need to check wheter the distance between the lower weights
        # and the upper weights isn"t collapsed into "one point" (short interval)
        loss_weights = 0.0
        for W_u, W_l in zip(upper_weights, lower_weights):
            loss_weights += (W_u - W_l).abs().mean()

        loss_current_task = criterion(
            y=tensor_output,
            z_l=lower_pred,
            z_u=upper_pred,
            kappa=kappa
        )

        # Get the worst case error
        worst_case_error = criterion.worst_case_error
            
        loss_regularization = 0.

        if current_no_of_task > 0:
            loss_regularization = hreg.calc_fix_target_reg(
                hypernetwork, current_no_of_task,
                middle_targets=middle_reg_targets,
                mnet=target_network, prev_theta=previous_hnet_theta,
                prev_task_embs=previous_hnet_embeddings,
                eps=parameters["perturbated_epsilon"]
            )
        
        # Calculate total loss
        loss = loss_current_task + \
            parameters["beta"] * loss_regularization / max(1, current_no_of_task)
        
        # Save total loss to file
        if iteration > 0 or current_no_of_task > 0:
            header = ""
        else:
            header = "current_no_of_task;iteration;total_loss;loss_current_task;" + \
                     "worst_case_error;loss_regularization;loss_weights"
            
        append_row_to_file(
        filename=f'{parameters["saving_folder"]}total_loss',
        elements=f"{current_no_of_task};{iteration};{loss};{loss_current_task};"
                 f"{worst_case_error};{loss_regularization};{loss_weights}",
        header=header
        )  

        loss.backward()
        optimizer.step()

        if iteration % 500 == 499:
        # if iteration % 10 == 9:
            # Plot intervals over tasks" embeddings plot
            interval_plot_save_path = f'{parameters["saving_folder"]}/plots/'
            plot_universal_embedding = iteration >= iterations_to_adjust

            plot_intervals_around_embeddings(hypernetwork=hypernetwork,
                                            parameters=parameters,
                                            save_folder=interval_plot_save_path,
                                            iteration=iteration,
                                            current_task=current_no_of_task,
                                            plot_universal_embedding=plot_universal_embedding)

        if parameters["number_of_epochs"] is None:
            condition = (iteration % 10 == 0) or \
                        (iteration == (parameters["number_of_iterations"] - 1))
        else:
            condition = (iteration % 10 == 0) or \
                        (iteration == (parameters["number_of_iterations"] - 1)) or \
                        (((iteration + 1) % no_of_iterations_per_epoch) == 0)

        if condition:
            if parameters["number_of_epochs"] is not None:
                current_epoch = (iteration + 1) // no_of_iterations_per_epoch
                print(f"Current epoch: {current_epoch}")


            # Save distance between the upper and lower weights to file
            append_row_to_file(
            filename=f'{parameters["saving_folder"]}upper_lower_weights_distance',
            elements=f'{current_no_of_task};{iteration};{loss_weights}'
            )

            mse_loss = calculate_mse_loss(
                criterion,
                current_dataset_instance,
                target_network,
                lower_weights,
                target_weights,
                upper_weights,
                parameters={
                    "device": parameters["device"],
                    "use_batch_norm_memory": use_batch_norm_memory,
                    "number_of_task": current_no_of_task,
                    "full_interval": parameters["full_interval"]
                },
                evaluation_dataset="validation")
            
            print(f"Task {current_no_of_task}, iteration: {iteration + 1}, "
                f" loss: {loss.item()}, "
                f" worst case error: {worst_case_error}, "
                f" perturbated_epsilon: {eps}")
            # If the interval MSE loss on the validation dataset is lower
            # than previously
            if parameters["best_model_selection_method"] == "val_loss":
                if mse_loss < best_val_mse_loss:
                    best_val_mse_loss = mse_loss
                    best_hypernetwork = deepcopy(hypernetwork)
                    best_target_network = deepcopy(target_network)

    if parameters["best_model_selection_method"] == "val_loss":
        return best_hypernetwork, best_target_network
    else:
        return hypernetwork, target_network


def build_multiple_task_experiment(dataset_list_of_tasks,
                                   parameters,
                                   use_chunks=False):
    """
    Perform a series of experiments based on the hyperparameters.

    Parameters:
    -----------
    parameters: dict
        Contains multiple experiment hyperparameters.

    Returns:
    --------
    Tuple[hypnettorch.hnets, hypnettorch.mnets, pd.DataFrame]
        A tuple containing:
        - hypernetwork: A learned hypernetwork that generates weights for the target network.
        - target_network: A learned target network that performs regression.
        - dataframe: A Pandas DataFrame with single results from consecutive evaluations for all previous tasks.
    """
    if parameters["dataset"] == "SubsetImageNet":
        output_shape = dataset_list_of_tasks[0]._data["num_classes"]
    else:
        output_shape = list(
            dataset_list_of_tasks[0].get_train_outputs())[0].shape[0]

    # Create a target network
    if parameters["target_network"] == "MLP":
        if parameters["full_interval"]:
            target_network = IntervalMLP(n_in=parameters["input_shape"],
                                n_out=output_shape,
                                hidden_layers=parameters["target_hidden_layers"],
                                use_bias=parameters["use_bias"],
                                no_weights=True,
                                use_batch_norm=parameters["use_batch_norm"],
                                bn_track_stats=False,
                                dropout_rate=parameters["dropout_rate"]).to(parameters["device"])
        else:
            target_network = MLP(n_in=parameters["input_shape"],
                                n_out=output_shape,
                                hidden_layers=parameters["target_hidden_layers"],
                                use_bias=parameters["use_bias"],
                                no_weights=True,
                                use_batch_norm=parameters["use_batch_norm"],
                                bn_track_stats=False,
                                dropout_rate=parameters["dropout_rate"]).to(parameters["device"])
    elif parameters["target_network"] == "ResNet":
        if parameters["dataset"] == "TinyImageNet" or parameters["dataset"] == "SubsetImageNet":
            mode = "tiny"
        elif parameters["dataset"] in [
            "CIFAR100",
            "CIFAR100_FeCAM_setup",
            "CIFAR10",
            "CUB200"
        ]:
            mode = "cifar"
        else:
            mode = "default"
        
        if parameters["full_interval"]:
            target_network = IntervalResNetBasic(
                in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
                use_bias=False,
                use_fc_bias=parameters["use_bias"],
                bottleneck_blocks=False,
                num_classes=output_shape,
                num_feature_maps=[16, 16, 32, 64, 128],
                blocks_per_group=[2, 2, 2, 2],
                no_weights=True,
                use_batch_norm=parameters["use_batch_norm"],
                projection_shortcut=True,
                bn_track_stats=False,
                cutout_mod=True,
                mode=mode,
            ).to(parameters["device"])
        else:
            target_network = ResNetBasic(
                in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
                use_bias=False,
                use_fc_bias=parameters["use_bias"],
                bottleneck_blocks=False,
                num_classes=output_shape,
                num_feature_maps=[16, 16, 32, 64, 128],
                blocks_per_group=[2, 2, 2, 2],
                no_weights=True,
                use_batch_norm=parameters["use_batch_norm"],
                projection_shortcut=True,
                bn_track_stats=False,
                cutout_mod=True,
                mode=mode,
            ).to(parameters["device"])
    elif parameters["target_network"] == "ResNetIN":
        assert not parameters["full_interval"], "Full interval implementation of the chosen target network is not supported right now!"
        target_network = ResNetIN(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            num_classes=output_shape,
            use_fc_bias=parameters["use_bias"],
            bottleneck_blocks=False,
            no_weights=True,
            use_batch_norm=parameters["use_batch_norm"],
            bn_track_stats=False,
            cutout_mod=False
        )
    elif parameters["target_network"] == "ZenkeNet":
        raise ValueError("ZenkeNet is not supported right now!")
    elif parameters["target_network"] == "AlexNet" \
        and not parameters["full_interval"]:
        target_network = AlexNet(
            in_shape=(parameters["input_shape"], parameters["input_shape"], 3),
            num_classes=output_shape,
            no_weights=True,
            use_batch_norm=parameters["use_batch_norm"],
            bn_track_stats=False,
            distill_bn_stats=False
        )
    elif parameters["target_network"] == "LeNet" \
        and not parameters["full_interval"]:
        target_network = LeNet(
            in_shape=(28, 28, 1),
            num_classes=output_shape
        )
    if not use_chunks:
        hypernetwork = HMLP_IBP(
            perturbated_eps=parameters["perturbated_epsilon"],
            target_shapes=target_network.param_shapes,
            uncond_in_size=0,
            cond_in_size=parameters["embedding_size"],
            activation_fn=parameters["activation_function"],
            layers=parameters["hypernetwork_hidden_layers"],
            num_cond_embs=parameters["number_of_tasks"]).to(
                parameters["device"])
    else:
        raise Exception("Not implemented yet!")

    criterion = IntervalMSELoss()
    dataframe = pd.DataFrame(columns=[
        "after_learning_of_task", "tested_task", "MSE"])
    
    if (parameters["target_network"] == "ResNet") and \
       parameters["use_batch_norm"]:
        use_batch_norm_memory = True
    else:
        use_batch_norm_memory = False
    hypernetwork.train()
    target_network.train()

    # Declare total number of tasks
    no_tasks = parameters["number_of_tasks"]

    for no_of_task in range(no_tasks):

        hypernetwork, target_network = train_single_task(
            hypernetwork,
            target_network,
            criterion,
            parameters,
            dataset_list_of_tasks,
            no_of_task
        )

        if no_of_task <= (parameters["number_of_tasks"] - 1):
            
            if no_of_task > 0:
                # Remove previous parameters
                os.remove(
                    path=f'{parameters["saving_folder"]}/'
                        f'hypernetwork_after_{no_of_task-1}_task.pt'
                )

                os.remove(
                    path=f'{parameters["saving_folder"]}/'
                        f'target_network_after_{no_of_task-1}_task.pt'
                )

                os.remove(
                    path=f'{parameters["saving_folder"]}/'
                        f'perturbation_vectors_after_{no_of_task-1}_task.pt'
                )

            # Save current state of networks
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f'hypernetwork_after_{no_of_task}_task',
                hypernetwork.weights
            )
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f'target_network_after_{no_of_task}_task',
                target_network.weights
            )
            write_pickle_file(
                f'{parameters["saving_folder"]}/'
                f'perturbation_vectors_after_{no_of_task}_task',
                hypernetwork._perturbated_eps_T
            )
        
        # Freeze the already learned embeddings and radii
        hypernetwork.detach_tensor(idx = no_of_task)

        # Evaluate previous tasks
        dataframe = evaluate_previous_regression_tasks(
            criterion,
            hypernetwork,
            target_network,
            dataframe,
            dataset_list_of_tasks,
            parameters={
                "device": parameters["device"],
                "use_batch_norm_memory": use_batch_norm_memory,
                "number_of_task": no_of_task,
                "perturbated_epsilon": parameters["perturbated_epsilon"],
                "full_interval": parameters["full_interval"],
            }
        )
        dataframe = dataframe.astype({
            "after_learning_of_task": "int",
            "tested_task": "int"
        })
        dataframe.to_csv(f'{parameters["saving_folder"]}/'
                         f"results.csv",
                         sep=";")
              

        # Plot intervals over tasks" embeddings plot
        interval_plot_save_path = f'{parameters["saving_folder"]}/plots/'
        plot_intervals_around_embeddings(hypernetwork=hypernetwork,
                                        parameters=parameters,
                                        save_folder=interval_plot_save_path,
                                        current_task=no_of_task,
                                        plot_universal_embedding=True)

    return hypernetwork, target_network, dataframe


def main_running_experiments(parameters):
    
    """
    Perform a series of experiments based on the hyperparameters.

    Parameters:
    ----------
      parameters: Dict
        Contains multiple experiment hyperparameters

    Returns:
    --------
        Returns learned hypernetwork, target network and a dataframe
        with single results.
    """

    if parameters["dataset"] == "ToyRegression1D":
        dataset_tasks_list = prepare_toy_regression_tasks(
            seed=parameters["seed"],
            no_of_validation_samples=parameters["no_of_validation_samples"]
        )
    elif parameters["dataset"] == "GaussianDataset":
        dataset_tasks_list = prepare_gaussian_regression_tasks(
            seed=parameters["seed"],
            no_of_validation_samples=parameters["no_of_validation_samples"]
        )
    else:
        raise ValueError("Wrong name of the dataset!")
    
    start_time = time.time()

    hypernetwork, target_network, dataframe = build_multiple_task_experiment(
        dataset_tasks_list,
        parameters,
        use_chunks=parameters["use_chunks"]
    )

    #TODO: dokończyć, funkcja powinna przyjmować parametry modelu jako input i liczyć output wewnętrznie
    plot_regression_results(
        x: List[np.ndarray],  # One array per task
        y_pred: List[np.ndarray],  # One array per task
        dataset_name = parameters["dataset"],
        save_path = f"./{parameters["dataset"]}.png",
        t = 2.0
    )

    elapsed_time = time.time() - start_time

    # Calculate statistics of grid search results
    no_of_last_task = parameters["number_of_tasks"] - 1
    accuracies = dataframe.loc[
        dataframe["after_learning_of_task"] == no_of_last_task
    ]["MSE"].values
    row_with_results = (
        f"{dataset_tasks_list[0].get_identifier()};"
        f'{parameters["augmentation"]};'
        f'{parameters["embedding_size"]};'
        f'{parameters["seed"]};'
        f'{str(parameters["hypernetwork_hidden_layers"]).replace(" ", "")};'
        f'{parameters["target_network"]};'
        f'{str(parameters["target_hidden_layers"]).replace(" ", "")};'
        f'{parameters["resnet_number_of_layer_groups"]};'
        f'{parameters["resnet_widening_factor"]};'
        f'{parameters["best_model_selection_method"]};'
        f'{parameters["optimizer"]};'
        f'{parameters["activation_function"]};'
        f'{parameters["learning_rate"]};{parameters["batch_size"]};'
        f'{parameters["beta"]};'
        f'{parameters["perturbated_epsilon"]};'
        f'{parameters["kappa"]};'
        f"{np.mean(accuracies)};{np.std(accuracies)};"
        f"{elapsed_time}"
    )
    append_row_to_file(
        f'{parameters["grid_search_folder"]}'
        f'{parameters["summary_results_filename"]}.csv',
        row_with_results
    )
    
    return hypernetwork, target_network, dataframe


if __name__ == "__main__":
    dataset = "GaussianDataset"  # "ToyRegression1D", "GaussianDataset"
    part = 0
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # Generate timestamp
    create_grid_search = False

    if create_grid_search:
        summary_results_filename = "grid_search_results"
    else:
        summary_results_filename = "summary_results"
    hyperparameters = set_hyperparameters(
        dataset,
        grid_search=create_grid_search,
    )

    header = (
        "dataset_name;augmentation;embedding_size;seed;hypernetwork_hidden_layers;"
        "use_chunks;target_network;target_hidden_layers;"
        "layer_groups;widening;final_model;optimizer;"
        "hypernet_activation_function;learning_rate;batch_size;beta;mean_mse;std_mse;"
        "elapsed_time"
    )

    append_row_to_file(
        f'{hyperparameters["saving_folder"]}/{summary_results_filename}.csv',
        header
    )

    for no, elements in enumerate(
        product(hyperparameters["embedding_sizes"],
                hyperparameters["learning_rates"],
                hyperparameters["betas"],
                hyperparameters["hypernetworks_hidden_layers"],
                hyperparameters["batch_sizes"],
                hyperparameters["seed"],
                hyperparameters["perturbated_epsilon"],
                hyperparameters["dropout_rate"])
    ):
        embedding_size = elements[0]
        learning_rate = elements[1]
        beta = elements[2]
        hypernetwork_hidden_layers = elements[3]
        batch_size = elements[4]
        perturbated_eps = elements[6]
        dropout_rate = elements[7]
        
        # Of course, seed is not optimized but it is easier to prepare experiments
        # for multiple seeds in such a way
        seed = elements[5]

        parameters = {
            "input_shape": hyperparameters["shape"],
            "augmentation": hyperparameters["augmentation"],
            "number_of_tasks": hyperparameters["number_of_tasks"],
            "seed": seed,
            "dataset": dataset,
            "hypernetwork_hidden_layers": hypernetwork_hidden_layers,
            "activation_function": hyperparameters["activation_function"],
            "use_chunks": hyperparameters["use_chunks"],
            "target_network": hyperparameters["target_network"],
            "target_hidden_layers": hyperparameters["target_hidden_layers"],
            "resnet_number_of_layer_groups": hyperparameters["resnet_number_of_layer_groups"],
            "resnet_widening_factor": hyperparameters["resnet_widening_factor"],
            "learning_rate": learning_rate,
            "best_model_selection_method": hyperparameters["best_model_selection_method"],
            "lr_scheduler": hyperparameters["lr_scheduler"],
            "batch_size": batch_size,
            "no_of_validation_samples": hyperparameters[
                "no_of_validation_samples"
            ],
            "number_of_epochs": hyperparameters["number_of_epochs"],
            "number_of_iterations": hyperparameters["number_of_iterations"],
            "embedding_size": embedding_size,
            "optimizer": hyperparameters["optimizer"],
            "beta": beta,
            "padding": hyperparameters["padding"],
            "use_bias": hyperparameters["use_bias"],
            "use_batch_norm": hyperparameters["use_batch_norm"],
            "device": hyperparameters["device"],
            "saving_folder": f'{hyperparameters["saving_folder"]}/{TIMESTAMP}/{no}/',
            "grid_search_folder": hyperparameters["saving_folder"],
            "summary_results_filename": summary_results_filename,
            "perturbated_epsilon": perturbated_eps,
            "kappa": hyperparameters["kappa"],
            "dropout_rate": dropout_rate,
            "full_interval": hyperparameters["full_interval"]
        }

        if "no_of_validation_samples_per_class" in hyperparameters:
            parameters["no_of_validation_samples_per_class"] = hyperparameters[
                "no_of_validation_samples_per_class"
            ]

        os.makedirs(f'{parameters["saving_folder"]}', exist_ok=True)
        # start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_parameters(parameters["saving_folder"],
                        parameters,
                        name=f"parameters.csv")

        # Important! Seed is set before the preparation of the dataset!
        if seed is not None:
            set_seed(seed)

        hypernetwork, target_network, dataframe = \
            main_running_experiments(parameters)
