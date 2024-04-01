import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from typing import cast

def parse_predictions(x):
    """
    Parse the output of a target network to get lower, middle and upper predictions

    Arguments:
    ----------
        *x*: (torch.Tensor) the output to be parsed
    
    Returns:
    --------
        a tuple of lower, middle and upper predictions
    """

    return map(lambda x_: cast(torch.Tensor, x_.rename(None)), x.unbind("bounds"))  # type: ignore

def translate_output_MNIST_classes(relative_labels, task, mode):
    """
    Translate relative labels of form {0, 1} to the real labels
    of Split MNIST dataset.

    Arguments:
    ----------
       *labels*: (Numpy array | list) contains labels of the form
       *task*: (int) number of the currently calculated task,
               starting from 0
       *mode*: (str) "permuted" or "split", depending on the desired
               dataset
    """
    assert mode in ["permuted", "split"]
    if mode == "permuted":
        total_no_of_classes = 100
        no_of_classes_per_task = 10
        # Even if the classifier indicates '0' but from the wrong task
        # it has to get a penalty. Therefore, in Permuted MNIST there
        # are 100 unique classes.
    elif mode == "split":
        total_no_of_classes = 10
        no_of_classes_per_task = 2
    class_orders = [i for i in range(total_no_of_classes)]
    currently_used_classes = class_orders[
        (no_of_classes_per_task * task) : (no_of_classes_per_task * (task + 1))
    ]
    y_translated = np.array(
        [currently_used_classes[i] for i in relative_labels]
    )
    return y_translated

def get_task_and_class_prediction_based_on_logits(
    lower_inferenced_logits_of_all_tasks,
    middle_inferenced_logits_of_all_tasks,
    upper_inferenced_logits_of_all_tasks,
    setup, dataset
):
    """
    Get task prediction for consecutive samples based on entropy values
    of the output classification layer of the target network.

    Arguments:
    ----------
       *inferenced_logits_of_all_tasks*: shape: (number of tasks,
                            number of samples, number of output heads)
       *setup*: (int) defines how many tasks were performed in this
                experiment (in total)
       *dataset*: (str) name of the dataset for proper class translation

    Returns:
    --------
       *predicted_tasks*: torch.Tensor with the prediction of tasks for
                          consecutive samples
       *predicted_classes*: torch.Tensor with the prediction of classes for
                            consecutive samples.
       Positions of samples in the two above Tensors are the same.
    """
    predicted_classes, predicted_tasks = [], []
    number_of_samples = middle_inferenced_logits_of_all_tasks.shape[1]

    for no_of_sample in range(number_of_samples):
        lower_task_entropies = torch.zeros(middle_inferenced_logits_of_all_tasks.shape[0])
        middle_task_entropies = torch.zeros(middle_inferenced_logits_of_all_tasks.shape[0])
        upper_task_entropies = torch.zeros(middle_inferenced_logits_of_all_tasks.shape[0])

        lower_all_task_single_output_sample = lower_inferenced_logits_of_all_tasks[
            :, no_of_sample, :
        ]

        middle_all_task_single_output_sample = middle_inferenced_logits_of_all_tasks[
            :, no_of_sample, :
        ]

        upper_all_task_single_output_sample = upper_inferenced_logits_of_all_tasks[
            :, no_of_sample, :
        ]

        # Calculate entropy based on results from all tasks
        for no_of_inferred_task in range(middle_task_entropies.shape[0]):
            lower_softmaxed_inferred_task = F.softmax(
                lower_all_task_single_output_sample[no_of_inferred_task], dim=-1
            )
            lower_task_entropies[no_of_inferred_task] = -1 * torch.sum(
                lower_softmaxed_inferred_task * torch.log(lower_softmaxed_inferred_task)
            )

            middle_softmaxed_inferred_task = F.softmax(
                middle_all_task_single_output_sample[no_of_inferred_task], dim=-1
            )
            middle_task_entropies[no_of_inferred_task] = -1 * torch.sum(
                middle_softmaxed_inferred_task * torch.log(middle_softmaxed_inferred_task)
            )

            upper_softmaxed_inferred_task = F.softmax(
                upper_all_task_single_output_sample[no_of_inferred_task], dim=-1
            )
            upper_task_entropies[no_of_inferred_task] = -1 * torch.sum(
                upper_softmaxed_inferred_task * torch.log(upper_softmaxed_inferred_task)
            )

        lower_selected_task_id  = torch.argmin(lower_task_entropies)
        middle_selected_task_id = torch.argmin(middle_task_entropies)
        upper_selected_task_id  = torch.argmin(upper_task_entropies)

        selected_task_id = torch.stack([
            lower_selected_task_id,
            middle_selected_task_id,
            upper_selected_task_id
        ])

        selected_task_id = selected_task_id.mode(dim=0)
        selected_task_id = selected_task_id.values

        predicted_tasks.append(selected_task_id.item())
        target_output = middle_all_task_single_output_sample[selected_task_id.item()]
        output_relative_class = target_output.argmax().item()
        if dataset in ["PermutedMNIST", "SplitMNIST"]:
            mode = "permuted" if dataset == "PermutedMNIST" else "split"
            output_absolute_class = translate_output_MNIST_classes(
                [output_relative_class], selected_task_id.item(), mode=mode
            )
        else:
            raise ValueError("Wrong name of the dataset!")
        predicted_classes.append(output_absolute_class)
    predicted_tasks = torch.tensor(predicted_tasks, dtype=torch.int32)
    predicted_classes = torch.tensor(predicted_classes, dtype=torch.int32)
    return predicted_tasks, predicted_classes

def calculate_entropy_and_predict_classes_separately(experiment_models):
    """
    Select the target task automatically and calculate accuracy for
    consecutive samples

    Arguments:
    ----------
    *experiment_models*: A dictionary with the following keys:
       *hypernetwork*: an instance of HMLP class
       *target_network*: an instance of MLP or ResNet class
       *number_of_task*: a number of currently solved task
       *hyperparameters*: a dictionary with experiment's hyperparameters
       *dataset_CL_tasks*: list of objects containing consecutive tasks
       *perturbated_epsilon*: a integer which is a perturbated epsilon

    Returns Pandas Dataframe with results for the selected model.
    """
    hypernetwork = experiment_models["hypernetwork"]
    target_network = experiment_models["target_network"]
    hyperparameters = experiment_models["hyperparameters"]
    dataset_CL_tasks = experiment_models["list_of_CL_tasks"]
    current_task_id = experiment_models["number_of_task"]
    dataset_name = experiment_models["hyperparameters"]["dataset"]
    perturbated_eps = experiment_models["perturbated_epsilon"]
    saving_folder = experiment_models["saving_folder"]

    hypernetwork.eval()
    target_network.eval()

    results = []
    for task in range(current_task_id + 1):

        X_test, y_test, gt_tasks = extract_test_set_from_single_task(
            dataset_CL_tasks, task, dataset_name, hyperparameters["device"]
        )

        with torch.no_grad():
            lower_logits_outputs_for_different_tasks  = []
            middle_logits_outputs_for_different_tasks = []
            upper_logits_outputs_for_different_tasks  = []

            for inferenced_task in range(task + 1):
                lower_weights, middle_weights, upper_weights, _ = hypernetwork.forward(cond_id=inferenced_task, 
                                                                perturbated_eps=perturbated_eps,
                                                                return_extended_output=True)
                
                # Try to predict task for all samples from "task"
                logits = target_network.forward(
                    x=X_test,
                    lower_weights=lower_weights,
                    middle_weights=middle_weights,
                    upper_weights=upper_weights
                )

                lower_logits, middle_logits, upper_logits = parse_predictions(logits)


                lower_logits = (middle_logits + lower_logits)/2
                upper_logits = (upper_logits + middle_logits)/2

                lower_logits_outputs_for_different_tasks.append(lower_logits)
                middle_logits_outputs_for_different_tasks.append(middle_logits)
                upper_logits_outputs_for_different_tasks.append(upper_logits)

            all_inferenced_tasks_lower = torch.stack(
                lower_logits_outputs_for_different_tasks
            )

            all_inferenced_tasks_middle = torch.stack(
                middle_logits_outputs_for_different_tasks
            )

            all_inferenced_tasks_upper = torch.stack(
                upper_logits_outputs_for_different_tasks
            )

            # Sizes of consecutive dimensions represent:
            # number of tasks x number of samples x number of output heads
        
        (predicted_tasks, predicted_classes) = get_task_and_class_prediction_based_on_logits(
                                                        all_inferenced_tasks_lower,
                                                        all_inferenced_tasks_middle,
                                                        all_inferenced_tasks_upper,
                                                        hyperparameters["number_of_tasks"],
                                                        dataset_name,
                                                    )
        
        predicted_classes = predicted_classes.flatten().numpy()
        task_prediction_accuracy = (
            torch.sum(predicted_tasks == task).float()
            * 100.0
            / predicted_tasks.shape[0]
        ).item()
        print(f"task prediction accuracy: {task_prediction_accuracy}")
        sample_prediction_accuracy = (
            np.sum(predicted_classes == y_test) * 100.0 / y_test.shape[0]
        ).item()
        print(f"sample prediction accuracy: {sample_prediction_accuracy}")
        results.append(
            [task, task_prediction_accuracy, sample_prediction_accuracy]
        )
    results = pd.DataFrame(
        results, columns=["task", "task_prediction_acc", "class_prediction_acc"]
    )
    results.to_csv(
        f"{saving_folder}/entropy_statistics.csv", sep=";"
    )
    return results



def extract_test_set_from_single_task(
    dataset_CL_tasks, no_of_task, dataset, device
):
    """
    Extract test samples dedicated for a selected task
    and change relative output classes into absolute classes.

    Arguments:
    ----------
       *dataset_CL_tasks*: list of objects containing consecutive tasks
       *no_of_task*: (int) represents number of the currently analyzed task
       *dataset*: (str) defines name of the dataset used: 'PermutedMNIST',
                  'SplitMNIST' or 'CIFAR100_FeCAM_setup'
       *device*: (str) defines whether CPU or GPU will be used

    Returns:
    --------
       *X_test*: (torch.Tensor) represents input samples
       *gt_classes*: (Numpy array) represents absolute classes for *X_test*
       *gt_tasks*: (list) represents number of task for corresponding samples
    """
    tested_task = dataset_CL_tasks[no_of_task]
    input_data = tested_task.get_test_inputs()
    output_data = tested_task.get_test_outputs()
    X_test = tested_task.input_to_torch_tensor(
        input_data, device, mode="inference"
    )
    test_output = tested_task.output_to_torch_tensor(
        output_data, device, mode="inference"
    )
    gt_classes = test_output.max(dim=1)[1]
    if dataset in ["PermutedMNIST", "SplitMNIST"]:
        mode = "permuted" if dataset == "PermutedMNIST" else "split"
        gt_classes = translate_output_MNIST_classes(
            gt_classes, task=no_of_task, mode=mode
        )
    else:
        raise ValueError("Wrong name of the dataset!")
    gt_tasks = [no_of_task for _ in range(output_data.shape[0])]
    return X_test, gt_classes, gt_tasks


def translate_output_CIFAR_classes(labels, setup, task):
    """
    Translate labels of form {0, 1, ..., N-1} to the real labels
    of CIFAR100 dataset.

    Arguments:
    ----------
       *labels*: (Numpy array | list) contains labels of the form {0, 1, ..., N-1}
                 where N is the the number of classes in a single task
       *setup*: (int) defines how many tasks were created in this
                training session
       *task*: (int) number of the currently calculated task
    Returns:
    --------
       A numpy array of the same shape like *labels* but with proper
       class labels
    """
    assert setup in [5, 6, 11, 21]
    # 5 tasks: 20 classes in each task
    # 6 tasks: 50 initial classes + 5 incremental tasks per 10 classes
    # 11 tasks: 50 initial classes + 10 incremental tasks per 5 classes
    # 21 tasks: 40 initial classes + 20 incremental tasks per 3 classes
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
    if setup in [6, 11]:
        no_of_initial_cls = 50
    elif setup == 21:
        no_of_initial_cls = 40
    else:
        no_of_initial_cls = 20
    if task == 0:
        currently_used_classes = class_orders[:no_of_initial_cls]
    else:
        if setup == 6:
            no_of_incremental_cls = 10
        elif setup == 11:
            no_of_incremental_cls = 5
        elif setup == 21:
            no_of_incremental_cls = 3
        else:
            no_of_incremental_cls = 20
        currently_used_classes = class_orders[
            (no_of_initial_cls + no_of_incremental_cls * (task - 1)) : (
                no_of_initial_cls + no_of_incremental_cls * task
            )
        ]
    y_translated = np.array([currently_used_classes[i] for i in labels])
    return y_translated