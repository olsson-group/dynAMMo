import argparse
import json
from functools import partial
import pathlib
from typing import Union

import torch
import matplotlib.pyplot as plt
import h5py

# from dynAMMo.tools.plot_utils import PlotDynAMMo
from dynAMMo.model.estimator import DynamicAugmentedMarkovModelEstimator
from dynamics_utils.msm import (
    calculate_acf_from_spectral_components as acf,
    calculate_stationary_observable,
)
from dynamics_utils.math import mean_center
from dynAMMo.tools.utils import loss_convergence, pad_and_stack_tensors
import dynamics_utils.nmr as nmr
from dynAMMo.tools.definitions import ROOTDIR
from dynAMMo.tools.file_handling.load_data import load_msm
from dynAMMo.model.estimator import DynamicAugmentedMarkovModel
from dynAMMo.base.experiments import DynamicExperiment, StaticExperiment


def parse_arguments(config_filename):
    """
    Parses command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=ROOTDIR / config_filename,
        help="path to setup file",
    )

    parser.add_argument("--n_run", type=int, default=0, help="number of run")

    parser.add_argument("--dev", type=str, help="developer mode identifier")

    parser.add_argument(
        "--verbose", "-v", default=True, action="store_true", help="verbosity"
    )
    return parser.parse_args()


def load_config(config_file_name):
    """
    Loads and returns the configuration from the specified JSON file.

    Parameters
    ----------
    config_file_name : str
        The path to the JSON configuration file.

    Returns
    -------
    dict
        A dictionary containing configuration data.
    """
    # Convert pathlib file name to string
    config_file_name = str(config_file_name)

    # Load config
    with open(config_file_name, "r") as config_file:
        config = json.load(config_file)

    # Add file name to dictionary
    config["config_file_name"] = config_file_name

    # Write updated config file
    with open(config_file_name, "w") as config_file:
        json.dump(config, config_file, indent=4)
        return config


def create_output_directory(args, config):
    """
    Creates the output directory based on the provided arguments and configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    config : dict
        Configuration dictionary containing directory settings.
    """
    if args.dev is not None:
        out_dir = ROOTDIR / f"plt/devel/" / args.config.stem / args.dev
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = ROOTDIR / config["out"] / "+".join(config["experiments"])
        out_dir.mkdir(parents=True, exist_ok=True)


def check_validity_of_experiment_names(config):
    """
    Validates the experiment names provided in the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing the names of the experiments.

    Raises
    ------
    ValueError
        If an experiment name is not valid.
    """
    valid_experiment_names = {"r1rho", "cpmg", "correlation"}
    invalid_names = set(config["experiments"]) - valid_experiment_names
    if invalid_names:
        raise ValueError(f"Invalid experiment names: {', '.join(invalid_names)}")


def initialize_msms(config):
    """
    Initializes and returns Markov State Models (MSMs) for both calculated and experimental data.

    This function performs the following steps:
    - Loads MSMs from the specified input directories.
    - Extracts transition matrices, count matrices, and active sets from the loaded MSMs.
    - Processes experimental MSM data.
    - Initializes a DynamicAugmentedMarkovModel with the extracted data.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing necessary parameters.

    Returns
    -------
    tuple
        A tuple containing the initialized MSM, experimental MSM, and active set.
    """
    # Load MSMs from calculated data
    calc_msms, T_calcs, C_calcs, active_set_calc = load_calculated_msms(config)

    # Load and process experimental MSM data
    msm_exp, observables_by_state_exp = load_and_process_experimental_msm(config)

    # Initialize DynamicAugmentedMarkovModel
    msm = initialize_dynamic_augmented_markov_model(
        T_calcs, C_calcs, observables_by_state_exp, config
    )

    return msm, msm_exp, active_set_calc


def load_calculated_msms(config):
    """
    Loads MSMs from calculated data and extracts relevant matrices and sets.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing input directories.

    Returns
    -------
    tuple
        A tuple containing loaded MSMs, transition matrices, count matrices, and active sets.
    """
    calc_msms = [
        load_msm(filename=ROOTDIR / inp_dir) for inp_dir in config["inp_calculated"]
    ]
    T_calcs = [msm["T"] for msm in calc_msms]
    C_calcs = [msm["count_matrix"] for msm in calc_msms]
    active_set_calc = torch.concat([msm["active_set"] for msm in calc_msms])

    return calc_msms, T_calcs, C_calcs, active_set_calc


def load_and_process_experimental_msm(config):
    """
    Loads the experimental MSM and processes its observables.

    Args:
        config (dict): Configuration dictionary containing experimental MSM parameters.

    Returns:
        tuple: A tuple containing the experimental MSM and processed observables by state.
    """
    with h5py.File(ROOTDIR / config["inp_experimental"], "r") as hf:
        msm_exp_dict = load_msm(filename=hf, group="0")

    msm_exp = DynamicAugmentedMarkovModel(
        transition_matrices=[msm_exp_dict["T"]],
        count_models=[msm_exp_dict["count_matrix"]],
        observables_by_state=None,
    )

    observables_by_state_exp = process_observables(msm_exp, config)
    msm_exp.observables_by_state = observables_by_state_exp.numpy()

    return msm_exp, observables_by_state_exp


def process_observables(msm_exp, config):
    """
    Processes observables from the MSM experimental data.

    Args:
        msm_exp (DynamicAugmentedMarkovModel): The experimental MSM.
        config (dict): Configuration dictionary containing observable processing parameters.

    Returns:
        torch.Tensor: Processed observables by state.
    """
    start, stop = 1, config["n_samples"] + 1
    if config["missing"]:
        # Handling missing states if specified in the configuration
        return mean_center(
            msm_exp.reigvecs[config["len_missing_subtraj"] :, start:stop].T
        )
    else:
        return mean_center(msm_exp.reigvecs[:, start:stop].T)


def initialize_dynamic_augmented_markov_model(
    T_calcs, C_calcs, observables_by_state_exp, config
):
    """
    Initializes a DynamicAugmentedMarkovModel with the given data.

    Args:
        T_calcs (list): List of transition matrices.
        C_calcs (list): List of count matrices.
        observables_by_state_exp (torch.Tensor): Observables by state from experimental data.
        config (dict): Configuration dictionary.

    Returns:
        DynamicAugmentedMarkovModel: The initialized model.
    """
    return DynamicAugmentedMarkovModel(
        transition_matrices=T_calcs,
        count_models=C_calcs,
        observables_by_state=[observables_by_state_exp],
        initialize_via_count_matrix=config.get("initialize_via_count_matrix", False),
    )


def set_up_experiments(config, msm, msm_exp):
    """
    Sets up and returns dynamic and static experiments based on MSM models.

    This function creates experiments for different types of observables, both dynamic and static,
    based on the configurations and MSM models provided.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing experiment parameters and settings.
    msm : DynamicAugmentedMarkovModel
        The MSM model for calculated data.
    msm_exp : DynamicAugmentedMarkovModel
        The MSM model for experimental data.

    Returns
    -------
    list
        A list of experiment objects created.
    torch.Tensor
        A tensor of dynamic observables from the Markov State Model.

    """
    indep_var_dict = setup_independent_variables(config)
    experiments = []

    for i, observables_by_state in enumerate(msm.observables_by_state):
        experiments += create_experiments_for_observable(
            i, observables_by_state, msm, msm_exp, config, indep_var_dict
        )

    dynamic_observables_msm = extract_dynamic_observables(experiments)
    return experiments, dynamic_observables_msm


def setup_independent_variables(config):
    """
    Sets up the independent variables for the experiments based on the provided configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing parameters for independent variables.

    Returns
    -------
    dict
        A dictionary of independent variables for different experiments.
    """
    return {
        "r1rho": torch.linspace(0, 1e9, 100),
        "cpmg": 1.0 / torch.linspace(1e3, 1e6, 50),
        "correlation": torch.arange(*config.get("indep_var", [0, 30, 1])),
    }


def create_experiments_for_observable(
    i, observables_by_state, msm, msm_exp, config, indep_var_dict
):
    """
    Creates dynamic and static experiments for a specific observable.

    Parameters
    ----------
    i : int
        Index of the observable.
    observables_by_state : torch.Tensor
        The observable state for which experiments are created.
    msm : DynamicAugmentedMarkovModel
        The MSM model for calculated data.
    msm_exp : DynamicAugmentedMarkovModel
        The MSM model for experimental data.
    config : dict
        Configuration dictionary.
    indep_var_dict : dict
        Dictionary of independent variables for the experiments.

    Returns
    -------
    list
        A list of experiment objects for the given observable.
    """
    experiments = []
    for experiment in config["experiments"]:
        indep_var = indep_var_dict[experiment]
        observable_function = get_observable_function(experiment, config)
        dynamic_experiment, static_experiment = create_single_experiment(
            i,
            observables_by_state,
            indep_var,
            msm,
            msm_exp,
            observable_function,
            config,
        )
        experiments.extend([dynamic_experiment, static_experiment])
    return experiments


def get_observable_function(experiment, config):
    """
    Retrieves the appropriate observable function based on the experiment type.

    Parameters
    ----------
    experiment : str
        The type of experiment.
    config : dict
        Configuration dictionary containing experiment settings.

    Returns
    -------
    function
        The function to calculate observables for the given experiment.
    """
    if experiment == "r1rho":
        return partial(nmr.r1rho_msm, nu0=config["nu0"])
    elif experiment == "cpmg":
        return partial(nmr.cpmg_msm, nu0=config["nu0"])
    else:
        return acf


def create_single_experiment(
    i, observables_by_state, indep_var, msm, msm_exp, observable_function, config
):
    """
    Creates a single dynamic and static experiment for a given observable.

    Parameters
    ----------
    i : int
        Index of the observable.
    observables_by_state : torch.Tensor
        The observable state for the experiment.
    indep_var : torch.Tensor
        The independent variable for the experiment.
    msm : DynamicAugmentedMarkovModel
        The MSM model for calculated data.
    msm_exp : DynamicAugmentedMarkovModel
        The MSM model for experimental data.
    observable_function : function
        The function to calculate observables for the experiment.
    config : dict
        Configuration dictionary.

    Returns
    -------
    DynamicExperiment
        The created dynamic experiment.
    StaticExperiment
        The created static experiment.
    """
    dynamic_experiment = DynamicExperiment(
        indep_var=indep_var,
        observables_by_state=observables_by_state,
        observable_function=observable_function,
        name=f"{i}",
    )
    # Set observable predictions for the dynamic experiment
    dynamic_experiment.observables_pred = dynamic_experiment(
        msm.leigvecs, msm.eigvals, lagtime=config["lag"]
    )
    dynamic_experiment.observables_exp = observable_function(
        indep_var,
        msm_exp.observables_by_state[i],
        msm_exp.leigvecs,
        msm_exp.eigvals,
        lagtime=config["lag"],
    )
    dynamic_experiment.observables_msm = dynamic_experiment(
        msm.leigvecs, msm.eigvals, lagtime=config["lag"]
    ).detach()

    # Create a static experiment
    static_experiment = StaticExperiment(
        observables_by_state=observables_by_state, name=f"{i}"
    )
    static_experiment.observables_pred = static_experiment(msm.stationary_distribution)
    static_experiment.observables_exp = calculate_stationary_observable(
        msm_exp.observables_by_state[i], msm_exp.stationary_distribution
    )

    return dynamic_experiment, static_experiment


def extract_dynamic_observables(experiments):
    """
    Extracts dynamic observables from a list of experiments.

    Parameters
    ----------
    experiments : list
        A list of experiment objects.

    Returns
    -------
    torch.Tensor
        A tensor of dynamic observables from the experiments.
    """
    return torch.vstack(
        [
            exp.observables_msm
            for exp in experiments
            if isinstance(exp, DynamicExperiment)
        ]
    )


def set_up_optimizer(config, model):
    """
    Sets up and returns an Adam optimizer with custom learning rates for different model parameters.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing optimizer settings.
    model : torch.nn.Module
        The model for which the optimizer is being set up.

    Returns
    -------
    torch.optim.Optimizer
        The configured Adam optimizer for the model.
    """
    # Separate parameters for custom learning rates
    custom_params = get_custom_params(model)
    default_params = get_default_params(model)

    optimizer = torch.optim.Adam(
        [{"params": default_params}, *custom_params],
        lr=config["lr"],
    )
    return optimizer


def get_custom_params(model):
    """
    Retrieves parameters with custom learning rates.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to retrieve parameters.

    Returns
    -------
    list
        A list of dictionaries containing parameters and their custom learning rates.
    """
    custom_lr_params = {
        "orthonormal_reigvecs": 1,
        "a": 0.0,
        "b": 0.0,
        "nu": 1e-2,
        "chi": 1e-2,
    }
    return [
        {"params": getattr(model, name), "lr": lr}
        for name, lr in custom_lr_params.items()
    ]


def get_default_params(model):
    """
    Retrieves default parameters that do not have custom learning rates.

    Parameters
    ----------
    model : torch.nn.Module
        The model from which to retrieve parameters.

    Returns
    -------
    list
        A list of default parameters without custom learning rates.
    """
    excluded_params = ["orthonormal_reigvecs", "a", "b", "nu", "chi"]
    return [
        model.get_parameter(name)
        for name, _ in model.named_parameters()
        if name not in excluded_params
    ]


def training_loop(model, optimizer, plots, config, results_filename, n=0):
    """
    Executes the training loop for the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for training.
    plots : PlotDynAMMo
        Object for handling plotting operations.
    config : dict
        Configuration dictionary containing training parameters.
    results_filename : str
        File name for saving results.
    n : int, default: 1
        Identifier for the current training run.

    """
    converged = False
    epochs = 0
    while not converged:
        epochs += 1
        loss = perform_training_iteration(model, optimizer, plots)

        if should_save_results(epochs):
            save_results(model, loss, results_filename, n, config, plots)
            converged = check_convergence(epochs, plots.loss, config)
            if converged:
                print("Converged after {} epochs".format(epochs))


def perform_training_iteration(model, optimizer, plots):
    """
    Performs a single training iteration.

    Parameters
    ----------
    model : torch.nn.Module
        The model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer used for training.
    plots : PlotDynAMMo
        Object for handling plotting operations.

    Returns
    -------
    float
        The loss value for the training iteration.
    """
    # Forward pass and loss computation
    model()
    nll, lloss, l_exp, l_other = model.single_losses()
    loss = model.loss(nll, lloss)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    model.project_eigfunc_grads()
    torch.nn.utils.clip_grad_norm_(
        [model.orthonormal_reigvecs], max_norm=1e1, norm_type=2
    )
    optimizer.step()

    # Update plots
    plots.loss = loss.item()
    return loss.item()


def should_save_results(epochs):
    """
    Determines whether the results should be saved at the current epoch.

    Parameters
    ----------
    epochs : int
        Current epoch number.

    Returns
    -------
    bool
        True if results should be saved, False otherwise.
    """
    return epochs % 500 == 0


def check_convergence(epochs, loss, config):
    """
    Checks if the training process has converged.

    Parameters
    ----------
    epochs : int
        Current epoch number.
    loss : list
        List of loss values.
    config : dict
        Configuration dictionary containing training parameters.

    Returns
    -------
    bool
        True if the training has converged, False otherwise.
    """
    return (
        loss_convergence(loss[-10:], eps=config.get("loss_eps", 1e-4))
        and epochs > 2000
        or epochs == 6000
    )


def save_results(model, loss, results_filename, n, config, plots):
    """
    Saves training results to a file.

    Parameters
    ----------
    model : torch.nn.Module
        current model
    loss : float
        The current loss value.
    results_filename : str
        File name for saving results.
    n : int
        Identifier for the current training run.
    config : dict
        Configuration dictionary containing training parameters.
    plots : PlotDynAMMo
        Object for handling plotting operations.
    """
    with torch.no_grad():
        print(loss)

        out_hf = h5py.File(results_filename, "a")
        try:
            results = out_hf.create_group(f"{n}")
        except ValueError:
            del out_hf[f"{n}"]
            results = out_hf.create_group(f"{n}")

        # Save various training results
        save_training_data(model, results, plots.loss)

        out_hf.attrs["config_filename"] = config["config_file_name"]
        out_hf.attrs.update(config)
        out_hf.close()

        plots.results = results_filename
        generate_figures(plots, config)


def save_training_data(model, results, loss):
    """
    Saves training data to the results file.

    Parameters
    ----------
    model : torch.nn.Module
        The model being trained.
    results : h5py.Group
        Group in the HDF5 file to save results.
    loss: torch.Tensor
        Loss tensor.
    """
    results.create_dataset("transition_matrix", data=model.transition_matrix)
    results.create_dataset("eigvals", data=model.eigvals)
    results.create_dataset("reigvecs", data=model.reigvecs)
    results.create_dataset(
        "stationary_distribution", data=model.stationary_distribution
    )
    results.create_dataset("active_set", data=model.msm.active_set)
    results.create_dataset("original_active_set", data=model.msm.original_active_set)
    results.create_dataset(
        "dynamic_observables_by_state", data=model.dynamic_observables_by_state
    )
    results.create_dataset("a", data=model.a)
    results.create_dataset("b", data=model.b)
    indep_var_ = [
        exp.indep_var for exp in model.experiments if isinstance(exp, DynamicExperiment)
    ]
    results.create_dataset("indep_var", data=pad_and_stack_tensors(indep_var_))
    results.create_dataset(
        "dynamic_observables_dynammo", data=model.dynamic_observables_pred
    )
    results.create_dataset(
        "dynamic_observables_experimental", data=model.dynamic_observables_exp
    )
    results.create_dataset(
        "dynamic_observables_msm", data=model.dynamic_observables_msm
    )
    results.create_dataset("loss", data=loss)


def generate_figures(plots, config):
    """
    Generates and saves figures based on the training results.

    Parameters
    ----------
    plots : PlotDynAMMo
        Object for handling plotting operations.
    config : dict
        Configuration dictionary containing figure generation parameters.
    """

    plots.make_toy_figure_supplement(ROOTDIR / config["out"] / "results.pdf")
    plt.close()
