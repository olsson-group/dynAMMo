import torch
from typing import *
import numpy as np

from dynamics_utils.msm import eigendecomposition

from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM


def run_msm(dtrajs, lag):
    count_model = TransitionCountEstimator(lag, "sliding").fit_fetch(dtrajs)
    msm = MaximumLikelihoodMSM().fit_fetch(count_model.submodel_largest())
    active_set = count_model.connected_sets()[
        0
    ]  # TODO: if first connected set is not the largest, then it's a
    # problem... Should I do something about it?
    msm_dict = dict()

    msm_dict["T"] = msm.transition_matrix
    msm_dict["count_matrix"] = count_model.count_matrix[:, active_set][active_set, :]
    msm_dict["leigvecs"] = msm.eigenvectors_left().T
    msm_dict["reigvecs"] = msm.eigenvectors_right()
    msm_dict["eigvals"] = msm.eigenvalues()[1:]
    msm_dict["active_set"] = active_set
    msm_dict["stationary_distribution"] = msm.eigenvectors_left()[:, 0]

    return msm_dict, msm


def pad_and_stack_tensors(
    list_of_tensors: List[torch.Tensor], axis: int = 1, value: Any = torch.nan
) -> torch.Tensor:
    """
    Pad tensors of different lengths and stack them along a given axis

    Parameters
    ----------
    list_of_tensors:    torch.Tensor
                        list of tensors to pack and stack

    axis:               int
                        Axis along which to stack and pad

    value:              Any, default: torch.nan
                        Value with which to pad tensors

    Returns
    -------
                        torch.Tensor
                        padded and stacked tensors

    """
    # expected shape per array: (n_obs, n_datapoints)

    # make sure list_of_tensors is at least 2d
    list_of_tensors = [torch.atleast_2d(arr) for arr in list_of_tensors]

    padded_tensors = []
    max_len = max([arr.shape[axis] for arr in list_of_tensors])

    for arr in list_of_tensors:
        len_arr = arr.shape[1]
        parr = torch.nn.functional.pad(arr, (0, max_len - len_arr), value=value)
        padded_tensors.append(parr)
    if axis == 0:
        return torch.hstack(padded_tensors)
    elif axis == 1:
        return torch.vstack(padded_tensors)
    else:
        raise ValueError(f"Stacking only works for axis 0 or 1, not axis = {axis}")


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, axis: int):
    """
    Calculates the mean along a certain axis of a masked tensor

    Parameters
    ----------
    tensor: torch.Tensor
            Tensor whose mean should be calculated

    mask:   torch.Tensor
            Boolean mask, same shape as tensor

    axis:   int, default: 1
            Axis along which to calculate mean

    Returns
    -------
    masked_mean:    torch.Tensor
                    mean of masked tensor
    """
    tensor[~mask] = 0.0
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=axis) / mask.sum(dim=axis)


def loss_convergence(
    loss: Union[torch.Tensor, list], eps: float = 1e-5, method: str = "median"
):
    """
    Calculates the delta delta loss as a function of epoch and returns whether the convergence criterion
    is met

    Parameters
    ----------
    loss:   torch.Tensor
            Loss as a function of epoch

    eps:    float, default: 1e-5
            Convergence criterion

    method: str, default: 'median'
            Method with which to compare the convergence criterion with

    Returns
    -------
    is_converged:   bool
                    True if converged, False if not
    """

    dl = abs_delta_loss(loss)
    ddl = abs_delta_loss(dl)

    if method == "median":
        return torch.median(ddl) < eps

    elif method == "mean":
        return torch.mean(ddl) < eps

    else:
        raise ValueError(
            f"Method {method} is not implemented. Choose between `median` or `mean`"
        )


def abs_delta_loss(loss: torch.Tensor):
    """
    Calculates the absolute difference between successive elements in loss tensor

    Parameters
    ----------
    loss:   torch.Tensor, (n,)

    Returns
    -------
    delta_loss: torch.Tensor, (n-1,)

    """
    return torch.tensor([abs(j - i) for i, j in zip(loss[:-1], loss[1:])])


def filter_trajectory(dtraj, idx, len_threshold):
    subtrajs = dict()
    count = 0
    subtrajs[count] = []
    for i, ii in enumerate(idx):
        if ii != idx[-1] and ii + 1 == idx[i + 1]:
            subtrajs[count].append(ii)
        else:
            count += 1
            subtrajs[count] = []

    # filter out short trajectories
    strajs = []
    fidx = []
    for straj in subtrajs.values():
        if len(straj) > len_threshold:
            strajs.append(dtraj[straj])
            fidx.append(straj)
    return strajs, np.concatenate(fidx)


def calculate_chi_squared(observed: torch.Tensor, expected: torch.Tensor, ax: int = 1):
    """
    Calculates the chi squared value for a given observed and expected value

    Parameters
    ----------
    observed:   torch.Tensor
                Observed values

    expected:   torch.Tensor
                Expected values

    ax:         int, default: 1

    Returns
    -------
    chi_squared:    float
                    chi squared value
    """
    observed = replace_nans(observed, value=1)
    expected = replace_nans(expected, value=1)
    return torch.sum((observed - expected) ** 2 / expected, axis=ax)


def replace_nans(tensor: torch.Tensor, value: object = 0.0) -> torch.Tensor:
    """
    Replaces nans in a tensor with value

    Parameters
    ----------
    tensor: torch.Tensor
    value:  float, default: 0.0

    Returns
    -------
    tensor: torch.Tensor
            Tensor with nans replaced by value
    """
    tensor[torch.isnan(tensor)] = value
    return tensor
