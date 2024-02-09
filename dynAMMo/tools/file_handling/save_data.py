import pathlib
from typing import Union

import h5py
from deeptime.markov.msm import MarkovStateModelCollection

from dynamics_utils.file_handling.load_files import HDFLoader


def save_spectral_components(
    file_name: Union[str, pathlib.PosixPath], msm: Union[MarkovStateModelCollection]
):
    """
    Save the spectral components of a Markov state model
    Parameters
    ----------
    file_name: Union[str, pathlib.PosixPath]
        The name of the file
    msm: Union[MarkovStateModelCollection]
        The Markov model

    Returns
    -------
    None

    """
    # Check if file exists
    file_name = pathlib.Path(file_name)
    if not file_name.exists():
        file_name.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(file_name, "w") as f:
            f.close()

    hdf_loader = HDFLoader(file_name)
    msm_dict = {
        "reigvecs": msm.eigenvectors_right(),
        "leigvecs": msm.eigenvectors_left().T,
        "eigvals": msm.eigenvalues(),
        "stationary_distribution": msm.stationary_distribution,
        "T": msm.transition_matrix,
        "active_set": msm.count_model.states,
        "count_matrix": msm.count_model.count_matrix,
    }

    hdf_loader.write(msm_dict)
