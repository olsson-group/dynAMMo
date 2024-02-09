from typing import *
import pathlib
import pickle

import torch
import numpy as np
import h5py

from ..definitions import ROOTDIR


def load_msm(filename: Optional[str] = None, group: Union[int, str] = '0', observables: List[str] = []) -> Dict:
    """
    Reads HDF file containing MSM data and returns dictionary with all relevant attributes

    Parameters
    ----------
    filename:   Optional[str], default: None
    group:      Union[int, str], default: 0
    observables:List[str], default: []

    Returns
    -------
    hdf_dict:   Dict
                Dictionary containing all MSM-relevant attributes (T, C, reigvecs, eigvals, stationary_distribution,
                 etc.)


    """

    # load HDF file
    if filename is None:
        hf = h5py.File(ROOTDIR / filename, 'r')
    elif type(filename) == h5py._hl.files.File:
        hf = filename
    else:
        hf = h5py.File(ROOTDIR / filename, 'r')

    # fill dictionary
    hf_dict = dict()
    keys = ['T', 'count_matrix', 'leigvecs', 'reigvecs', 'eigvals', 'active_set']
    for key in keys:
        hf_dict[key] = torch.tensor(np.array(hf[f'{group}'].get(key)))

    try:
        hf_dict['obs'] = list(hf[f'{group}']['obs'].values())
        hf_dict['obs'] = torch.vstack([torch.tensor(np.array(o_calc)) for o_calc in hf_dict['obs']])  # n_obs, n_states
    except KeyError:
        pass

    # parse additional keys
    assert type(keys) == list

    for key in observables:
        try:
            hf_dict[key] = list(hf[f'{group}'][key].values())
            hf_dict[key] = torch.vstack(
                [torch.tensor(np.array(o_calc)) for o_calc in hf_dict[key]])  # n_obs, n_states
        except KeyError:
            pass

    hf_dict['stationary_distribution'] = hf_dict['leigvecs'][:, 0]

    # close HDF
    hf.close()

    return hf_dict


def load_dtraj(filename: Optional[str] = None) -> List[np.ndarray]:
    """
        Convenience function to load dtraj

        Parameters
        ----------
        filename:   Optional[str], default: None

        Returns
        -------
        dtraj:      List[np.ndarray]
                    List of dtrajs

        """
    return load_hdf(filename, dataset='dtraj')


def load_hdf(filename: Optional[str] = None, dataset: Optional[str] = None) -> List:
    """
        Reads HDF file and returns list of all its datasets --> works for flat data sets without groups

        Parameters
        ----------
        filename:   Optional[str], default: None
        dataset:    Optional[str], default: None

        Returns
        -------
        out:        List
                    list of datasets
        """

    # load HDF file
    if filename is None:
        hf = h5py.File(ROOTDIR / filename, 'r')
    elif type(filename) == h5py._hl.files.File:
        hf = filename
    else:
        hf = h5py.File(ROOTDIR / filename, 'r')

    if dataset is None:
        list_of_datasets = [np.array(hf.get(key)) for key in hf.keys()]
        hf.close()
    else:
        list_of_datasets = [np.array(hf[key]) for key in hf.keys() if dataset in key]

    return list_of_datasets


def load_pickle(fname: Union[pathlib.PosixPath, str]) -> Any:
    """
    Loads pkl object and returns content

    Parameters
    ----------
    fname:  Union[pathlib.PosixPath, str]
            Filename or path to file

    Returns
    -------
    obj:    Any
            Unpickled object

    """
    with open(fname, 'rb') as file:
        f = pickle.load(file)
        file.close()
        return f
