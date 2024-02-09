import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
from dynamics_utils.potential import LangevinSampler, OneDimensionalDoubleWellPotential

from dynAMMo.tools.definitions import ROOTDIR, verboseprint


def sample_double_well_potential(a, b, kT, seed):
    potential = OneDimensionalDoubleWellPotential(a, b)
    sampler = LangevinSampler(potential, x0=x0, dt=dt, kT=kT, mGamma=mGamma, seed=seed)
    return sampler.run(n_steps).flatten()


def discretize(ftraj, xs):
    return np.digitize(ftraj, xs)


def save_dtraj(dtraj, fname):
    hf = h5py.File(output_directory / fname, "w")
    hf.create_dataset("dtraj", data=dtraj)
    hf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir", type=str, help="output directory", default="data/toy/double-well/"
    )
    parser.add_argument("--dt", type=float, help="time step", default=1.0)
    parser.add_argument("--kT", type=float, help="temperature", default=2.0)
    parser.add_argument("--mGamma", type=float, help="", default=1000)
    parser.add_argument(
        "--n_steps", type=int, help="number of time steps", default=50000
    )
    parser.add_argument(
        "--verbose", "-v", default=True, action="store_true", help="verbosity"
    )
    args = parser.parse_args()

    output_directory = ROOTDIR / args.out_dir
    dt = args.dt
    kT = args.kT
    mGamma = args.mGamma
    n_steps = args.n_steps
    verbose = args.verbose

    if not output_directory.exists():
        verboseprint(f"Creating directory {output_directory}.", v=verbose)
        output_directory.mkdir(exist_ok=True)

    # Initial position of the particle
    x0 = 0.5

    toy_experimental_ftraj = sample_double_well_potential(5.0, 1.0, 2.0, 10)
    toy_simulation_ftraj = sample_double_well_potential(7.0, 1.0, 4.0, 20)

    xs = np.linspace(-1.5, 1.5, 50)
    toy_experimental_dtraj = discretize(toy_experimental_ftraj, xs[:-1])
    toy_simulation_dtraj = discretize(toy_simulation_ftraj, xs[:-1])

    save_dtraj(toy_experimental_dtraj, "experimental-data.h5")
    save_dtraj(toy_simulation_dtraj, "simulation-data.h5")

    plt.hist(
        toy_experimental_dtraj, bins=50, alpha=0.8, density=True, label="experimental"
    )
    plt.hist(toy_simulation_dtraj, bins=50, alpha=0.8, density=True, label="simulation")
    plt.legend()
    plt.savefig(output_directory / "histogram.pdf")
    plt.close()
