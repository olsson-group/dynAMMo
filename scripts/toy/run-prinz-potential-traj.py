#%%
import argparse
import matplotlib.pyplot as plt
import numpy as np
from deeptime.data import prinz_potential
import h5py

from dynAMMo.tools.definitions import ROOTDIR, verboseprint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', '-o',
                        type=str,
                        default='data/toy/prinz-potential-connected/',
                        help='output directory')

    parser.add_argument('--length',
                        type=int,
                        default=100000,
                        help='number of integration steps')

    parser.add_argument('--n_steps',
                        type=int,
                        default=1000,
                        help='number of integration steps')

    parser.add_argument('--step_size',
                        type=float,
                        default=1e-6,
                        help='integrator step size')

    parser.add_argument("--verbose", "-v",
                        default=True,
                        action="store_true",
                        help="verbosity")

    args = parser.parse_args()

    output_directory = ROOTDIR / args.out_dir
    length = args.length
    n_steps = args.n_steps
    h = args.step_size
    verbose = args.verbose

    if not output_directory.exists():
        verboseprint(f'Creating directory {output_directory}.', v=verbose)
        output_directory.mkdir(exist_ok=True)

    verboseprint('Running trajectory...', v=verbose)

    system = prinz_potential(n_steps=n_steps, h=h)
    xs = np.linspace(-1, 1, 100)
    energy = system.potential(xs.reshape((-1, 1)))
    traj = system.trajectory([[0.]], length)
    dtraj = np.digitize(traj, bins=xs[:-1], right=False).squeeze()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(energy, xs)
    ax1.set_ylabel('x coordinate')
    ax1.set_xlabel('energy(x)')
    ax2.plot(xs[dtraj])
    ax2.set_xlabel('time (a.u.)')
    plt.savefig(output_directory / 'trajectory-plot.pdf')
    plt.close()

    verboseprint('Saving trajectory...', v=verbose)

    hf = h5py.File(output_directory / 'data.h5', 'w')
    hf.create_dataset('dtraj', data=dtraj)
    hf.close()

    verboseprint('Done!', v=verbose)