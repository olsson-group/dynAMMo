import argparse
import matplotlib.pyplot as plt
import numpy as np
from deeptime.data import triple_well_1d
import h5py

from dynAMMo.tools.definitions import ROOTDIR, verboseprint
from dynAMMo.tools.utils import filter_trajectory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', '-o',
                        type=str,
                        default='data/toy/triple-well-potential/',
                        help='output directory')

    parser.add_argument('--length',
                        type=int,
                        default=5000,
                        help='number of integration steps')

    parser.add_argument('--n_steps',
                        type=int,
                        default=20000,
                        help='number of integration steps')

    parser.add_argument('--h_val',
                        type=float,
                        default=1.8e-2,
                        help='number of integration steps')

    parser.add_argument('--threshold',
                        type=int,
                        default=1,
                        help='subtrajectory length threshold')

    parser.add_argument('--n_subsystems',
                        type=int,
                        default=1,
                        help='number of separate subtrajectories')

    parser.add_argument("--verbose", "-v",
                        default=True,
                        action="store_true",
                        help="verbosity")

    args = parser.parse_args()

    output_directory = ROOTDIR / args.out_dir
    length = args.length
    n_steps = args.n_steps
    h = args.h_val
    threshold = args.threshold
    n_subsystems = args.n_subsystems
    verbose = args.verbose

    if not output_directory.exists():
        verboseprint(f'Creating directory {output_directory}.', v=verbose)
        output_directory.mkdir(exist_ok=True)

    verboseprint('Running trajectory...', v=verbose)

    system = triple_well_1d(h=h, n_steps=n_steps)
    xs = np.linspace(0, 6., num=70)
    ys = system.potential(xs.reshape(-1, 1))

    traj = system.trajectory(x0=0.5, length=length, seed=53)
    dtraj = np.digitize(traj, bins=xs[:-1], right=False).squeeze()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,3), gridspec_kw={'width_ratios': [1, 4]})
    ax1.plot(ys.reshape(-1), xs)
    ax1.set_ylabel('RC')
    ax1.set_xlabel('energy(x)')
    ax2.plot(xs[dtraj])
    ax2.set_xlabel('time (a.u.)')
    plt.tight_layout()
    plt.savefig(output_directory / f'trajectory-plot.pdf')
    plt.close()

    verboseprint('Saving full trajectory...', v=verbose)
    hf = h5py.File(output_directory / 'data.h5', 'w')
    hf.create_dataset('dtraj', data=dtraj)
    hf.close()

    if n_subsystems == 1:
        pass
    else:
        if n_subsystems == 2:
            straj1_idx = np.argwhere(dtraj < 18).flatten()
            straj2_idx = np.argwhere(dtraj >= 18).flatten()
            subtrajs = [straj1_idx, straj2_idx]
        elif n_subsystems == 3:
            straj1_idx = np.argwhere(dtraj < 18).flatten()
            straj2_idx = np.argwhere((dtraj >= 18) & (dtraj < 52)).flatten()
            straj3_idx = np.argwhere(dtraj >= 52).flatten()
            subtrajs = [straj1_idx, straj2_idx, straj3_idx]
        else:
            raise ValueError('Number of subtrajectories is wrong.')

        for i, straj_idx in enumerate(subtrajs):
            # save subtrajectories
            verboseprint('Saving subtrajectories...', v=verbose)
            strajs, _ = filter_trajectory(dtraj, straj_idx, threshold)
            hf = h5py.File(output_directory / f'triple-well-potential-subtrajectories-{i + 1}.h5', 'w')

            for j, straj in enumerate(strajs):
                hf.create_dataset(f'dtraj-{j}', data=straj)

            hf.close()

    verboseprint('Done!', v=verbose)