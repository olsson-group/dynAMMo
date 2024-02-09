from ..definitions import ROOTDIR
import MDAnalysis as mda


def load_jpcb_traj() -> mda.Universe():
    traj_file = ROOTDIR / "data/md/ubq/deshaw_jpcb_sim/file_superpose.xtc"
    topo_file = ROOTDIR / "data/md/ubq/deshaw_jpcb_sim/system-protein.pdb"
    u = mda.Universe(topo_file, traj_file, in_memory=True)
    return u


def load_pnas_traj() -> mda.Universe():
    traj_file = ROOTDIR / "data/md/ubq/deshaw_pnas_sim/file_superpose.xtc"
    topo_file = ROOTDIR / "data/md/ubq/deshaw_jpcb_sim/system-protein.pdb"
    u = mda.Universe(topo_file, traj_file, in_memory=True)
    return u