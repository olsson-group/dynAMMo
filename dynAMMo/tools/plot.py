# %%
from typing import *
from pathlib import Path
import json
from pathlib import Path
import pathlib
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from dynamics_utils.file_handling.load_files import HDFLoader
from matplotlib.ticker import LogFormatter, FormatStrFormatter

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import string
import networkx as nx

import training.utils

np.seterr(all="raise")
import torch
import h5py
from deeptime.util import confidence_interval
import deeptime as dt

from dynamics_utils.msm import (
    calculate_leigvecs,
    timescales_from_eigvals,
    calculate_free_energy_potential,
    calculate_metastable_decomposition,
    calculate_mfpt_rates,
)
import dynAMMo
from dynAMMo.base.experiments import DynamicExperiment
from dynAMMo.model.estimator import DynamicAugmentedMarkovModel

# from dynAMMo.tools.file_handling.load_toy_data import load_toy_dataset

from dynAMMo.tools.definitions import (
    ROOTDIR,
    quantitative_1,
    quantitative_2,
    dark_theme_colors,
    cm,
)

plt.rcParams.update({"font.family": "sans-serif"})
plt.rcParams.update({"font.sans-serif": "Helvetica"})
plt.rcParams.update({"font.size": 6})
plt.rcParams["axes.labelpad"] = "1"
plt.rcParams["xtick.major.pad"] = "1"
plt.rcParams["ytick.major.pad"] = "1"


@dataclass
class ScenarioContainer:
    config: Union[None, dict] = None
    lag: Union[None, int] = None
    dt_traj: Union[None, float] = None
    n_eigvals: Union[None, int] = None
    n_metastable_states: Union[None, int] = (None,)
    disconnected: bool = False
    transition_matrix: Union[None, torch.Tensor] = None
    pcca: Union[None, dt.markov.pcca] = None
    metastable_assignments: Union[None, torch.Tensor] = None
    eigvals_dynammo: Union[None, torch.Tensor] = None
    timescales_dynammo: Union[None, torch.Tensor] = None
    timescales_dynammo_lower: Union[None, torch.Tensor] = None
    timescales_dynammo_upper: Union[None, torch.Tensor] = None
    stationary_distribution_dynammo: Union[None, torch.Tensor] = None
    stationary_distribution_dynammo_lower: Union[None, torch.Tensor] = None
    stationary_distribution_dynammo_upper: Union[None, torch.Tensor] = None
    reigvecs_dynammo: Union[None, torch.Tensor] = None
    reigvecs_dynammo_lower: Union[None, torch.Tensor] = None
    reigvecs_dynammo_upper: Union[None, torch.Tensor] = None
    leigvecs_dynammo: Union[None, torch.Tensor] = None

    eigvals_gt: Union[None, torch.Tensor] = None
    timescales_gt: Union[None, torch.Tensor] = None
    stationary_distribution_gt: Union[None, torch.Tensor] = None
    reigvecs_gt: Union[None, torch.Tensor] = None

    eigvals_msm: Union[None, torch.Tensor] = None
    timescales_msm: Union[None, torch.Tensor] = None
    stationary_distribution_msm: Union[None, torch.Tensor] = None
    reigvecs_msm: Union[None, torch.Tensor] = None

    active_set_dynammo: Union[None, torch.Tensor] = None
    original_active_set_dynammo: Union[None, torch.Tensor] = None
    active_set_gt: Union[None, torch.Tensor] = None

    dynamic_observables_dynammo: Union[None, torch.Tensor] = None
    dynamic_observables_gt: Union[None, torch.Tensor] = None
    dynamic_observables_msm: Union[None, torch.Tensor] = None
    a: Union[None, torch.Tensor] = None
    b: Union[None, torch.Tensor] = None
    chi_squared: Union[None, torch.Tensor] = None
    indep_var: Union[None, torch.Tensor] = None
    dynamic_observables_by_state: Union[None, torch.Tensor] = None

    msm_available: bool = False
    experiments_available: bool = False
    missing: bool = False
    loss: Union[None, torch.Tensor] = None


class PlotDynAMMo:
    """
    Class for plotting all relevant results
    """

    def __init__(
        self,
        results: Union[List[h5py.File], h5py.File, str, List[str]] = None,
        plot_msm: bool = True,
        plot_experiments: bool = True,
        figsize: Tuple = (6, 5),
        dark_theme: bool = False,
        **kwargs,
    ):
        self.plot_msm: bool = plot_msm
        self.plot_experiments: bool = plot_experiments
        self.figsize: Tuple = figsize
        self._loss: List = []
        self.kwargs = kwargs
        self.results: Union[List[ScenarioContainer], None] = results
        if dark_theme:
            plt.style.use("dark_background")
            mpl.rcParams["axes.prop_cycle"] = dark_theme_colors
        else:
            mpl.rcParams["axes.prop_cycle"] = quantitative_1

        self.labels = ["dynAMMo", "Ground truth", "MSM"]
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def make_toy_figure_supplement(self, fname: str = None):
        """
        Convenience function that plots all results in one figure

        Parameters
        ----------
        fname:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        # make layout of figure according to n_eigvals
        results = self._results[0]
        fig, axes = self._layout_subplot_grids_toy_systems(results)
        self.plot_eigfuncs(
            ax=[axes["eigfuncs-label"]]
            + [axes[f"eigfuncs-{i}"] for i in range(results.n_eigvals)],
            label_axes=True,
        )
        self.plot_free_energy_potential(ax=axes["stationary_distribution"])
        self.plot_timescales(ax=axes["timescales"])
        axes["timescales"].set_title("Implied timescales")
        axes["timescales"].set_xticks(
            range(2, results.n_eigvals + 2),
            labels=[str(process + 2) for process in range(results.n_eigvals)],
        )
        axes["timescales"].set_xlabel("Process")
        axes["timescales"].set_ylabel("Timescale [a.u.]")

        self.plot_loss(ax=axes["loss"], loss=results.loss, axis_labels=True)
        self.plot_correlation_function(
            ax=[axes["observables"], axes["observables-resiudals"]]
        )
        axes["observables"].set_title("Dynamic observables")

        axes["legend"].legend(
            handles=[
                mpatches.Patch(color=self.colors[i], label=label)
                for i, label in enumerate(["dynAMMo", "ground truth", "MSM"])
            ],
            loc=8,
            ncol=3,
        )

        fig.align_ylabels()
        fig.align_xlabels()

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return fig, axes

    def make_bpti_figure_supplement(self, fname: str = None):
        """
        Convenience function that plots all results in one figure

        Parameters
        ----------
        fname:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        # make layout of figure according to n_eigvals
        results = self._results[0]
        n_eigfuncs = 3
        fig, axes = self._layout_subplot_grids_bpti_supplement(n_eigfuncs)

        self.plot_eigfuncs2D(
            ax=[axes["eigfuncs-label"]]
            + [axes[f"eigfuncs-{i}"] for i in range(n_eigfuncs)],
            label_axes=True,
            n_eigfuncs=n_eigfuncs,
        )
        [axes[f"eigfuncs-{i}"].set_rasterized(True) for i in range(n_eigfuncs)]

        self.plot_free_energy_potential2D(
            ax=axes["stationary_distribution"], axis_labels=True, cbar=False
        )
        axes["stationary_distribution"].set_title("Free energy surface")

        self._results[0].timescales_dynammo *= 1e3
        self._results[0].timescales_dynammo_lower *= 1e3
        self._results[0].timescales_dynammo_upper *= 1e3

        self.plot_timescales(ax=axes["timescales"])
        axes["timescales"].set_yscale("log")
        axes["timescales"].set_title("Implied timescales")
        axes["timescales"].set_xticks(
            range(2, results.n_eigvals + 2),
            labels=[str(process + 2) for process in range(results.n_eigvals)],
        )
        axes["timescales"].set_xlabel("Process")
        axes["timescales"].set_ylabel("Timescale [ms]")

        self.plot_loss(ax=axes["loss"], loss=results.loss, axis_labels=True)

        axes["legend"].legend(
            handles=[
                mpatches.Patch(color=self.colors[i], label=label)
                for i, label in enumerate(["dynAMMo"])
            ],
            loc=8,
            ncol=3,
        )

        fig.align_ylabels()
        fig.align_xlabels()

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return fig, axes

    def make_toy_figure_paper(self, fname: str = None):
        """
        Convenience function that plots all toy systems in one figure (main document)

        Parameters
        ----------
        fname:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        # make layout of figure according to n_eigvals
        fig, axes = self._layout_subplot_grids_toy_figure_paper()
        assert len(self._results) == 6

        systems = ["ppc", "ppd", "ppu", "twc", "twd", "twu"]
        label_fontsize = 6

        # *** ~~~ *** FREE ENERGY *** ~~~ ***
        # _ = [setattr(self._results[i], 'msm_available', False) for i in range(len(self._results))]

        self.plot_free_energy_potential(
            ax=axes[f"{systems[0]}-fep"], results=self._results[0]
        )
        axes[f"{systems[0]}-fep"].set_xlabel(
            "Reaction coordinate", fontdict={"fontsize": label_fontsize}
        )
        axes[f"{systems[0]}-fep"].set_ylabel(r"Free energy [$k_\mathrm{B}T$]")

        for i in range(6):
            if i > 0:
                self.plot_free_energy_potential(
                    ax=axes[f"{systems[i]}-fep"], results=self._results[i]
                )
            axes[f"{systems[i]}-fep"].set_xlabel(
                "Reaction coordinate", fontdict={"fontsize": label_fontsize}
            )
            axes[f"{systems[i]}-fep"].tick_params(
                axis="both", which="major", labelsize=label_fontsize
            )
            axes[f"{systems[i]}-fep"].set_ylim()
            if i < 3:
                axes[f"{systems[i]}-fep"].sharey(axes[f"{systems[0]}-fep"])
                if i != 0:
                    plt.setp(axes[f"{systems[i]}-fep"].get_yticklabels(), visible=False)
            else:
                if i != 3:
                    plt.setp(axes[f"{systems[i]}-fep"].get_yticklabels(), visible=False)
                axes[f"{systems[i]}-fep"].sharey(axes[f"{systems[3]}-fep"])

        axes["ppd-fep"].axvline(x=50, ls=":", color=self.colors[-1])
        axes["twd-fep"].axvline(x=20, ls=":", color=self.colors[-1])

        # *** ~~~ *** TIMESCALES *** ~~~ ***
        _ = [
            setattr(self._results[i], "msm_available", True)
            for i in range(len(self._results))
        ]

        axes[f"{systems[0]}-ts"].set_ylabel("Timescales [steps]")

        for i in range(6):
            self.plot_timescales(ax=axes[f"{systems[i]}-ts"], results=self._results[i])
            axes[f"{systems[i]}-ts"].set_xlabel(
                "Process", fontdict={"fontsize": label_fontsize}
            )
            axes[f"{systems[i]}-ts"].tick_params(
                axis="both", which="major", labelsize=label_fontsize
            )
            axes[f"{systems[i]}-ts"].set_xticks(
                range(2, self._results[i].n_eigvals + 2),
                labels=[
                    str(process + 2) for process in range(self._results[i].n_eigvals)
                ],
            )
            if i not in [0, 3]:
                plt.setp(axes[f"{systems[i]}-ts"].get_yticklabels(), visible=False)

            axes[f"{systems[i]}-ts"].set_yscale("log")
            axes[f"{systems[i]}-ts"].set_ylim(1e0, 1e4)
            # axes[f'{systems[i]}-ts'].yaxis.set_minor_formatter(NullFormatter())
            axes[f"{systems[i]}-ts"].yaxis.set_minor_formatter(LogFormatter(base=10))
            # yticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            # yticks += [j * 10 ** i for i in range(1, 5) for j in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
            # axes[f'{systems[i]}-ts'].set_yticks([1e0, 1e1,1e2, 1e3, 1e4], minor=True)
            # axes[f'{systems[i]}-ts'].yaxis.set_minor_formatter(LogFormatter(base=10))

        axes["ppd-ts"].axvline(x=2.5, ls=":", color=self.colors[-1])
        axes["twd-ts"].axvline(x=2.5, ls=":", color=self.colors[-1])

        # *** ~~~ *** OBSERVABLES ~~~ *** ~~~
        _ = [
            setattr(self._results[i], "msm_available", False)
            for i in range(len(self._results))
            if i in [1, 4]
        ]

        self.plot_correlation_function(
            ax=[axes[f"{systems[0]}-obs"], axes[f"{systems[0]}-obs-res"]],
            results=self._results[0],
            n=0,
        )
        axes[f"{systems[0]}-obs-res"].set_xlabel(
            r"$\tau$", fontdict={"fontsize": label_fontsize}
        )
        axes[f"{systems[0]}-obs"].set_ylabel(
            "Observables"
        )  # , horizontalalignment='left', y=1.0)
        axes[f"{systems[0]}-obs-res"].set_ylabel("Residuals")

        for i in range(6):
            axes[f"{systems[i]}-obs"].set_ylim(-0.1, 1.1)
            if i > 0:
                self.plot_correlation_function(
                    ax=[axes[f"{systems[i]}-obs"], axes[f"{systems[i]}-obs-res"]],
                    results=self._results[i],
                    n=0,
                )
            axes[f"{systems[i]}-obs-res"].set_xlabel(
                r"$\tau$", fontdict={"fontsize": label_fontsize}
            )
            axes[f"{systems[i]}-obs"].tick_params(
                axis="both", which="major", labelsize=label_fontsize
            )
            axes[f"{systems[i]}-obs-res"].tick_params(
                axis="both", which="major", labelsize=label_fontsize
            )
            axes[f"{systems[i]}-obs-res"].ticklabel_format(
                axis="y", style="sci", scilimits=(0, 0), useMathText=True
            )
            axes[f"{systems[i]}-obs-res"].yaxis.get_offset_text().set_fontsize(
                label_fontsize
            )
            if i < 3:
                if i != 0:
                    plt.setp(axes[f"{systems[i]}-obs"].get_yticklabels(), visible=False)
                    plt.setp(
                        axes[f"{systems[i]}-obs-res"].get_yticklabels(), visible=False
                    )
                    axes[f"{systems[i]}-obs"].sharey(axes[f"{systems[0]}-obs"])
                    axes[f"{systems[i]}-obs-res"].sharey(axes[f"{systems[0]}-obs-res"])
            else:
                if i != 3:
                    plt.setp(axes[f"{systems[i]}-obs"].get_yticklabels(), visible=False)
                    plt.setp(
                        axes[f"{systems[i]}-obs-res"].get_yticklabels(), visible=False
                    )
                    axes[f"{systems[i]}-obs"].sharey(axes[f"{systems[3]}-obs"])
                    axes[f"{systems[i]}-obs-res"].sharey(axes[f"{systems[3]}-obs-res"])

        for i in range(6):
            axes[f"{systems[i]}-obs"].text(
                31.5,
                1.2,
                r"$\chi^2:$ " + f"{round(figure._results[i].chi_squared[0].item(), 3)}",
                fontsize=label_fontsize,
                ha="right",
            )

        fig.align_ylabels()
        fig.align_xlabels()

        # ugly way of changing xaxis of fep from state indices to RC
        for i in range(6):
            if i <= 2:
                axes[f"{systems[i]}-fep"].set_xticks([1, 100], ["-1", "1"])
            else:
                axes[f"{systems[i]}-fep"].set_xticks([0, 50], ["0", "5"])

        i = 0
        for j in range(6):
            for row in ["fep", "ts", "obs"]:
                i += 1
                label = string.ascii_uppercase[i - 1]
                t = axes[f"{systems[j]}-{row}"].text(
                    0.11,
                    0.11,
                    label,
                    backgroundcolor="lightgrey",
                    transform=axes[f"{systems[j]}-{row}"].transAxes,
                    fontsize=7,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                    zorder=-1,
                )

                if row == "obs":
                    t.set_position((0.105, 1.2))

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return fig, axes

    def make_bpti_figure_observables(self, fname: str = None):
        """
        Convenience function that plots BPTI results in one figure

        Parameters
        ----------
        fname:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        # make layout of figure according to n_eigvals
        fig, axes = self._layout_subplot_grids_bpti()

        selected_residue_idx = [6, 7, 25, 26, 31, 46, 62, 63, 64]
        selected_residues = [
            r"Lys 15$^\dag$",
            r"Ala 16$^\dag$",
            r"Arg 39$^\dag$",
            r"Ala 40$^\dag$",
            r"Lys 46$^\dag$",
            r"Arg 17$^\ddag$",
            r"Gly 36$^\ddag$",
            r"Arg 39$^\ddag$",
            r"Ala 40$^\ddag$",
        ]
        idx = 0
        for i in range(3):
            for j in range(3):
                row = i
                col = j
                ax0 = axes[f"observables-{row}{col}"]
                ax1 = axes[f"observables-residuals-{row}{col}"]
                self.plot_relaxation_dispersion(
                    ax=[ax0, ax1], n=selected_residue_idx[idx]
                )
                ax0.set_title(f"{selected_residues[idx]}", fontsize=8)
                ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                chi_squared = (
                    self._results[0].chi_squared[selected_residue_idx[idx]].item()
                )
                ax0.text(
                    0.98,
                    0.8,
                    r"$\chi^2:$ " + f"{round(chi_squared, 3)}",
                    transform=ax0.transAxes,
                    ha="right",
                )
                idx += 1

        for i in range(3):
            axes[f"observables-{i}0"].set_ylabel(r"$R_2$ [s$^{-1}$]")

            axes[f"observables-residuals-2{i}"].set_xlabel(
                r"$\tau_{\mathrm{cp}}^{-1}$ [ms$^{-1}$]"
            )

        fig.align_ylabels()
        fig.align_xlabels()

        fig.legend(
            handles=[
                mpatches.Patch(color=self.colors[i], label=label)
                for i, label in enumerate(["dynAMMo", "experimental"])
            ],
            loc=8,
            ncol=len(self.labels),
        )

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return fig, axes

    def plot_parameter_estimates(
        self,
        parameter_name: str,
        ax: plt.axis = None,
        results: Union[Path, str, ScenarioContainer] = None,
        selected_residue_idx: list = None,
        selected_residues: list = None,
    ):
        """
        Plot parameter estimates for a given parameter
        Parameters
        ----------
        ax:             plt.axis
                        axis object to plot on
        parameter_name: str
                        Name of parameter to plot
        selected_residue_idx:   list
                                List of residue indices to plot
        selected_residues:      list
                                List of residue names to plot

        Returns
        -------
        ax:             plt.Axis
                        Axis object with plot

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        arr = getattr(results, parameter_name)
        if arr.ndim != 1:
            raise ValueError(f"Parameter `{parameter_name}` must be 1D")

        if selected_residue_idx is None:
            selected_residue_idx = torch.arange(0, len(arr))

        if selected_residues is None:
            selected_residues = [f"{i}" for i in selected_residue_idx]

        ax.bar(selected_residues, arr[selected_residue_idx], color=self.colors[0])

        return ax

    def make_parameter_estimates_figure_supplement(self, output_dir: str = None):
        """
        Make figure with parameter estimates for all parameters
        Parameters
        ----------
        output_dir:     str
                        Output directory for figure

        Returns
        -------
        ax:             plt.axis

        """
        # make layout of figure according to n_eigvals
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=self.figsize)

        selected_residue_idx = [6, 7, 25, 26, 31, 46, 62, 63, 64]
        selected_residues = [
            r"Lys 15$^\dag$",
            r"Ala 16$^\dag$",
            r"Arg 39$^\dag$",
            r"Ala 40$^\dag$",
            r"Lys 46$^\dag$",
            r"Arg 17$^\ddag$",
            r"Gly 36$^\ddag$",
            r"Arg 39$^\ddag$",
            r"Ala 40$^\ddag$",
        ]

        for i, param in enumerate(["a", "b"]):
            self.plot_parameter_estimates(
                parameter_name=param,
                ax=axes[i],
                selected_residue_idx=selected_residue_idx,
                selected_residues=selected_residues,
            )

        axes[0].set_yscale("log")

        axes[0].set_ylabel(r"$m$ [a. u.]")
        axes[1].set_ylabel(r"$R_{2, \mathrm{ intrinsic}}$ $[s^{-1}]$")

        fig.tight_layout()

        if output_dir is not None:
            plt.savefig(Path(output_dir) / f"parameter-estimates.pdf")

        return fig, axes

    def make_bpti_figure_observables_supplement(self, output_dir: str):
        """
        Convenience function that plots BPTI results in one figure

        Parameters
        ----------
        output_dir:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        if len(self._results) != 2:
            raise ValueError("Please provide two results files")

        results_connected = self._results[1]
        results_disconnected = self._results[0]

        fig_500, axes_500 = plt.subplots(10, 4, figsize=(8.5, 11))
        fig_600, axes_600 = plt.subplots(10, 4, figsize=(8.5, 11))

        if self.kwargs["observable_names"] is None:
            raise ValueError("Please provide observable names as keyword argument")

        observable_names_path = self.kwargs["observable_names"]

        with open(observable_names_path, "r") as f:
            observable_names = [line.strip() for line in f]

        dataset_500_idx = (0, 38)
        dataset_600_idx = (0, 38)

        fig_list = [fig_500, fig_600]
        axes_list = [axes_500, axes_600]
        filenames = ["bpti-observables-600", "bpti-observables-500"]
        colors = [self.colors[0], self.colors[-1]]

        res_idx = 0
        ax_idx = 0
        for dataset_idx, (lower_idx, upper_idx) in enumerate(
            [dataset_500_idx, dataset_600_idx]
        ):
            fig = fig_list[dataset_idx]
            for i in range(lower_idx, upper_idx):
                ax = axes_list[dataset_idx].flatten()[i]
                tau_cp = results_connected.indep_var[i]
                obs_connected = results_connected.dynamic_observables_dynammo[res_idx]
                dynamic_observables_gt = results_connected.dynamic_observables_gt[
                    res_idx
                ]
                obs_disconnected = results_disconnected.dynamic_observables_dynammo[
                    res_idx
                ]
                ax.plot(tau_cp, obs_connected)
                ax.plot(tau_cp, dynamic_observables_gt, ".")
                ax.plot(tau_cp, obs_disconnected, "--", c=self.colors[-1])
                ax.set_title(f"{observable_names[i]}")

                if torch.std(dynamic_observables_gt).item() < 1:
                    ul = max(dynamic_observables_gt) + 2
                else:
                    ul = (
                        max(dynamic_observables_gt)
                        + torch.std(dynamic_observables_gt).item()
                    )
                ax.set_ylim(0.0, ul)
                ax.set_ylabel(r"$R_2$ [s$^{-1}$]")
                ax.set_xlabel(r"$\tau_{\mathrm{cp}}$ [ms$^{-1}$]")
                res_idx += 1

            fig.legend(
                handles=[
                    mpatches.Patch(color=colors[i], label=label)
                    for i, label in enumerate(["BPTI connected", "BPTI disconnected"])
                ],
                loc=8,
                ncol=len(self.labels),
            )
            fig.tight_layout()
            fig.savefig(
                Path(output_dir) / f"{filenames[ax_idx]}-raw.pdf", transparent=True
            )
            plt.show()
            ax_idx += 1

    def make_bpti_figure_network(self, fname: str = None):
        """
        Convenience function that plots BPTI network in one figure

        Parameters
        ----------
        fname:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        # make layout of figure according to n_eigvals
        fig, axes = self._layout_subplot_grids_bpti_network()

        pos = {
            0: np.array([8, 1]),
            1: np.array([9, 6]),
            2: np.array([3.5, 0]),
            3: np.array([0, 6]),
        }

        axes["network"].set_prop_cycle(quantitative_2)
        self.plot_kinetic_network(
            ax=axes["network"],
            node_positions=pos,
            results=self._results[0],
            axis_labels=False,
        )

        self._results[0].timescales_dynammo *= 1e3
        self._results[0].timescales_dynammo_lower *= 1e3
        self._results[0].timescales_dynammo_upper *= 1e3
        self._results[0].timescales_msm *= 1e3

        l = self.plot_timescales(
            ax=axes["timescales"], results=self._results[0], axis_labels=False
        )
        l.get_lines()[-1].set_color(self.colors[2])
        axes["timescales"].axhline(2.6, color=self.colors[1], linestyle="--")
        axes["timescales"].fill_between(
            np.arange(1, 11), 2.2, 3.22, color=self.colors[1], alpha=0.2
        )
        axes["timescales"].set_yscale("log")
        axes["timescales"].set_ylabel(r"$t_{\mathrm{ex}, i}$ [ms]")
        axes["timescales"].set_xticks([2, 4, 6, 8], [2, 4, 6, 8])
        axes["timescales"].set_xlabel(r"Process $i$")
        axes["timescales"].set_ylim(top=50)
        axes["timescales"].set_xlim(1.5, 9.5)
        axes["timescales"].legend(loc="upper right", fontsize=6)
        axes["timescales"].legend(
            handles=[
                mpatches.Patch(color=self.colors[i], label=label)
                for i, label in enumerate(["dynAMMo", "experimental", "MSM"])
            ]
        )

        axis_names = ["network", "timescales"]
        axes["network"].text(
            0.05,
            0.962,
            "B",
            backgroundcolor="lightgrey",
            transform=axes["network"].transAxes,
            fontsize=7,
            fontweight="bold",
            va="top",
            ha="right",
            zorder=2,
        )

        axes["timescales"].text(
            0.11,
            0.962,
            "C",
            backgroundcolor="lightgrey",
            transform=axes["timescales"].transAxes,
            fontsize=7,
            fontweight="bold",
            va="top",
            ha="right",
            zorder=2,
        )

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return fig, axes

    def make_bpti_figure_disconnected(self, fname: str = None):
        """
        Convenience function that plots BPTI network in one figure

        Parameters
        ----------
        fname:      str
                    filename for savefig

        Returns
        -------
        fig:        matplotlib.pyplot.Figure
                    Figure object
        ax:         matplotlib.pyplot.axes
                    Axes object

        """
        # make layout of figure according to n_eigvals
        fig, axes = self._layout_subplot_grids_bpti_disconnected()

        pos = {
            0: np.array([8, 1]),
            1: np.array([9, 6]),
            2: np.array([3.5, 0]),
            3: np.array([0, 6]),
        }

        axes["network"].set_prop_cycle(quantitative_2)
        self.plot_kinetic_network(
            ax=axes["network"],
            node_positions=pos,
            results=self._results[0],
            axis_labels=False,
            node_scaling_factor=5e3,
            n_metastable_states=4,
        )

        for i in range(2):
            self._results[i].timescales_dynammo *= 1e3
            self._results[i].timescales_dynammo_lower *= 1e3
            self._results[i].timescales_dynammo_upper *= 1e3
            self._results[i].timescales_msm *= 1e3

        l = self.plot_timescales(
            ax=axes["timescales"], results=self._results[1], axis_labels=False
        )
        l.get_lines()[-1].set_color(self.colors[2])

        self._results[0].msm_available = False
        self.plot_timescales(
            ax=axes["timescales"],
            results=self._results[0],
            axis_labels=False,
            color=self.colors[-1],
        )

        # axes['timescales'].legend(['BPTI MSM', 'BPTI biased', 'BPTI non-ergodic', ], loc='upper center', fontsize=6)

        axes["timescales"].set_ylabel(r"$t_{\mathrm{ex}, i}$ [ms]")
        axes["timescales"].set_xlabel(r"Process $i$")
        axes["timescales"].set_yscale("log")
        # axes['timescales'].set_ylim(top=600)

        selected_residue_idx = [26, 31, 63, 64]
        selected_residues = [
            r"Ala 40$^{\dag}$",
            r"Lys 46$^{\dag}$",
            r"Arg 39$^{\ddag}$",
            r"Ala 40$^{\ddag}$",
        ]
        idx = 0
        for j in range(4):
            col = j
            ax0 = axes[f"observables-{col}"]
            ax1 = axes[f"observables-residuals-{col}"]
            self.plot_relaxation_dispersion(
                ax=[ax0, ax1], n=selected_residue_idx[idx], results=self._results[1]
            )
            tau_cp = self._results[0].indep_var[selected_residue_idx[idx]]
            obs = self._results[0].dynamic_observables_dynammo[
                selected_residue_idx[idx]
            ]
            ax0.plot(tau_cp, obs, "--", c=self.colors[-1])
            ax0.set_title(f"{selected_residues[idx]}", fontsize=8)
            ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax1.set_xlabel(r"$\tau_{\mathrm{cp}}^{-1}$ [ms$^{-1}$]")

            chi_squared = self._results[0].chi_squared[selected_residue_idx[idx]].item()
            ax0.text(
                0.98,
                0.8,
                r"$\chi^2:$ " + f"{round(chi_squared, 3)}",
                transform=ax0.transAxes,
                ha="right",
            )

            idx += 1

        axes["observables-0"].set_ylabel(r"$R_2$ [s$^{-1}$]")

        axes["network"].text(
            0.08,
            0.965,
            "A",
            backgroundcolor="lightgrey",
            transform=axes["network"].transAxes,
            fontsize=7,
            fontweight="bold",
            va="top",
            ha="right",
            zorder=2,
        )

        axes["timescales"].text(
            0.96,
            0.96,
            "B",
            backgroundcolor="lightgrey",
            transform=axes["timescales"].transAxes,
            fontsize=7,
            fontweight="bold",
            va="top",
            ha="right",
            zorder=2,
        )

        colors = [self.colors[0], self.colors[1], self.colors[2], self.colors[-1]]
        fig.legend(
            handles=[
                mpatches.Patch(color=colors[i], label=label)
                for i, label in enumerate(
                    ["dynAMMo", "experimental", "MSM", "dynAMMo (disconnected)"]
                )
            ],
            loc=8,
            ncol=len(self.labels),
        )

        if fname is not None:
            plt.savefig(fname, transparent=True)

        return fig, axes

    def plot_stationary_distribution(
        self, ax: plt.axis = None, results: Union[Path, str, ScenarioContainer] = None
    ):
        """
        Plots stationary distribution of the n slowest processes

        Parameters
        ----------
        ax:         matplotlib.pyplot.axis
                    axis
        results:    Union[Path, str, ScenarioContainer], default = None
                    if is Path or str: results file will be used, else: ScenarioContainer will be used

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        ax.plot(
            results.active_set_dynammo,
            results.stationary_distribution_dynammo,
            label=self.labels[0],
        )
        ax.fill_between(
            results.active_set_dynammo,
            results.stationary_distribution_dynammo_lower,
            results.stationary_distribution_dynammo_upper,
            alpha=0.5,
        )

        if results.experiments_available:
            ax.plot(
                results.active_set_gt,
                results.stationary_distribution_gt,
                "--",
                label=self.labels[1],
            )

        if results.msm_available:
            ax.plot(
                results.active_set_dynammo,
                results.stationary_distribution_msm,
                alpha=0.5,
                label=self.labels[2],
            )

        ax.set_xlabel("States")
        ax.set_ylabel("Probability")
        ax.set_title("Stationary distribution")

        return ax

    def plot_free_energy_potential(
        self,
        ax: plt.axis = None,
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
    ):
        """
        Plots free energy potential from stationary distribution

        Parameters
        ----------
        ax:         matplotlib.pyplot.axis
                    axis
        results:    Union[Path, str, ScenarioContainer], default = None
                    if is Path or str: results file will be used, else: ScenarioContainer will be used
        axis_labels:bool
                    Plots x- and y-axis labels if true

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        ax.plot(
            results.active_set_dynammo,
            calculate_free_energy_potential(results.stationary_distribution_dynammo),
            label=self.labels[0],
        )
        ax.fill_between(
            results.active_set_dynammo,
            results.stationary_distribution_dynammo_lower,
            results.stationary_distribution_dynammo_upper,
            alpha=0.5,
        )

        if results.experiments_available:
            ax.plot(
                results.active_set_gt,
                calculate_free_energy_potential(results.stationary_distribution_gt),
                "--",
                label=self.labels[1],
            )

        if results.msm_available:
            ax.plot(
                results.active_set_dynammo,
                calculate_free_energy_potential(results.stationary_distribution_msm),
                alpha=1.0,
                linewidth=3,
                label=self.labels[2],
                zorder=-2,
            )

        if axis_labels:
            ax.set_xlabel("Reaction coordinate")
            ax.set_ylabel(r"Free energy [$kT$]")

        return ax

    def plot_free_energy_potential2D(
        self,
        ax: plt.axis = None,
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
        cbar: bool = False,
    ):
        """
        Plots free energy potential from stationary distribution

        Parameters
        ----------
        ax:         matplotlib.pyplot.axis
                    axis
        results:    Union[Path, str, ScenarioContainer], default = None
                    if is Path or str: results file will be used, else: ScenarioContainer will be used
        axis_labels:bool
                    Plots x- and y-axis labels if true
        cbar:       bool
                    Plots colorbar if true

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        if ("tica" or "dtraj") not in self.kwargs.keys():
            raise ValueError(
                "`plot_eigfuncs_2D` requires `tica` and `dtraj` trajectory as keyword argument"
            )

        tica = self.kwargs["tica"][:, :2]
        dtraj = self.kwargs["dtraj"]

        tica = tica[np.in1d(dtraj, results.original_active_set_dynammo)]
        dtraj = dtraj[np.in1d(dtraj, results.original_active_set_dynammo)]
        state_map = {
            ele.item(): results.active_set_dynammo[i].item()
            for i, ele in enumerate(results.original_active_set_dynammo)
        }
        dtraj_mapped = np.vectorize(state_map.get)(dtraj)

        # TODO: check if this is really necessary
        # plot_free_energy(*tica.T, weights=results.stationary_distribution_dynammo[dtraj_mapped], ax=ax, cbar=cbar)

        if axis_labels:
            ax.set_xlabel("tIC1")
            ax.set_ylabel(r"tIC2")
        #     # cb = plt.colorbar(dynammo_plt, cax=cax, use_gridspec=True)
        #     # cb.ax.set_ylabel(r'Free energy [$kT$]')

        return ax

    def plot_eigfuncs(
        self,
        ax: plt.axis = None,
        results: Union[Path, str, ScenarioContainer] = None,
        label_axes: bool = False,
    ):
        """
        Plots eigenfunctions of the n slowest processes

        Parameters
        ----------
        ax:         matplotlib.pyplot.axis
                    axis
        results:    Union[Path, str, ScenarioContainer], default = None
                    if is Path or str: results file will be used, else: ScenarioContainer will be used

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            _, ax = plt.subplots(1, results.n_eigvals + 1)
            if type(ax) is not np.ndarray:
                ax = np.array(ax)

        for i in range(results.n_eigvals):
            r_calc = results.reigvecs_dynammo[:, i]
            r_calc_l = results.reigvecs_dynammo_lower[:, i]
            r_calc_u = results.reigvecs_dynammo_lower[:, i]

            idx = i + 1
            ax[idx].plot(results.active_set_dynammo, r_calc, label=self.labels[0])
            ax[idx].fill_between(
                results.active_set_dynammo,
                r_calc,
                r_calc_l,
                r_calc_u,
                alpha=0.5,
                label="0.95 CI",
            )

            if results.experiments_available:
                r_exp = results.reigvecs_gt[:, i]
                # change sign of eigfunc if necessary
                as_exp_idx = (
                    torch.isin(results.active_set_gt, results.active_set_dynammo)
                    .nonzero()
                    .flatten()
                )
                as_calc_idx = (
                    torch.isin(results.active_set_dynammo, results.active_set_gt)
                    .nonzero()
                    .flatten()
                )
                diff1 = torch.abs(r_calc[as_calc_idx] - r_exp[as_exp_idx]).max()
                diff2 = torch.abs(r_calc[as_calc_idx] + r_exp[as_exp_idx]).max()
                if diff1 > diff2:
                    r_exp = -r_exp
                ax[idx].plot(results.active_set_gt, r_exp)

            if label_axes:
                ax[idx].set_title(rf"$\Phi_{i + 2}$")
                # ax[idx].set_box_aspect(1)
                # asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[idx].get_ylim())[0]
                # ax[idx].set_aspect(asp)
                ax[idx].set_xlabel("States", labelpad=0)

        asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
        ax[1].set_aspect(asp)

        if label_axes:
            # label stuff
            ax[0].set_ylabel("Amplitude [a. u.]")
            ax[0].spines["top"].set_color("none")
            ax[0].spines["bottom"].set_color("none")
            ax[0].spines["left"].set_color("none")
            ax[0].spines["right"].set_color("none")
            ax[0].tick_params(
                labelcolor="w", top=False, bottom=False, left=False, right=False
            )
            ax[0].get_yaxis().set_ticks([])
            ax[0].get_xaxis().set_ticks([])
            # ax[0].set_title('Eigenfunctions')

        return ax

    def plot_eigfuncs2D(
        self,
        ax: plt.axis = None,
        results: Union[Path, str, ScenarioContainer] = None,
        label_axes: bool = False,
        n_eigfuncs: Union[None, int] = None,
    ):
        """
        Plots eigenfunctions of the n slowest processes

        Parameters
        ----------
        ax:         matplotlib.pyplot.axis
                    axis
        results:    Union[Path, str, ScenarioContainer], default = None
                    if is Path or str: results file will be used, else: ScenarioContainer will be used
        label_axes: bool, default = False
                    if True, axes will be labeled
        n_eigfuncs: Union[None, int], default = None
                    number of eigenfunctions to plot

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            _, ax = plt.subplots(1, results.n_eigvals + 1)
            if type(ax) is not np.ndarray:
                ax = np.array(ax)

        if n_eigfuncs is None:
            n_eigfuncs = results.n_eigvals

        if ("tica" or "dtraj") not in self.kwargs.keys():
            raise ValueError(
                "`plot_eigfuncs_2D` requires `tica` and `dtraj` trajectory as keyword argument"
            )

        tica = self.kwargs["tica"][:, :2]
        dtraj = self.kwargs["dtraj"]

        tica = tica[np.in1d(dtraj, results.original_active_set_dynammo)]
        dtraj = dtraj[np.in1d(dtraj, results.original_active_set_dynammo)]
        state_map = {
            ele.item(): results.active_set_dynammo[i].item()
            for i, ele in enumerate(results.original_active_set_dynammo)
        }
        dtraj_mapped = np.vectorize(state_map.get)(dtraj)

        for i in range(n_eigfuncs):
            r_calc = results.reigvecs_dynammo[:, i]
            r_calc_l = results.reigvecs_dynammo_lower[:, i]
            r_calc_u = results.reigvecs_dynammo_lower[:, i]

            idx = i + 1
            ax[idx].scatter(*tica.T, c=r_calc[dtraj_mapped], s=1, edgecolors=None)
            # ax[idx].fill_between(results.active_set_dynammo, r_calc, r_calc_l, r_calc_u, alpha=0.5,
            #                      label='0.95 CI')

            if results.experiments_available:
                ...
            if label_axes:
                ax[idx].set_title(rf"$\Phi_{i + 2}$")
                # ax[idx].set_box_aspect(1)
                # asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[idx].get_ylim())[0]
                # ax[idx].set_aspect(asp)
                ax[idx].set_xlabel("tIC1 [a.u.]", labelpad=0)

        # asp = np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0]
        # ax[1].set_aspect(asp)
        #
        if label_axes:
            # label stuff
            ax[0].set_ylabel("tIC2 [a.u.]")
            ax[0].spines["top"].set_color("none")
            ax[0].spines["bottom"].set_color("none")
            ax[0].spines["left"].set_color("none")
            ax[0].spines["right"].set_color("none")
            ax[0].tick_params(
                labelcolor="w", top=False, bottom=False, left=False, right=False
            )
            ax[0].get_yaxis().set_ticks([])
            ax[0].get_xaxis().set_ticks([])
            # ax[0].set_title('Eigenfunctions')

        return ax

    def plot_timescales(
        self,
        ax: plt.axis = None,
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
        color: str = None,
    ):
        """
        Plots timescales of the n slowest processes

        Parameters
        ----------
        ax:         matplotlib.pyplot.axis
                    axis
        results:    Union[Path, str, ScenarioContainer], default = None
                    if is Path or str: results file will be used, else: ScenarioContainer will be used
        color:      str, default = None
                    sets color of dynAMMo data
        axis_labels:bool
                    Plots x- and y-axis labels if true

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        x_axis = range(2, results.n_eigvals + 2)

        ax.errorbar(
            x_axis,
            results.timescales_dynammo,
            yerr=(
                (results.timescales_dynammo - results.timescales_dynammo_lower).numpy(),
                (results.timescales_dynammo_upper - results.timescales_dynammo).numpy(),
            ),
            fmt=".",
            capsize=5.0,
            label=self.labels[0],
            color=color,
            zorder=10,
        )
        # ax.plot(x_axis, results.timescales_dynammo, 's', label=self.labels[0])
        all_timescales = []
        all_timescales.append(results.timescales_dynammo)

        if results.experiments_available:
            ax.plot(
                x_axis, results.timescales_gt, "d", label=self.labels[1], color=color
            )
            all_timescales.append(results.timescales_gt)

        if results.msm_available:
            if results.disconnected or results.missing:
                ax.plot(
                    x_axis[1:],
                    results.timescales_msm[1:],
                    "x",
                    label=self.labels[2],
                    color=color,
                )
                all_timescales.append(results.timescales_msm[1:])

            else:
                ax.plot(
                    x_axis,
                    results.timescales_msm,
                    "x",
                    label=self.labels[2],
                    color=color,
                )
                all_timescales.append(results.timescales_msm)

        max_timescale = np.max(all_timescales)

        ax.margins(y=0.1)
        if results.n_eigvals <= 2:
            ax.margins(x=0.3)
        else:
            ax.margins(x=0.1)

        ax.set_ylim(bottom=0, top=max_timescale + 1 / 4 * (max_timescale - 0))

        if axis_labels:
            ax.set_xticks(x_axis, labels=[str(process + 2) for process in x_axis])
            ax.set_ylabel("Timescale")
            ax.set_xlabel("Process")

            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position("both")
            ax.tick_params(
                axis="y",
                direction="in",
            )
            ax.yaxis.set_label_position("right")

            ax.set_title("Timescales")

        return ax

    def plot_loss(
        self, ax: plt.axis = None, loss: torch.Tensor = None, axis_labels: bool = False
    ):

        if ax is None:
            ax = plt.gca()

        if loss is None:
            ax.plot(self._loss)
        else:
            ax.plot(loss)

        if axis_labels:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss")

        return ax

    def plot_correlation_function(
        self,
        ax: List[plt.axis] = None,
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
        n: int = 0,
    ):
        """
        Plots observables and residuals

        Parameters
        ----------
        ax:         List[matplotlib.pyplot.axis]
                    List of axes
        results:    Union[Path, str, ScenarioContainer], default = None
                    If is Path or str: results file will be used, else: ScenarioContainer will be used
        axis_labels:bool
                    Plots x- and y-axis labels if true

        n:          int
                    Index of observable which should be plotted

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            _, ax = plt.subplots(2, 1)

        dynamic_observables_dynammo = results.dynamic_observables_dynammo[n]
        dynamic_observables_gt = results.dynamic_observables_gt[n]
        dynamic_observables_dynammo = (
            dynamic_observables_dynammo / dynamic_observables_gt[0]
        )
        dynamic_observables_gt = dynamic_observables_gt / dynamic_observables_gt[0]

        ax[0].plot(dynamic_observables_dynammo)
        ax[0].plot(dynamic_observables_gt, "--")
        if results.msm_available:
            if not results.disconnected:
                dynamic_observables_msm = results.dynamic_observables_msm[n]
                dynamic_observables_msm = (
                    dynamic_observables_msm / dynamic_observables_gt[0]
                )
                ax[0].plot(dynamic_observables_msm, "--")

        ax[0].set_xticklabels([])

        residuals = dynamic_observables_dynammo - dynamic_observables_gt
        std = torch.std(residuals)
        ylim = max(torch.abs(residuals) + std)
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ax[1].bar(range(len(residuals)), residuals, color=c[3])
        ax[1].axhline(y=0.0, xmin=0, xmax=len(residuals), color=c[4], linestyle="--")
        ax[1].set_ylim(-ylim, ylim)

        # ax[0].get_shared_x_axes().join(ax[0], ax[1])

        if axis_labels:
            ax[1].set_xlabel(r"$\tau$ [steps]")
            for ax_i in ax:
                ax_i.yaxis.tick_right()
                ax_i.yaxis.set_ticks_position("both")
                ax_i.tick_params(
                    axis="y",
                    direction="in",
                )
                ax_i.yaxis.set_label_position("right")

        return ax

    def plot_relaxation_dispersion(
        self,
        ax: List[plt.axis] = None,
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
        n: int = 0,
    ):
        """
        Plots observables and residuals

        Parameters
        ----------
        ax:         List[matplotlib.pyplot.axis]
                    List of axes
        results:    Union[Path, str, ScenarioContainer], default = None
                    If is Path or str: results file will be used, else: ScenarioContainer will be used
        axis_labels:bool
                    Plots x- and y-axis labels if true

        n:          int
                    Index of observable which should be plotted

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        # if results.config['experiments'][0] == 'cpmg':
        #     pred = cpmg_msm(results.indep_var[n], results.dynamic_observables_by_state[n], results.eigvals_dynammo,
        #                     results.leigvecs_dynammo, results.config['lag'], results.config['dt_traj'])

        dynamic_observables_dynammo = results.dynamic_observables_dynammo[n]

        dynamic_observables_gt = results.dynamic_observables_gt[n]

        nu = results.indep_var[n]

        ax[0].plot(nu, dynamic_observables_dynammo)
        ax[0].plot(nu, dynamic_observables_gt, ".")
        ax[0].set_xticklabels([])

        ax[0].set_ylim(
            0.0,
            max(dynamic_observables_gt)
            + torch.std(self.results[0].dynamic_observables_gt).item(),
        )

        residuals = torch.nan_to_num(
            dynamic_observables_dynammo - dynamic_observables_gt
        )
        std = torch.std(residuals)
        ylim = max(torch.abs(residuals) + std)
        c = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bar_width = 2 * min(torch.abs(torch.diff(torch.nan_to_num(nu))))
        ax[1].bar(nu, residuals, color=c[3], width=bar_width)
        ax[1].axhline(y=0.0, xmin=0, xmax=len(residuals), color=c[4], linestyle="--")
        ax[1].set_ylim(-ylim, ylim)

        ax[0].get_shared_x_axes().join(ax[0], ax[1])

        if axis_labels:
            ax[1].set_xlabel(r"$\tau$ [steps]")
            for ax_i in ax:
                ax_i.yaxis.tick_right()
                ax_i.yaxis.set_ticks_position("both")
                ax_i.tick_params(
                    axis="y",
                    direction="in",
                )
                ax_i.yaxis.set_label_position("right")

        return ax

    def plot_kinetic_network(
        self,
        ax: List[plt.axis],
        node_positions: dict,
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
        n_metastable_states: int = 4,
        node_scaling_factor: Union[float, int] = 1e4,
    ):
        """
        Plots the kinetic network

        Parameters
        ----------
        ax:         List[matplotlib.pyplot.axis]
                    List of axes
        node_positions: dict
                    Dictionary with node positions
        results:    Union[Path, str, ScenarioContainer], default = None
                    If is Path or str: results file will be used, else: ScenarioContainer will be used
        axis_labels: bool
                    Plots x- and y-axis labels if true
        n_metastable_states: int
                    Index of observable which should be plotted
        node_scaling_factor: Union[float, int]
                    Scaling factor for node sizes

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        imfpt = calculate_mfpt_rates(
            results.transition_matrix,
            results.metastable_assignments,
            n_metastable_states,
            results.lag * results.dt_traj,
        )
        print(imfpt)
        G = nx.from_numpy_array(imfpt.numpy(), create_using=nx.MultiDiGraph)

        off_diags = ~torch.eye(imfpt.shape[0]).to(torch.bool)
        linewidth = (imfpt / imfpt.sum())[off_diags] * 10

        node_sizes = (
            np.array(
                [
                    results.stationary_distribution_dynammo[
                        results.metastable_assignments == i
                    ].sum()
                    for i in range(n_metastable_states)
                ]
            )
            * node_scaling_factor
        )

        nodes = nx.draw_networkx_nodes(
            G,
            node_positions,
            node_size=node_sizes,
            node_color=[
                list(next(ax._get_lines.prop_cycler).values())[0]
                for _ in range(n_metastable_states)
            ],
            ax=ax,
        )

        edges = nx.draw_networkx_edges(
            G,
            node_positions,
            node_size=node_sizes,
            arrowstyle="->",
            arrows=True,
            connectionstyle="arc3, rad = 0.05",
            ax=ax,
        )

        for i, edge in enumerate(edges):
            edge.set_linewidth(linewidth[i])

        ax.set_facecolor("white")

        if axis_labels:
            ax.set_xlabel("tIC1")
            ax.set_ylabel("tIC2")

        return ax

    def plot_trajectory(
        self,
        ax: List[plt.axis],
        results: Union[Path, str, ScenarioContainer] = None,
        axis_labels: bool = False,
    ):
        """
        Plots the kinetic network

        Parameters
        ----------
        ax:         List[matplotlib.pyplot.axis]
                    List of axes
        results:    Union[Path, str, ScenarioContainer], default = None
                    If is Path or str: results file will be used, else: ScenarioContainer will be used
        axis_labels: bool
                    Plots x- and y-axis labels if true

        n_metastable_states: int
                    Index of observable which should be plotted

        Returns
        -------
        ax:     matplotlib.pyplot.axis
                axis

        """
        if results is None:
            results = self._results[0]
        elif type(results) == ScenarioContainer:
            results = results
        else:
            results = self._process_results_file(results)

        if ax is None:
            ax = plt.gca()

        ftraj_ori = self.kwargs["ftraj"]
        dtraj_ori = self.kwargs["dtraj"]

        dtraj = dtraj_ori[np.in1d(dtraj_ori, results.original_active_set_dynammo)]
        ftraj = ftraj_ori[np.in1d(dtraj_ori, results.original_active_set_dynammo)]
        state_map = {
            ele.item(): results.active_set_dynammo[i].item()
            for i, ele in enumerate(results.original_active_set_dynammo)
        }
        dtraj = np.vectorize(state_map.get)(dtraj)
        # ftraj = np.vectorize(state_map.get)(ftraj)

        if not isinstance(ftraj, (np.ndarray, torch.tensor)):
            raise ValueError("kwarg `trajectory` must be a torch.Tensor or np.ndarray")

        if isinstance(ftraj, np.ndarray):
            ftraj = torch.from_numpy(ftraj)

        metastable_traj = results.metastable_assignments[dtraj]
        for i in np.unique(metastable_traj):
            ax.scatter(
                np.argwhere(metastable_traj == i) * 1e-2,
                ftraj[metastable_traj == i],
                s=0.1,
                edgecolors=None,
                linewidths=0,
            )

        ax.set_zorder(2)

        if axis_labels:
            ax.set_xlabel("Timesteps")
            ax.set_ylabel(self.kwargs["trajectory_ylabel"])

        return ax

    def _layout_subplot_grids_toy_systems(self, results: ScenarioContainer):
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = GridSpec(7, results.n_eigvals * 2, figure=fig)

        axes = dict()
        axes["eigfuncs-label"] = fig.add_subplot(gs[:2, :], frameon=False)
        axes["eigfuncs-label"].set_xticks([])
        axes["eigfuncs-label"].set_yticks([])

        for n in range(results.n_eigvals):
            idx = n * 2
            axes[f"eigfuncs-{n}"] = fig.add_subplot(gs[:2, idx : idx + 2])

        axes["stationary_distribution"] = fig.add_subplot(gs[2:4, : results.n_eigvals])
        axes["timescales"] = fig.add_subplot(gs[2:4, results.n_eigvals :])

        axes["loss"] = fig.add_subplot(gs[4:6, : results.n_eigvals])
        axes["observables"] = fig.add_subplot(gs[4, results.n_eigvals :])
        axes["observables-resiudals"] = fig.add_subplot(gs[5, results.n_eigvals :])

        axes["legend"] = fig.add_subplot(gs[-1, :], frameon=False)
        axes["legend"].set_xticks([])
        axes["legend"].set_yticks([])

        return fig, axes

    def _layout_subplot_grids_bpti_supplement(self, n_eigfuncs: int):
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        n_cols = n_eigfuncs * 2
        gs = GridSpec(9, n_cols, figure=fig)

        axes = dict()
        axes["eigfuncs-label"] = fig.add_subplot(gs[:4, :], frameon=False)
        axes["eigfuncs-label"].set_xticks([])
        axes["eigfuncs-label"].set_yticks([])

        for n in range(n_cols // 2):
            idx = n * 2
            axes[f"eigfuncs-{n}"] = fig.add_subplot(gs[:4, idx : idx + 2])

        axes["stationary_distribution"] = fig.add_subplot(gs[4:8, : n_cols // 2])
        axes["timescales"] = fig.add_subplot(gs[4:6, n_cols // 2 :])

        axes["loss"] = fig.add_subplot(gs[6:8, n_cols // 2 :])
        # axes['observables'] = fig.add_subplot(gs[4, results.n_eigvals:])
        # axes['observables-resiudals'] = fig.add_subplot(gs[5, results.n_eigvals:])

        axes["legend"] = fig.add_subplot(gs[-1, :], frameon=False)
        axes["legend"].set_xticks([])
        axes["legend"].set_yticks([])

        return fig, axes

    def _layout_subplot_grids_toy_figure_paper(self):
        fig = plt.figure(figsize=self.figsize)
        gs_top = GridSpec(7, 6, figure=fig, hspace=2.5, wspace=0.5)
        gs_bottom = GridSpec(7, 6, figure=fig, hspace=0.7, wspace=0.5, top=1.1)
        # gs.update(wspace=0.025, hspace=0.05)
        axes = dict()

        for i, key in enumerate(
            ["ppc-fep", "ppd-fep", "ppu-fep", "twc-fep", "twd-fep", "twu-fep"]
        ):
            axes[key] = fig.add_subplot(gs_top[:2, i])

        for i, key in enumerate(
            ["ppc-ts", "ppd-ts", "ppu-ts", "twc-ts", "twd-ts", "twu-ts"]
        ):
            axes[key] = fig.add_subplot(gs_top[2:4, i])

        for i, key in enumerate(
            ["ppc-obs", "ppd-obs", "ppu-obs", "twc-obs", "twd-obs", "twu-obs"]
        ):
            axes[key] = fig.add_subplot(gs_bottom[5, i])
            axes[f"{key}-res"] = fig.add_subplot(gs_bottom[6, i])

        for i, key in enumerate(["biased", "disconnected", "unobserved"]):
            for j, system in enumerate(["pp", "tw"]):
                axes[f"{key}-{system}"] = fig.add_subplot(
                    gs_top[0, i + 3 * j], frameon=False
                )
                axes[f"{key}-{system}"].set_xticks([])
                axes[f"{key}-{system}"].set_yticks([])
                axes[f"{key}-{system}"].set_title(key)

        axes["legend"] = fig.add_subplot(frameon=False)
        axes["legend"].set_xticks([])
        axes["legend"].set_yticks([])

        axes["obs-ylabel"] = fig.add_subplot(gs_bottom[5:, :], frameon=False)
        axes["obs-ylabel"].set_xticks([])
        axes["obs-ylabel"].set_yticks([])

        fig.legend(
            handles=[
                mpatches.Patch(color=self.colors[i], label=label)
                for i, label in enumerate(self.labels)
            ],
            loc=8,
            ncol=len(self.labels),
        )

        return fig, axes

    def _layout_subplot_grids_bpti(self):
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = GridSpec(9, 3, figure=fig)

        axes = dict()

        for i in range(3):
            for j in range(3):
                row = i * 3
                col = j
                axes[f"observables-{i}{j}"] = fig.add_subplot(gs[row : row + 2, col])
                axes[f"observables-residuals-{i}{j}"] = fig.add_subplot(
                    gs[row + 2, col]
                )

        return fig, axes

    def _layout_subplot_grids_bpti_network(self):
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = GridSpec(4, 6, figure=fig)

        axes = dict()

        axes["network"] = fig.add_subplot(gs[:4, :4])
        axes["timescales"] = fig.add_subplot(gs[:, 4:])

        return fig, axes

    def _layout_subplot_grids_bpti_disconnected(self):
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        gs = GridSpec(12, 4, figure=fig)

        axes = dict()

        axes["network"] = fig.add_subplot(gs[:8, :2])
        axes["timescales"] = fig.add_subplot(gs[:8, 2:])

        for i in range(4):
            axes[f"observables-{i}"] = fig.add_subplot(gs[8:11, i])
            axes[f"observables-residuals-{i}"] = fig.add_subplot(gs[11, i])

        return fig, axes

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, new_loss_val: float):
        if type(new_loss_val) == torch.Tensor:
            self._loss.append(new_loss_val.detach().numpy())
        elif type(new_loss_val) == float:
            self._loss.append(new_loss_val)
        else:
            raise ValueError(
                f"Check format of loss value. Expected `float` or `torch.Tensor` and not"
                f" `{type(new_loss_val)}`."
            )

    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results_input: Union[List[Path], Tuple[Path], Path]):
        if type(results_input) == list or type(results_input) == tuple:
            self._results = [
                self._process_results_file(filename) for filename in results_input
            ]
        elif type(results_input) in [
            Path,
            pathlib.PosixPath,
            pathlib.PurePosixPath,
            pathlib.PurePath,
            str,
        ]:
            self._results = [self._process_results_file(results_input)]
        elif results_input is None:
            self._results = []
        else:
            raise ValueError(
                f"Results input must be either a (list of) pathlib.Path object(s) or string(s) not "
                f"`{type(results_input)}`."
            )

    @results.getter
    def results(self):
        return self._results

    def _process_results_file(self, results_filename):
        # load dynAMMo results
        results_file = h5py.File(results_filename, "r")

        n_run = "0"
        # stack tensors
        eigvals_dynammo = torch.vstack(
            [
                torch.tensor(results_file[group]["eigvals"]).detach()
                for group in results_file.keys()
            ]
        )
        stationary_distribution_dynammo = torch.vstack(
            [
                torch.tensor(results_file[group]["stationary_distribution"]).detach()
                for group in results_file.keys()
            ]
        )
        reigvecs_dynammo = torch.stack(
            [
                torch.tensor(results_file[group]["reigvecs"][:, 1:]).detach()
                for group in results_file.keys()
            ]
        )
        active_set_dynammo = torch.tensor(results_file[n_run]["active_set"])
        original_active_set_dynammo = torch.tensor(
            results_file[n_run].get("original_active_set", active_set_dynammo)
        )
        dynamic_observables_dynammo = torch.from_numpy(
            np.array(results_file[n_run]["dynamic_observables_dynammo"])
        )
        dynamic_observables_gt = torch.from_numpy(
            np.array(results_file[n_run]["dynamic_observables_experimental"])
        )
        dynamic_observables_msm = torch.from_numpy(
            np.array(results_file[n_run]["dynamic_observables_msm"])
        )
        a = torch.abs(torch.tensor(results_file[n_run]["a"]))
        b = torch.tensor(results_file[n_run]["b"])
        indep_var = torch.from_numpy(np.array(results_file[n_run]["indep_var"]))
        dynamic_observables_by_state = torch.from_numpy(
            np.array(results_file[n_run]["dynamic_observables_by_state"])
        )
        transition_matrix = torch.from_numpy(
            np.array(results_file[n_run]["transition_matrix"])
        )
        loss = torch.tensor(results_file[n_run].get("loss", default=0))
        config_filename = results_file.attrs["config_filename"]
        results_file.close()

        # load config file
        with open(ROOTDIR / config_filename, "r") as config_file:
            config = json.load(config_file)

        n_eigvals = config["n_eigvals"]
        n_metastable_states = config["n_metastable_states"]
        lag = config["lag"]
        dt_traj = config["dt_traj"]
        if len(config["inp_calculated"]) > 1:
            disconnected = True
        else:
            disconnected = False

        pcca = calculate_metastable_decomposition(
            transition_matrix, n_metastable_states
        )

        if self.kwargs.get("state_assignment_dict") is None:
            mapped_metastable_assignment = pcca.assignments
        else:
            # metastable_assignment = calculate_metastable_assignment(transition_matrix, n_metastable_states)
            # assignment_counts = np.array([(metastable_assignment == i).sum() for i in np.unique(metastable_assignment)])
            # sorted_state_indices = np.argsort(assignment_counts)
            # state_map = {ele: sorted_state_indices[i] for i, ele in enumerate(range(n_metastable_states))}
            # mapped_metastable_assignment = torch.from_numpy(np.vectorize(state_map.get)(metastable_assignment))
            state_map = self.kwargs["state_assignment_dict"]
            mapped_metastable_assignment = np.vectorize(state_map.get)(pcca.assignments)

        # eigvals
        eigvals_mean = eigvals_dynammo.mean(axis=0)[:n_eigvals]

        # timescales (mean and confidence intervals)
        timescales = timescales_from_eigvals(
            eigvals_dynammo[:, :n_eigvals], lag=lag, dt_traj=dt_traj
        )
        timescales_dynammo = timescales_from_eigvals(
            eigvals_mean, lag=lag, dt_traj=dt_traj
        )
        timescales_dynammo_lower, timescales_dynammo_upper = confidence_interval(
            timescales
        )

        # stationary distribution (mean and confidence intervals)
        stationary_distribution_dynammo_lower, stationary_distribution_dynammo_upper = (
            confidence_interval(stationary_distribution_dynammo)
        )
        stationary_distribution_dynammo = stationary_distribution_dynammo.mean(axis=0)

        # eigenfunctions (mean and confidence intervals)
        reigvecs_dynammo_lower, reigvecs_dynammo_upper = confidence_interval(
            reigvecs_dynammo
        )
        reigvecs_dynammo = torch.squeeze(reigvecs_dynammo.mean(axis=0))

        leigvecs_dynammo = calculate_leigvecs(
            stationary_distribution_dynammo, reigvecs_dynammo
        )

        # observables
        dynamic_observables_dynammo = (
            a[:, None] * dynamic_observables_dynammo + b[:, None]
        )
        chi_squared = dynAMMo.tools.utils.calculate_chi_squared(
            dynamic_observables_dynammo, dynamic_observables_gt
        )

        # handle missing states case
        if config["missing"]:
            missing = True
            eigvals_dynammo = torch.concat((torch.tensor([torch.nan]), eigvals_mean))[
                :n_eigvals
            ]
            insert = torch.tensor(
                [torch.nan for _ in range(reigvecs_dynammo.size()[0])]
            )
            reigvecs_dynammo = torch.hstack([insert[:, None], reigvecs_dynammo])
            timescales_dynammo = torch.concat(
                (torch.tensor([np.nan]), timescales_dynammo)
            )[:n_eigvals]
            timescales_dynammo_lower = np.concatenate(
                ([np.nan], timescales_dynammo_lower)
            )[:n_eigvals]
            timescales_dynammo_upper = np.concatenate(
                ([np.nan], timescales_dynammo_upper)
            )[:n_eigvals]
        else:
            missing = False

        timescales_dynammo_lower = torch.from_numpy(timescales_dynammo_lower)
        timescales_dynammo_upper = torch.from_numpy(timescales_dynammo_upper)

        # Experimental and MSM stuff
        if self.plot_msm:
            msm_available = True
            hdf_loaders = [
                HDFLoader(ROOTDIR / inp_dir)
                for i, inp_dir in enumerate(config["inp_calculated"])
            ]

            T_calcs = [loader.read("T") for loader in hdf_loaders]
            C_calcs = [loader.read("count_matrix") for loader in hdf_loaders]
            msm_initial = DynamicAugmentedMarkovModel(
                transition_matrices=T_calcs,
                count_models=C_calcs,
                initialize_via_count_matrix=config.get(
                    "initialize_via_count_matrix", False
                ),
            )

            reigvecs_msm = msm_initial.reigvecs
            eigvals_msm = msm_initial.eigvals
            stationary_distribution_msm = msm_initial.stationary_distribution

            timescales_msm = timescales_from_eigvals(eigvals_msm, lag, dt_traj)[
                :n_eigvals
            ]
        else:
            msm_available = False
            eigvals_msm = None
            timescales_msm = None
            stationary_distribution_msm = None
            reigvecs_msm = None

        if self.plot_experiments:
            experiments_available = True
            hdf_loader = HDFLoader(ROOTDIR / config["inp_experimental"])

            reigvecs_gt = hdf_loader.read("reigvecs")
            eigvals_gt = hdf_loader.read("eigvals")
            stationary_distribution_gt = hdf_loader.read("stationary_distribution")
            active_set_gt = hdf_loader.read(
                "active_set",
                default=torch.arange(0, stationary_distribution_gt.size()[0]),
            )
            timescales_gt = timescales_from_eigvals(eigvals_gt, lag)[1 : n_eigvals + 1]
            reigvecs_gt = reigvecs_gt[:, 1:]
        else:
            experiments_available = False
            eigvals_gt = None
            timescales_gt = None
            stationary_distribution_gt = None
            reigvecs_gt = None
            active_set_gt = None

        # save results in ScenarioContainer
        scenario = ScenarioContainer(
            config=config,
            lag=lag,
            dt_traj=dt_traj,
            n_eigvals=n_eigvals,
            n_metastable_states=n_metastable_states,
            disconnected=disconnected,
            missing=missing,
            transition_matrix=transition_matrix,
            pcca=pcca,
            metastable_assignments=mapped_metastable_assignment,
            eigvals_dynammo=eigvals_dynammo,
            timescales_dynammo=timescales_dynammo,
            timescales_dynammo_upper=timescales_dynammo_upper,
            timescales_dynammo_lower=timescales_dynammo_lower,
            stationary_distribution_dynammo=stationary_distribution_dynammo,
            stationary_distribution_dynammo_lower=stationary_distribution_dynammo_lower,
            stationary_distribution_dynammo_upper=stationary_distribution_dynammo_upper,
            reigvecs_dynammo=reigvecs_dynammo,
            reigvecs_dynammo_lower=reigvecs_dynammo_lower,
            reigvecs_dynammo_upper=reigvecs_dynammo_upper,
            leigvecs_dynammo=leigvecs_dynammo,
            eigvals_gt=eigvals_gt,
            timescales_gt=timescales_gt,
            stationary_distribution_gt=stationary_distribution_gt,
            reigvecs_gt=reigvecs_gt,
            eigvals_msm=eigvals_msm,
            timescales_msm=timescales_msm,
            stationary_distribution_msm=stationary_distribution_msm,
            reigvecs_msm=reigvecs_msm,
            active_set_dynammo=active_set_dynammo,
            active_set_gt=active_set_gt,
            original_active_set_dynammo=original_active_set_dynammo,
            dynamic_observables_dynammo=dynamic_observables_dynammo,
            dynamic_observables_gt=dynamic_observables_gt,
            dynamic_observables_msm=dynamic_observables_msm,
            a=a,
            b=b,
            indep_var=indep_var,
            chi_squared=chi_squared,
            dynamic_observables_by_state=dynamic_observables_by_state,
            msm_available=msm_available,
            experiments_available=experiments_available,
            loss=loss,
        )
        return scenario
