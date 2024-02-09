from dataclasses import dataclass

import numpy as np
import torch
from typing import *


class Experiment:
    """
    Base class for representing an experiment.
    """

    def __init__(
        self,
        _,
        observables_exp: torch.Tensor = None,
        observables_pred: torch.Tensor = None,
        observables_msm: torch.Tensor = None,
        observables_by_state: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        name: str = None,
    ):
        """
        Initializes the Experiment object.

        Parameters
        ----------
        observables_exp : torch.Tensor, optional
            Experimental observable data.
        observables_pred : torch.Tensor, optional
            Predicted observable data.
        observables_msm : torch.Tensor, optional
            Observable data from Markov State Model.
        observables_by_state : torch.Tensor or Tuple[torch.Tensor], optional
            Observable data by state.
        name : str, optional
            Name of the experiment.
        """
        self.observables_exp = observables_exp
        self._observables_pred = observables_pred
        self.observables_msm = observables_msm
        self.observables_by_state = _initialize_observables_by_state(
            observables_by_state
        )
        self.name = name

    @property
    def observables_pred(self):
        """Gets the predicted observable."""
        return self._observables_pred

    @observables_pred.setter
    def observables_pred(self, observables_pred: torch.Tensor):
        """Sets the predicted observable with size validation."""
        self._set_observables_pred(observables_pred)

    def _set_observables_pred(self, observables_pred):
        """Validates and sets the predicted observable."""
        if self.observables_exp is not None and observables_pred is not None:
            assert observables_pred.size() == self.observables_exp.size()
        self._observables_pred = observables_pred


class StaticExperiment(Experiment):
    """
    Represents a static experiment. Inherits from Experiment.

    A static experiment typically involves data that does not change over time.
    """

    def __init__(
        self,
        observables_exp: torch.Tensor = None,  # Experimental output
        observables_pred: torch.Tensor = None,  # Prediction from dynAMMO
        observables_msm: torch.Tensor = None,  # Prediction from MSM
        observables_by_state: Union[
            torch.Tensor, Tuple[torch.Tensor]
        ] = None,  # Microscopic observable
        name: str = None,
    ):
        """
        Initializes the StaticExperiment object.

        Parameters
        ----------
        observables_exp : torch.Tensor, optional
            Experimental observable data.
        observables_pred : torch.Tensor, optional
            Predicted observable data.
        observables_by_state : torch.Tensor or Tuple[torch.Tensor], optional
            Observable data by state.
        name : str, optional
            Name of the experiment.
        """
        super().__init__(
            self,
            observables_exp=observables_exp,
            observables_pred=observables_pred,
            observables_msm=observables_msm,
            observables_by_state=observables_by_state,
            name=name,
        )

    def __call__(self, pi: torch.Tensor) -> torch.Tensor:
        """
        Calculates the expected value of the observables in the stationary distribution.

        Parameters
        ----------
        pi : torch.Tensor
            Stationary distribution of the Markov state model.

        Returns
        -------
        torch.Tensor
            Expected value of the observables.
        """
        return torch.atleast_1d(pi.matmul(self.observables_by_state))

    def __repr__(self):
        return f"StaticExperiment {self.name}"


class DynamicExperiment(Experiment):
    """
    Represents a dynamic experiment. Inherits from Experiment.

    A dynamic experiment typically involves data that changes over time or depends on an independent variable.
    """

    def __init__(
        self,
        indep_var: torch.Tensor = None,  # Independent variable to observable_function
        observables_exp: torch.Tensor = None,  # Experimental observable
        observables_pred: torch.Tensor = None,  # Prediction
        observables_msm: torch.Tensor = None,  # MSM observable
        observables_by_state: Union[
            torch.Tensor, Tuple[torch.Tensor]
        ] = None,  # Microscopic observable
        observable_function: Callable = None,
        name: str = None,
    ):
        """
        Initializes the DynamicExperiment object.

        Parameters
        ----------
        indep_var : torch.Tensor, optional
            Independent variable for the observable function.
        observables_exp : torch.Tensor, optional
            Experimental observable data.
        observables_pred : torch.Tensor, optional
            Predicted observable data.
        observables_msm : torch.Tensor, optional
            Observable data from Markov State Model.
        observables_by_state : torch.Tensor or Tuple[torch.Tensor], optional
            Observable data by state.
        observable_function : Callable, optional
            Function to calculate observables.
        name : str, optional
            Name of the experiment.
        """
        super().__init__(
            self,
            observables_exp=observables_exp,
            observables_pred=observables_pred,
            observables_msm=observables_msm,
            observables_by_state=observables_by_state,
            name=name,
        )
        self.indep_var = indep_var
        self.observable_function = observable_function
        self.auto_correlation = _determine_auto_correlation(observables_by_state)

    def __call__(self, leigvecs, eigvals, lagtime: int, **kwargs) -> torch.Tensor:
        """
        Executes the observable function for dynamic experiments.

        Parameters
        ----------
        leigvecs : torch.Tensor
            Left eigenvectors of the transition matrix.
        eigvals : torch.Tensor
            Eigenvalues of the transition matrix.
        lagtime : int
            Lag time for the Markov state model.

        Returns
        -------
        torch.Tensor
            Calculated observables.
        """
        if self.auto_correlation:
            return self.observable_function(
                self.indep_var,
                self.observables_by_state,
                leigvecs,
                eigvals,
                lagtime,
                **kwargs,
            )
        else:
            return self.observable_function(
                self.indep_var,
                *self.observables_by_state,
                leigvecs,
                eigvals,
                lagtime,
                **kwargs,
            )

    def __repr__(self):
        return f"DynamicExperiment {self.name}"

    def get_data_dim(self) -> int:
        """
        Retrieves the dimension of the independent variable data.

        Returns
        -------
        int
            Dimension of the independent variable data.
        """
        return self.indep_var.size()[0]


@dataclass
class ExperimentalData:
    """
    Class for storing experimental data such as R1Rho, CPMG, and other types.

    Attributes
    ----------
    name : str
        Name of the experiment.
    data : torch.Tensor
        Experimental data.
    error_data : torch.Tensor
        Error data associated with the experiment.
    observables_by_state : torch.Tensor
        Observable data by state.
    indep_var : torch.Tensor
        Independent variable data.
    resids : torch.Tensor
        Residuals data.
    resname : torch.Tensor
        Residue names.
    temperature : int
        Temperature at which the experiment was conducted.
    field_strength : int
        Magnetic field strength used in the experiment.
    """

    name: str = None
    data: torch.Tensor = None
    error_data: torch.Tensor = None
    observables_by_state: torch.Tensor = None
    indep_var: torch.Tensor = None
    resids: torch.Tensor = None
    resname: torch.Tensor = None
    temperature: int = None
    field_strength: int = None

    def __repr__(self) -> str:
        return f"ExperimentalData {self.name}"


def _initialize_observables_by_state(
    observables_by_state: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    """Initializes the observable by state."""
    if isinstance(observables_by_state, torch.Tensor):
        return observables_by_state
    elif isinstance(observables_by_state, np.ndarray):
        return torch.tensor(observables_by_state)
    elif isinstance(observables_by_state, tuple):
        assert len(observables_by_state) == 2
        return observables_by_state
    return None


def _determine_auto_correlation(
    observables_by_state: Union[torch.Tensor, Tuple[torch.Tensor]]
) -> bool:
    """
    Determines whether the experiment involves autocorrelation based on the observable by state.

    Parameters
    ----------
    observables_by_state : Union[torch.Tensor, Tuple[torch.Tensor]]

    Returns
    -------
    bool
        True if the experiment involves autocorrelation, False otherwise.
    """
    return not isinstance(observables_by_state, tuple)
