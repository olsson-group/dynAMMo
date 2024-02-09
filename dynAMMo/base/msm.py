import torch
from typing import *

import numpy as np
from scipy.linalg import block_diag
from deeptime.markov import TransitionCountModel
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.msm import MarkovStateModel

from dynamics_utils.msm import eigendecomposition, calculate_leigvecs
from deeptime.markov.msm import MarkovStateModelCollection


class DynamicAugmentedMarkovModel(MarkovStateModelCollection):
    def __init__(
        self,
        transition_matrices: List[np.ndarray] = None,
        stationary_distributions: List[np.ndarray] = None,
        count_models: List[TransitionCountModel] = None,
        transition_matrix_tolerance: float = 1e-10,
        reversible: Optional[bool] = True,
        observables_by_state: Union[torch.Tensor, List[torch.Tensor]] = None,
        initialize_via_count_matrix: bool = True,
    ):
        """
        Initialize the DynamicAugmentedMarkovModel.

        Parameters
        ----------
        transition_matrices : List[np.ndarray]
            List of transition matrices.
        stationary_distributions : List[np.ndarray]
            List of stationary distributions.
        count_models : List[TransitionCountModel]
            List of transition count models.
        transition_matrix_tolerance : float
            Tolerance level for transition matrices.
        reversible : bool
            Indicates if the model is reversible.
        observables_by_state : Union[torch.Tensor, List[torch.Tensor]]
            Observables by state as a tensor or list of tensors.
        initialize_via_count_matrix : bool
            Flag to initialize using a count matrix.
        """

        # Basic properties of the model
        self.transition_matrices = transition_matrices
        self.stationary_distributions = stationary_distributions
        self.count_models = count_models
        self.reversible = reversible
        self.transition_matrix_tolerance = transition_matrix_tolerance
        self.observables_by_state = observables_by_state
        self.initialize_via_count_matrix = initialize_via_count_matrix

        # Validate model reversibility
        if not self.reversible:
            raise NotImplementedError(
                "Only reversible MSMs are supported at the moment"
            )

        # Internal state initialization
        self.deeptime_msm = None
        self._count_model = None
        self._count_model_init = None
        self._count_matrix = None
        self._count_matrix_init = None
        self._transition_matrix = None
        self._active_set = None
        self.n_states = None

        # Build a connected Markov State Model
        self.build_connected_msm(use_count_matrix=self.initialize_via_count_matrix)

        # If count model is available, update the state space information
        if self._count_model is not None:
            self.n_states = self._count_model_init.n_states
            self._active_set = self._count_model_init.connected_sets()[0]

        # Eigendecomposition to calculate eigenvalues and eigenvectors
        self.reigvecs, self.eigvals, self.pi = eigendecomposition(
            self.transition_matrix
        )
        self.leigvecs = calculate_leigvecs(self.pi, self.reigvecs)

        # Set stationary distribution and active set as class properties
        self.stationary_distribution = self.pi
        self.original_active_set = self._active_set
        # TODO: super().__init__()?

    def build_connected_msm(self, use_count_matrix: bool = True):
        """
        Builds a connected Markov State Model (MSM) from the disconnected transition matrices, count matrices, and observables.

        This method initializes the count matrix and count model, and then determines the transition matrix
        either from the count matrix or directly, based on the `use_count_matrix` parameter.

        Parameters
        ----------
        use_count_matrix : bool, optional, default=True
            If True, the count matrix is used to initialize the MSM. If False, the transition matrix is used.

        Returns
        -------
        None
        """
        # Initialize the count matrices for the model
        self._count_matrix = self._initialize_count_matrix()
        self._count_matrix_init = self._calculate_count_matrix_init()

        # Create transition count models from the count matrices
        self._count_model_init = TransitionCountModel(self._count_matrix_init)
        self._count_model = TransitionCountModel(self._count_matrix)

        # Initialize the observables by state
        self._observables_by_state = self._initialize_observables_by_state()

        # Calculate the transition matrix based on the method specified
        if use_count_matrix:
            # Calculate the transition matrix from count matrix
            self._transition_matrix = self.calculate_transition_matrix_from_counts()
        else:
            # Initialize the transition matrix directly
            self._transition_matrix = self._initialize_transition_matrix()

    def _initialize_observables_by_state(self):
        """
        Initializes and normalizes the observables by state if they are provided.

        This method concatenates the observables if they are in a disconnected state and
        then centers them by subtracting their mean. If observables are not provided,
        the method returns None.

        Returns
        -------
        np.ndarray or None
            The concatenated and mean-centered observables by state, or None if no observables are provided.
        """
        # Return None if no observables are provided
        if self._observables_by_state is None:
            return None

        # Concatenate observables from different states
        observables_by_state_disconnected = torch.hstack(tuple(self._observables_by_state))

        # Ensure dimensionality
        observables_by_state_disconnected = torch.atleast_2d(observables_by_state_disconnected)

        # Mean center the observables by subtracting the column-wise mean
        observables_by_state_disconnected -= observables_by_state_disconnected.mean(
            axis=1, keepdims=True
        )

        return observables_by_state_disconnected

    def _initialize_count_matrix(self):
        """
        Initializes the count matrix by combining count matrices from individual models.

        Returns
        -------
        np.ndarray
            The combined count matrix as a block diagonal matrix.
        """
        # Create a block diagonal matrix from individual count matrices
        return block_diag(
            *[count_matrix.count_matrix for count_matrix in self._count_models]
        )

    def _calculate_count_matrix_init(self):
        """
        Calculates the initial count matrix with adjustments for disconnected states.

        Returns
        -------
        np.ndarray or None
            The adjusted initial count matrix, or None if count models are not available.
        """
        if self._count_models is None:
            return None

        # Scale the count matrix
        C_disconnected_init = self._count_matrix * 1e3
        len_submatrices = tuple(
            [len(submatrix.count_matrix) for submatrix in self.count_models]
        )

        # Initialize off-diagonal elements with small values
        submatrices = tuple([np.ones([length, length]) for length in len_submatrices])
        concatenated = block_diag(*submatrices)
        inverted = np.logical_not(concatenated).astype(bool)
        off_diag_matrix = inverted.astype(int)
        off_diag_idx = np.argwhere(off_diag_matrix)

        # Set off-diagonal elements in the initial count matrix
        C_disconnected_init[off_diag_idx[:, 0], off_diag_idx[:, 1]] = 1
        C_disconnected_init[off_diag_idx[:, 1], off_diag_idx[:, 0]] = 1

        return C_disconnected_init

    def calculate_transition_matrix_from_counts(self):
        """
        Calculates the transition matrix from the initial count matrix using the deeptime MSM estimator.

        Returns
        -------
        np.ndarray
            The transition matrix calculated from the count matrix.
        """
        # Fit a reversible Maximum Likelihood MSM model using the count matrix
        self.deeptime_msm = (
            MaximumLikelihoodMSM(reversible=True)
            .fit(self._count_matrix_init)
            .fetch_model()
        )
        # Retrieve and return the transition matrix
        return self.deeptime_msm.transition_matrix

    def _initialize_transition_matrix(self):
        """
        Initializes the transition matrix from the provided transition matrices.

        If only one transition matrix is provided, it's used directly. Otherwise, a connected
        transition matrix is constructed from multiple matrices, ensuring reversibility.

        Returns
        -------
        np.ndarray or None
            The initialized transition matrix or None if no transition matrices are provided.
        """
        # Return None if no transition matrices are provided
        if self._transition_matrices is None:
            return None

        # Handle the case with a single transition matrix
        if len(self.transition_matrices) == 1:
            T_disconnected = self.transition_matrices[0].numpy()
        else:
            # Process multiple transition matrices to construct a connected transition matrix
            T_disconnected = self._construct_connected_transition_matrix()

        return T_disconnected

    def _construct_connected_transition_matrix(self):
        """
        Constructs a connected transition matrix from multiple transition matrices.

        Ensures that the final matrix is reversible and connected.

        Returns
        -------
        np.ndarray
            The constructed connected transition matrix.
        """
        pis = [eigendecomposition(T)[-1] for T in self.transition_matrices]
        T_norm = [
            torch.diag(pis[i]).mm(self.transition_matrices[i])
            for i in range(len(self.transition_matrices))
        ]
        T_disconnected = torch.block_diag(*T_norm).to(torch.float64)
        T_disconnected = T_disconnected.numpy()
        len_submatrices = tuple(
            [len(submatrix) for submatrix in self._transition_matrices]
        )

        # Fill off-diagonal elements with small values to ensure connectivity
        off_diag_matrix = self._create_off_diagonal_matrix(len_submatrices)
        off_diag_idx = np.argwhere(off_diag_matrix)
        T_disconnected = self._ensure_connectivity(T_disconnected, off_diag_idx)

        return T_disconnected

    def _create_off_diagonal_matrix(self, len_submatrices):
        """
        Creates a matrix with ones on the off-diagonal elements.

        Parameters
        ----------
        len_submatrices : tuple
            Tuple containing the sizes of the submatrices.

        Returns
        -------
        np.ndarray
            The matrix with ones on the off-diagonal elements.
        """
        submatrices = tuple([np.ones([length, length]) for length in len_submatrices])
        concatenated = block_diag(*submatrices)
        inverted = np.logical_not(concatenated).astype(bool)
        return inverted.astype(int)

    def _ensure_connectivity(self, T_disconnected, off_diag_idx):
        """
        Modifies the disconnected transition matrix to ensure it is connected.

        Parameters
        ----------
        T_disconnected : np.ndarray
            The initial disconnected transition matrix.
        off_diag_idx : np.ndarray
            Indices of off-diagonal elements.

        Returns
        -------
        np.ndarray
            The modified transition matrix that is connected.

        Raises
        ------
        ValueError
            If the transition matrices cannot be connected.
        """
        connected = False
        off_diag_val = 1e-10

        while not connected:
            off_diag_val *= 1e1
            T_disconnected[off_diag_idx[:, 0], off_diag_idx[:, 1]] = off_diag_val
            T_disconnected[off_diag_idx[:, 1], off_diag_idx[:, 0]] = off_diag_val

            # Normalize the matrix
            T_disconnected /= T_disconnected.sum(axis=1, keepdims=True)

            # Check connectivity
            msm = MarkovStateModel(transition_matrix=T_disconnected, reversible=True)
            if np.isclose(msm.eigenvalues(5), 1.0).sum() > 1:
                connected = True

            if off_diag_val == 1e0:
                raise ValueError("Could not connect transition matrices")

        return T_disconnected

    # Property for accessing and setting transition matrices
    @property
    def transition_matrices(self):
        """Gets the transition matrices."""
        return self._transition_matrices

    @transition_matrices.setter
    def transition_matrices(self, value):
        """
        Sets the transition matrices, ensuring they are a list of numpy arrays or torch tensors.
        Converts torch tensors to numpy arrays.
        """
        # Validation and conversion of input to numpy arrays
        self._transition_matrices = _validate_and_convert_input(
            value, expected_type=[np.ndarray, torch.Tensor]
        )

    @transition_matrices.getter
    def transition_matrices(self):
        """Returns the transition matrices as torch tensors if available."""
        return (
            [torch.from_numpy(x) for x in self._transition_matrices]
            if self._transition_matrices is not None
            else None
        )

    # Property for accessing and setting count models
    @property
    def count_models(self):
        """Gets the count models."""
        return self._count_models

    @count_models.setter
    def count_models(self, value):
        """
        Sets the count models, converting input tensors or matrices to TransitionCountModel objects.
        """
        # Validation and conversion of input to TransitionCountModel objects
        validated_input = _validate_and_convert_input(
            value, expected_type=[np.ndarray, torch.Tensor, TransitionCountModel]
        )

        if all(isinstance(val, TransitionCountModel) for val in validated_input):
            self._count_models = validated_input
        else:
            self._count_models = [
                TransitionCountModel(val, "sliding") for val in validated_input if type(val) is not TransitionCountModel
            ]

    # @count_models.getter
    # def count_models(self):
    #         return self._count_models

    @property
    def transition_matrix(self):
        """Gets the transition matrix, converting to torch.Tensor if it's not None."""
        if self._transition_matrix is not None:
            return torch.from_numpy(self._transition_matrix)
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, value):
        """Sets the transition matrix."""
        self._transition_matrix = value

    @property
    def observables_by_state(self):
        """Gets the observables by state, converting to torch.Tensor if they're not None."""
        if type(self._observables_by_state) is np.ndarray:
            return torch.from_numpy(self._observables_by_state)
        return self._observables_by_state

    @observables_by_state.setter
    def observables_by_state(self, value):
        """Sets the observables by state."""
        self._observables_by_state = value

    @property
    def count_model(self):
        """Gets the count model."""
        return self._count_model

    @count_model.setter
    def count_model(self, value):
        """Sets the count model."""
        self._count_model = value

    @property
    def count_matrix(self):
        """Gets the count matrix, applying a transformation if it's not None."""
        if self._count_matrix is not None:
            # Apply a specific transformation to the count matrix
            count_matrix = self._count_matrix[self.active_set][:, self.active_set]
            return torch.from_numpy(count_matrix)
        return self._count_matrix

    @count_matrix.setter
    def count_matrix(self, value):
        """Sets the count matrix."""
        self._count_matrix = value

    @property
    def n_states(self):
        """Gets the number of states."""
        return self._n_states

    @n_states.setter
    def n_states(self, value):
        """Sets the number of states."""
        self._n_states = value

    @property
    def reversible(self):
        """Gets the reversible status."""
        return self._reversible

    @reversible.setter
    def reversible(self, value):
        """Sets the reversible status."""
        self._reversible = value

    @property
    def transition_matrix_tolerance(self):
        """Gets the transition matrix tolerance."""
        return self._transition_matrix_tolerance

    @transition_matrix_tolerance.setter
    def transition_matrix_tolerance(self, value):
        """Sets the transition matrix tolerance."""
        self._transition_matrix_tolerance = value

    @property
    def stationary_distribution(self):
        """Gets the stationary distribution."""
        return self._stationary_distribution

    @stationary_distribution.setter
    def stationary_distribution(self, value):
        """Sets the stationary distribution."""
        self._stationary_distribution = value

    @property
    def active_set(self):
        """Gets the active set."""
        return self._active_set

    @active_set.setter
    def active_set(self, value):
        """
        Sets the active set. The value can be a list, a torch tensor, a numpy array, or None.

        Parameters
        ----------
        value : list, torch.Tensor, np.ndarray, or None
            The new value for the active set.

        Raises
        ------
        ValueError
            If the value is not a list, torch tensor, numpy array, or None.
        """
        # Validate and set the active set based on its type
        if isinstance(value, list):
            self._active_set = torch.concat(value)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            self._active_set = value
        elif value is None:
            self._active_set = None
        else:
            raise ValueError("active_set must be a list, torch tensor or numpy array")


def _validate_and_convert_input(value, expected_type):
    """
    Validates the input value and converts it to the desired format.

    Parameters
    ----------
    value : Any
        The input value to be validated and converted.
    expected_type : list
        A list of expected types for the input value.

    Returns
    -------
    Any
        The converted value.

    Raises
    ------
    ValueError
        If the input value is not of the expected type.
    """
    # Validate the input type
    if not isinstance(value, list):
        raise ValueError(
            f"Input must be a list of {', '.join([t.__name__ for t in expected_type])}"
        )

    # Convert torch tensors to numpy arrays, or retain numpy arrays as they are
    return [
        x.numpy() if isinstance(x, torch.Tensor) else x
        for x in value
        if isinstance(x, tuple(expected_type))
    ]
