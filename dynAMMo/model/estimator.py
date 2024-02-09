from typing import *

import torch
from torch.nn import Parameter
from dynamics_utils.msm import row_normalise, rdl_recomposition

from ..base.experiments import StaticExperiment, DynamicExperiment
from ..base.static_amm import AugmentedMSMEstimator
from ..tools.utils import pad_and_stack_tensors
from ..base.msm import DynamicAugmentedMarkovModel


class DynamicAugmentedMarkovModelEstimator(torch.nn.Module):
    """
    Estimator for a Dynamic Augmented Markov Model (dynAMMo), which is designed to integrate both
    dynamic and static experimental data to improve the estimation of Markov state models (MSMs).

    Attributes
    ----------
    msm : DynamicAugmentedMarkovModel
        An instance of a dynamic augmented Markov model.
    experiments : List[Union[DynamicExperiment, StaticExperiment]]
        A list of experiments, which can be either dynamic or static.
    lag : int
        The lag time used in the Markov state model.
    dt_traj : float, default=1.0
        The time step between frames in the trajectory data.
    eigfuncs : Any, optional
        Eigenfunction orthonormality constraint, e.g., an instance of a Stiefel manifold updater.
    config: dict, optional
        Config dictionary
    alpha : int, default=-13
        Scaling factor for the negative log-likelihood and Lagrange loss.
    beta : float, default=1e-3
        Scaling factor for the experimental loss.
    modify_n_eigvals : Union[None, int], default=None
        If not None, specifies the number of eigenvalues to modify to avoid overfitting.
    eps : float, default=0.02
        Convergence threshold for the stationary distribution.
    static_amm_eps : float, default=0.05
        Convergence threshold for the augmented Markov model when using static data.
    omega_scale : float, default=0.01
        Scaling factor for experimental weights in AMM
    dtype : str, default='torch.float64'
        Data type for tensors.
    max_iter_static_amm : int, default=50000
        Maximum number of iterations for the augmented Markov model using static data.
    std_noise : float, default=1e-2
        Standard deviation of the noise added to the parameters to avoid overfitting.
    """

    def __init__(
        self,
        msm: DynamicAugmentedMarkovModel,
        experiments: List[Union[DynamicExperiment, StaticExperiment]],
        eigfuncs: Any = None,
        config: dict = None,
        lag: int = 1,
        dt_traj: float = 1.0,
        alpha: int = -13,
        beta: float = 1e-3,
        modify_n_eigvals: Union[None, int] = None,
        eps: float = 0.02,
        static_amm_eps: float = 0.05,
        omega_scale: float = 0.01,
        dtype: torch.dtype = torch.float64,
        max_iter_static_amm: int = 50000,
        std_noise: float = 1e-2,
    ):
        super(DynamicAugmentedMarkovModelEstimator, self).__init__()

        # Initialize model parameters and experimental data
        self.msm = msm
        self.experiments = experiments
        self.eigfuncs = eigfuncs
        self.config = config

        # If config is None, use arguments instead
        if config is None:
            config = {}

        self.lag = config.get("lag", lag)
        self.dt_traj = config.get("dt_traj", dt_traj)
        self.alpha = torch.tensor(float(config.get("alpha", alpha)))
        self.beta = torch.tensor(float(config.get("beta", beta)))
        self.modify_n_eigvals = config.get("modify_n_eigvals", modify_n_eigvals)
        self.eps = config.get("eps", eps)
        self.static_amm_eps = config.get("static_amm_eps", static_amm_eps)
        self.omega_scale = config.get("omega_scale", omega_scale)
        self.dtype = config.get("dtype", dtype)
        self.max_iter_static_amm = config.get(
            "max_iter_static_amm", max_iter_static_amm
        )
        self.std_noise = config.get("std_noise", std_noise)
        self.pi_hat_converged = False

        # Separate dynamic and static experiments for convenience
        self.dynamic_experiments = [
            exp for exp in experiments if isinstance(exp, DynamicExperiment)
        ]
        self.static_experiments = [
            exp for exp in experiments if isinstance(exp, StaticExperiment)
        ]

        # Initialize observables and parameters based on the type of experiments provided
        self._initialize_observables_and_parameters()

        # Set the stationary distribution and related parameters
        self._initialize_stationary_distribution()

        # Initialize transition matrix and Lagrange multipliers
        self._initialize_transition_matrix_and_multipliers()

    def _initialize_observables_and_parameters(self):
        # This method initializes observables and parameters related to dynamic and static experiments.
        # It should set up tensors for experimental observables, predicted observables, and any masks or scaling factors.

        # Initialize MSM parameters.
        self.reigvecs, self.leigvecs = self.msm.reigvecs, self.msm.leigvecs
        self._eigvals = Parameter(self.msm.eigvals[: self.modify_n_eigvals])
        self.n_eigvals = len(self.msm.eigvals)
        self.n_states = self.msm.n_states

        # Initialize observables.
        if any(
            isinstance(experiment, StaticExperiment) for experiment in self.experiments
        ):
            self.static_observables_provided = True
        else:
            self.static_observables_provided = False

        self.dynamic_observables_by_state: Union[torch.Tensor, None] = None
        self.dynamic_observables_exp: Union[torch.Tensor, None] = None
        self.dynamic_observables_exp_mask: Union[torch.Tensor, None] = None
        self.dynamic_observables_pred: Union[torch.Tensor, None] = None
        self.dynamic_observables_pred_mask: Union[torch.Tensor, None] = None
        self.static_observables_by_state: Union[torch.Tensor, None] = None
        self.static_observables_exp: Union[torch.Tensor, None] = None
        self.static_observables_pred: Union[torch.Tensor, None] = None
        self._set_observable_attributes()
        self.n_obs = len(self.experiments)

        # Initialize regression parameters of observables.
        self.a = Parameter(
            torch.ones(self.dynamic_observables_pred.size()[0], dtype=self.dtype)
        )

        self.b = Parameter(
            torch.zeros(self.dynamic_observables_pred.size()[0], dtype=self.dtype)
        )

    def _initialize_stationary_distribution(self):
        # This method initializes the stationary distribution and related parameters.
        # It should handle different cases based on whether static observables are provided or not.
        if self.static_observables_provided:
            self.stationary_distribution = None
            self.static_amm = None
            self.omega = Parameter(torch.ones(self.leigvecs[:, 0].shape))
            self._initialize_static_amm()
        else:
            self.stationary_distribution = Parameter(self.leigvecs[:, 0])

        self.stationary_distribution_ori = self.leigvecs[:, 0]
        self._pi_hat = self.stationary_distribution_ori
        self.orthonormal_reigvecs = Parameter(
            self.reigvecs * torch.sqrt(self.stationary_distribution)[:, None]
        )

    def _initialize_transition_matrix_and_multipliers(self):
        # This method initializes the transition matrix and Lagrange multipliers.
        # It should set up the initial transition matrix and parameters for enforcing constraints.
        self.T_hat = self.msm.transition_matrix

        self.nu = Parameter(
            torch.ones_like(self.dynamic_observables_exp)
            + self.normal_noise(self.dynamic_observables_exp, std=self.std_noise)
        )  # acf regression for each lag time

        self.xi = Parameter(
            torch.ones(self.n_eigvals)
            + self.normal_noise(torch.zeros(self.n_eigvals), std=self.std_noise)
        )  # enforce eigvals < 1

        self.phi = Parameter(
            (
                torch.ones(self.n_states)
                + self.normal_noise(torch.zeros(self.n_states), std=self.std_noise)
            ).to(self.dtype)
        )  # enforce sum(stationary distribution) = 1

        self.tau = Parameter(
            torch.tensor(1.0)
        )  # enforce reversibility on sum of stationary_distribution

        self.chi = Parameter(
            torch.ones_like(self.T_hat) * 4
            + self.normal_noise(self.T_hat, std=self.std_noise)
        )  # enforce 0 <= p_ij <= 1 for all i, j

    def normal_noise(self, tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0):
        return torch.normal(mean=mean, std=std, size=tensor.size())

    def forward(self):
        # Calculate dynamic and static observables for the model.
        i = j = 0
        for experiment in self.experiments:
            if isinstance(experiment, DynamicExperiment):
                prediction = experiment(
                    self.leigvecs, self.eigvals, self.lag, dt_traj=self.dt_traj
                )
                mask = self.dynamic_observables_pred_mask[i]
                self.dynamic_observables_pred[i][mask] = prediction
                i += 1
            else:
                prediction = experiment(self.stationary_distribution)
                self.static_observables_pred[j] = prediction
                j += 1
            experiment.observables_pred = prediction

        # Set the updated attributes.
        self._set_observable_attributes(only_predictions=True)

    def loss(self, nll: torch.float, lloss: torch.float) -> torch.Tensor:
        """
        Actual loss function that combines all loss terms:
        NLL * sigma(alpha) + LagrLoss * sigma(1 - alpha) + L1_reg

        Parameters
        ----------
        nll:    float
                negative log-likelihood
        lloss:  float
                Lagrange loss
        l1loss: float
                L1-regularization term

        Returns
        -------
        loss:   float
                final loss

        """
        return nll * self.alpha.sigmoid() + lloss * (1 - self.alpha).sigmoid()

    def single_losses(self):
        """
        Returns NLL, Lagrange and L1 loss for monitoring

        Returns
        -------
        torch.Float
            Negative log-likelihood
        torch.Float
            Total loss term
        torch.Float
            Loss term for experimental data
        torch.Float
            Loss term for other constraints
        """
        if not self.pi_hat_converged:
            self.update_stationary_distribution()
            self.test_pi_hat_convergence()
        self._update_T_hat()
        nll = self.msm_nll()
        total_loss, l_exp, l_other = self.lagrange_loss()
        return nll, total_loss, l_exp, l_other

    def test_pi_hat_convergence(self):
        # if abs(static_observables_pred - static_observables_exp) < eps and sum(stationary_distribitution) = 1,
        # set pi_hat_converged = True

        if (
            torch.abs(self.static_observables_pred - self.static_observables_exp).mean()
            < self.static_amm_eps
        ):
            if torch.allclose(
                self.stationary_distribution.sum(), torch.ones(1, dtype=torch.float64)
            ):
                self.pi_hat_converged = True
                self.stationary_distribution = self.stationary_distribution.detach()
                self.omega = Parameter(self.omega.detach())
                print("~~~ *** ~~~ Static AMM converged ~~~ *** ~~~")

    def lagrange_loss(self):
        """
        Calculate the Lagrange loss for the estimator. This loss function is a combination of several terms:
        - Mean Squared Error (MSE) between predicted and experimental dynamic observables.
        - Penalties for eigenvalues greater than 1 to ensure stability.
        - Reversibility condition, ensuring detailed balance.
        - Stochasticity condition, ensuring that the transition probabilities are non-negative.
        - Constraint that the stationary distribution sums to 1.
        - Relative entropy between the current and original stationary distributions.

        Returns
        -------
        torch.Tensor
            Total Lagrange loss combining experimental and other loss terms.
        torch.Tensor
            Loss derived from experimental data.
        torch.Tensor
            Sum of all other loss terms (penalties and constraints).
        """
        # Calculate MSE between predicted and experimental dynamic observables
        loss_experimental, loss_other = self._calculate_experimental_loss()

        # Add penalties for eigenvalues greater than 1
        loss_other += self._enforce_eigenvalue_constraint()

        # Add reversibility condition loss
        loss_other += self._enforce_reversibility()

        # Enforce that the stationary distribution sums to 1
        loss_other += self._enforce_stationary_distribution()

        # Enforce transition probabilities are non-negative
        loss_other += self._enforce_transition_probabilities()

        # Add relative entropy term
        loss_other += self._calculate_relative_entropy()

        # Combine experimental and other loss terms
        loss = loss_experimental * self.beta + loss_other

        return loss, loss_experimental, loss_other

    def _calculate_experimental_loss(self):
        dynamic_observables_exp = torch.nan_to_num(self.dynamic_observables_exp)
        dynamic_observables_pred = torch.nan_to_num(self.dynamic_observables_pred)

        # Scale and mask the predicted dynamic observables
        dynamic_observable_scaled = (
            torch.abs(self.a[:, None]) * dynamic_observables_pred + self.b[:, None]
        )
        dynamic_observable_scaled *= self.dynamic_observables_pred_mask.to(int)

        # Calculate the error between scaled predictions and experimental data
        error_observables = (dynamic_observable_scaled - dynamic_observables_exp) ** 2

        # Apply sigmoid to nu and calculate the experimental loss
        loss_experimental = torch.sum(self.nu.sigmoid() * error_observables)

        # Add penalty for nu being too small
        penalty = self.nu.sigmoid() ** -1 - torch.ones_like(self.nu)
        penalty[penalty < self.eps] = 0
        loss_experimental += torch.sum(penalty)

        return loss_experimental, 0

    def _enforce_eigenvalue_constraint(self):
        # Penalty for eigenvalues greater than 1
        eigenvalue_penalty = torch.sum(
            torch.abs(self.xi)
            * (
                torch.maximum(torch.ones(self.n_eigvals), torch.abs(self.eigvals))
                - torch.ones(self.n_eigvals)
            )
        )
        return eigenvalue_penalty

    def _enforce_reversibility(self):
        # Penalty for violating reversibility condition
        reversibility_penalty = torch.sum(
            torch.abs(
                self.phi.matmul(
                    (
                        self.stationary_distribution[:, None] * self.T_hat
                        - (self.stationary_distribution[:, None] * self.T_hat).T
                    )
                )
            )
        )
        return reversibility_penalty

    def _enforce_stationary_distribution(self):
        # Penalty for stationary distribution not summing to 1
        stationary_distribution_penalty = self.tau * (
            self.stationary_distribution.sum() - 1
        )
        return stationary_distribution_penalty

    def _enforce_transition_probabilities(self):
        # Penalty for negative transition probabilities
        transition_probabilities_penalty = torch.sum(
            torch.abs(self.chi)
            * (
                torch.maximum(torch.ones_like(self.T_hat), self.T_hat)
                - torch.ones_like(self.T_hat)
            )
        )
        transition_probabilities_penalty += torch.sum(
            torch.abs(self.chi)
            * (-torch.minimum(torch.zeros_like(self.T_hat), self.T_hat))
        )

        # Add penalty for chi being too small
        penalty = self.chi.sigmoid() ** -1 - torch.ones_like(self.chi)
        penalty[penalty < self.eps] = 0
        transition_probabilities_penalty += torch.sum(penalty)

        return transition_probabilities_penalty

    def _calculate_relative_entropy(self):
        # Calculate the relative entropy term
        relative_entropy = torch.sum(
            self.stationary_distribution
            * torch.log(self.stationary_distribution / self.stationary_distribution_ori)
        )
        return relative_entropy

    def msm_nll(self):
        """
        Calculate negative log likelihood: sum(-C_ij * log(T_ij))

        Returns
        -------
        NLL:    float
                Negative log likelihood

        """
        count_matrix = self.msm.count_matrix
        transition_matrix = self.T_hat[self.msm.active_set, :][:, self.msm.active_set]
        return -torch.dot(
            count_matrix[transition_matrix > 0],
            torch.log(transition_matrix[transition_matrix > 0]),
        )

    def update_stationary_distribution(self):
        """
        Calulate stationary distribution (as per static AMM paper)

        Returns
        -------
        None


        """
        self.static_amm.state.m_hat = torch.squeeze(self.static_observables_pred)
        self.static_amm.fit(self.msm.count_matrix.numpy(), msm=self.msm)
        self.stationary_distribution = self.static_amm.state.pi_hat

    def _update_leigvecs(self):
        """
        Update left eigenvectors
        leigvecs = diag(pi) * reigvecs

        Returns
        -------
        None

        """
        self.leigvecs = torch.diag(
            self.stationary_distribution.clone().detach()
        ).matmul(self.reigvecs)

    def _update_T_hat(self):
        """
        Update transition matrix estimate, T_hat

        Returns
        -------
        None

        """
        updated_eigvals = torch.diag(torch.cat((torch.tensor([1.0]), self.eigvals)))
        T_hat = rdl_recomposition(self.reigvecs, updated_eigvals, self.leigvecs)
        self.T_hat = row_normalise(T_hat)

    def _update_reigvecs(self):
        """
        Update right eigenvectors (calling eigfuncs object)

        Returns
        -------
        None

        """
        self.eigfuncs(self)
        stationary_distribution = self.stationary_distribution.detach().clone()
        reigvecs = (
            self.orthonormal_reigvecs / torch.sqrt(stationary_distribution)[:, None]
        )
        self.reigvecs = torch.hstack([torch.ones(self.n_states, 1), reigvecs[:, 1:]])

    def project_eigfunc_grads(self):
        """
        Projects the eigenfunction gradients on the Stiefel manifold

        Returns
        -------
        None

        """
        self._update_reigvecs()
        self._update_leigvecs()

    def _set_observable_attributes(self, only_predictions: bool = False) -> None:
        """
        Sets/updates observable attributes (dynamic/static)

        Parameters
        ----------
        only_predictions:   bool, default: False
                            If set to True, only updates the prediction attributes

        Returns
        -------
        None

        """
        attr_list = [
            "observables_by_state",
            "observables_exp",
            "observables_pred",
        ]
        if only_predictions:
            attr_list = ["observables_pred"]
        names = ["dynamic", "static"]

        if self.static_observables_provided:
            experiment_types = [DynamicExperiment, StaticExperiment]
        else:
            experiment_types = [DynamicExperiment]

        for i, kind in enumerate(experiment_types):
            for j, attr in enumerate(attr_list):
                temp_list = []
                for experiment in self.experiments:
                    if isinstance(experiment, kind):
                        temp_list.append(getattr(experiment, attr))
                        # only set dynamic_experiments and static_experiments once
                        if not only_predictions and j == 0:
                            getattr(self, f"{names[i]}_experiments").append(experiment)
                if kind == DynamicExperiment and attr in [
                    "observables_exp",
                    "observables_pred",
                ]:
                    padded_tensor = pad_and_stack_tensors(temp_list, value=torch.nan)
                    mask = ~torch.isnan(padded_tensor)
                    setattr(self, f"{names[i]}_{attr}", padded_tensor)
                    setattr(self, f"{names[i]}_{attr}_mask", mask)
                else:
                    setattr(self, f"{names[i]}_{attr}", torch.vstack(temp_list))

            if not only_predictions:
                setattr(self, f"{names[i]}_observables_msm", None)

    def _initialize_static_amm(self):
        static_experimental_weights = (
            torch.ones_like(self.static_observables_exp) * self.omega_scale
        )

        self.static_amm = AugmentedMSMEstimator(
            self.static_observables_by_state.T,
            torch.squeeze(self.static_observables_exp),
            torch.squeeze(static_experimental_weights),
            eps=self.static_amm_eps,
            maxiter=self.max_iter_static_amm,
        )
        self.static_amm.fit(self.msm.count_matrix.numpy(), msm=self.msm)

        self.stationary_distribution = self.static_amm.state.pi_hat

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, new_a):
        self._set_property_with_shape_check(new_a, "a")

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, new_b):
        self._set_property_with_shape_check(new_b, "b")

    @property
    def stationary_distribution(self):
        return self._pi_hat

    @stationary_distribution.setter
    def stationary_distribution(self, new_pi_hat):
        self._pi_hat = new_pi_hat

    # Property 'eigvals' with conditional return
    @property
    def eigvals(self):
        return (
            torch.concat([self._eigvals, self.msm.eigvals[self.modify_n_eigvals :]])
            if self.modify_n_eigvals is not None
            else self._eigvals
        )

    @property
    def transition_matrix(self):
        transition_matrix = self.T_hat
        transition_matrix[transition_matrix < 0] = 0
        return row_normalise(transition_matrix)

    def _set_property_with_shape_check(self, new_value, attribute_name):
        old_value = getattr(self, "_" + attribute_name)
        if new_value.size() != old_value.size():
            raise ValueError(
                f"Shape mismatch. Parameter `{attribute_name}` should have shape {old_value.size()} but has shape {new_value.size()}."
            )
        setattr(self, "_" + attribute_name, new_value)
