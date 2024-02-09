import torch
from typing import *


class StiefelUpdate(torch.nn.Module):
    """
    A module for updating right eigenvectors while preserving matrix orthonormality constraints.
    Ensures that the eigenvectors stay on the Stiefel manifold during optimization.

    Attributes
    ----------
    step_size : float
        Step size or learning rate for the update.
    """

    def __init__(self, step_size: float):
        """
        Initializes the StiefelUpdate object with a specified step size.

        Parameters
        ----------
        step_size : float
            Step size or learning rate for the update.
        """
        super(StiefelUpdate, self).__init__()
        self.step_size = step_size

    def forward(self, self_dynAMMo: Any):
        """
        Forward pass of the module. Updates the gradients of the right eigenvectors.

        Parameters
        ----------
        self_dynAMMo : Any
            The instance of the main dynAMMo module containing the orthonormal right eigenvectors.
        """
        reigvecs = self_dynAMMo.orthonormal_reigvecs
        grad = reigvecs.grad

        if grad is not None:
            self.stiefel_update(reigvecs)

    def stiefel_update(self, reigvecs: torch.Tensor) -> None:
        """
        Applies the Stiefel manifold update to the right eigenvectors.

        Parameters
        ----------
        reigvecs : torch.Tensor
            Orthonormal right eigenvectors of the dynamical system.

        Returns
        -------
        None
        """
        n_states = reigvecs.size()[0]

        with torch.no_grad():
            # Preparing matrices for the Stiefel update
            u = torch.hstack([reigvecs.grad, reigvecs])
            v = torch.hstack([reigvecs, -reigvecs.grad])

            product_term = v.T.conj().matmul(u)
            inverse_term = (
                torch.eye(product_term.shape[0]) + (self.step_size / 2.0) * product_term
            )

            # Calculating the update
            a = self.step_size * (u.matmul(torch.linalg.inv(inverse_term)))
            b = v.T.conj().matmul(reigvecs)
            grads = a.matmul(b)

            # Applying the update to the gradients
            reigvecs.grad = torch.hstack([torch.zeros(n_states, 1), grads[:, 1:]])
