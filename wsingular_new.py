# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:30:25 2025

@author: bresch
"""

import torch
import numpy as np
from typing import Callable, Tuple
from wsingular import distance
from wsingular import utils

from torch import linalg as LA

from sklearn.metrics import silhouette_score

import ot
from tqdm import tqdm

def sinkhorn_singular_vectors_seq_norm(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    p: int = 1,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs sequential power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    # assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)
    
    print(torch.abs(R_A).max())
    print(torch.abs(R_B).max())

    D = R_A.clone()
    C = R_B.clone()

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    kk = 0
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.sinkhorn_map(
                A,
                C,
                R=R_A,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer!=None:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref,sink_seq_norm", loss_D[-1], k)
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "inf D,D_new,sink_seq_norm",
                    torch.abs(D - D_new/D_new.max()).max(), # / torch.abs(D_new).max(),
                    k
                )
                writer.add_scalar(
                        "Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                        #"AWS,D_new,sink_seq_norm", silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                        # k
                    )
                if label!=None:
                    writer.add_scalar(
                        #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                        "AWS,D_new,sink_seq_norm", silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                        k
                    )
                    
            # Normalize D
            D = D_new / D_new.max()

            # Compute C using D
            C_new = distance.sinkhorn_map(
                B,
                D,
                R=R_B,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref,sink_seq_norm", loss_C[-1], k)
                writer.add_scalar(
                    #"Hilbert C,C_new,sink_seq_norm", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,sink_seq_norm",
                    torch.abs(C - C_new/C_new.max()).max(), # / torch.abs(C_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert C,C_new,sink_seq_norm", utils.hilbert_distance(C, C_new), k
                    # "inf C,C_new,sink_seq_norm",
                    # torch.abs(C - C_new).max() / torch.abs(C_new).max(),
                    # k
                )

            # Normalize C
            C = C_new / C_new.max()

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D
    
def stochastic_sinkhorn_singular_vectors_norm(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    sample_prop: float = 1e-1,
    p: int = 1,
    step_fn: Callable = lambda k: 1 / np.sqrt(k),
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float, optional): The entropic regularization parameter. Defaults to 5e-2.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop <= 1  # a valid proportion
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize the singular vectors as the regularization matrices.
    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    D = R_A.clone()
    C = R_B.clone()

    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    mu, lbda = 1, 1

    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Set the decreasing step size.
            step_size = step_fn(k)

            # Log the step size if there is a writer.
            # if writer:
            #     writer.add_scalar("step_size", step_size, k)

            # Update a random subset of the indices.
            C_new, xx, yy = stochastic_sinkhorn_map(
                B,
                C,
                D,
                R=R_B,
                gamma=1,
                sample_prop=sample_prop,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the approximation of the singular value lambda.
            lbda = (1 - step_size) * lbda + step_size * torch.sum(
                C_new[xx, yy] * C[xx, yy]
            ) / torch.sum(C[xx, yy] ** 2)

            # Log the approximation of lambda if there is a writer.
            # if writer:
            #     writer.add_scalar("lambda", lbda, k)

            # Rescale the updated indices by the approximation of the singular value.
            C_new[xx, yy] /= lbda

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp((1 - step_size) * C.log() + step_size * C_new.log())
            else:
                C_new = (1 - step_size) * C + step_size * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:
                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref", hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert C,C_new,sink_seq_norm", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,stoch_sink_norm_%f" %(sample_prop),
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )

            # Rescale the singular vector.
            C = C_new / C_new.max()

            # Update a random subset of the indices.
            D_new, xx, yy = stochastic_sinkhorn_map(
                A,
                D,
                C,
                R=R_A,
                gamma=1,
                sample_prop=sample_prop,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the approximation of the singular value mu.
            mu = (1 - step_size) * mu + step_size * torch.sum(
                D_new[xx, yy] * D[xx, yy]
            ) / torch.sum(D[xx, yy] ** 2)

            # Log the approximation of mu if there is a writer.
            # if writer:
            #     writer.add_scalar("mu", mu, k)

            # Rescale the updated indices by the approximation of the singular value mu.
            D_new[xx, yy] /= mu

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp((1 - step_size) * D.log() + step_size * D_new.log())
            else:
                D_new = (1 - step_size) * D + step_size * D_new

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref", hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert C,C_new,sink_seq_norm", utils.hilbert_distance(C, C_new), k
                    "inf D,D_new,stoch_sink_norm_%f" %(sample_prop),
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,stoch_sink_norm_%f" %(sample_prop), silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                    k
                )
            # Rescale the singular vector.
            D = D_new / D_new.max()

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    return C, D

    
def sinkhorn_singular_vectors_par_norm(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    p: int = 1,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs parallel power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.sinkhorn_map(
                A,
                C,
                R=R_A,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref,sink_par_norm", loss_D[-1], k)
                writer.add_scalar(
                    # "Hilbert D,D_new,sink_par_norm", utils.hilbert_distance(D, D_new), k
                    "inf D,D_new,sink_par_norm",
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,sink_par_norm", silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                    k
                )

            # Compute C using D
            C_new = distance.sinkhorn_map(
                B,
                D,
                R=R_B,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref,sink_par_norm", loss_C[-1], k)
                writer.add_scalar(
                    # "Hilbert C,C_new,sink_par_norm", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,sink_par_norm",
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )

            # Normalize
            D = D_new / D_new.max()
            C = C_new / C_new.max()

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            C /= C.max()
            D /= D.max()
            break

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D
    
def sinkhorn_singular_vectors_seq_step(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs sequential gradient iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.sinkhorn_map(
                A,
                C,
                R=R_A,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref,sink_seq_step", loss_D[-1], k)
                writer.add_scalar(
                    # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                    "inf D,D_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    torch.abs(D - D_new).max(), # / torch.abs(D_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert D,D_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), utils.hilbert_distance(D, D_new), k
                    # "inf D,D_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    # torch.abs(D - D_new).max() / torch.abs(D_new).max(),
                    # k
                )
                if writer!=None and label!=None:
                    writer.add_scalar(
                        #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                        "AWS,D_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                        k
                    )

            # Update D
            D = gamma_D * D_new 

            # Compute C using D
            C_new = distance.sinkhorn_map(
                B,
                D,
                R=R_B,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref,sink_seq_step", loss_C[-1], k)
                writer.add_scalar(
                    # "Hilbert C,C_new,sink_seq_step", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    torch.abs(C - C_new).max(), # / torch.abs(C_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert C,C_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), utils.hilbert_distance(C, C_new), k
                    # "inf C,C_new,sink_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    # torch.abs(C - C_new).max() / torch.abs(C_new).max(),
                    # k
                )

            # Update C
            C = (1 - alpha) * C + alpha * gamma_C * C_new

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            # C /= C.max()
            # D /= D.max()
            break

    # C /= C.max()
    # D /= D.max()

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D

def wasserstein_singular_vectors_seq_norm(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    p: int = 1,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs sequential gradient iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.wasserstein_map(
                A,
                C,
                R=R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref,sink_seq_norm", loss_D[-1], k)
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                    "inf D,D_new,wass_seq_norm",
                    torch.abs(D - D_new/D_new.max()).max(), # / torch.abs(D_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert D,D_new,wass_seq_norm", utils.hilbert_distance(D, D_new), k
                    #"inf D,D_new,wass_seq_norm",
                    #torch.abs(D - D_new).max() / torch.abs(D_new).max(),
                    #k
                )
                if label:
                    writer.add_scalar(
                        #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                        "AWS,D_new,wass_seq_norm", silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                        k
                    )

            # Normalize D
            D = D_new / D_new.max()

            # Compute C using D
            C_new = distance.wasserstein_map(
                B,
                D,
                R=R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref,wass_seq_norm", loss_C[-1], k)
                writer.add_scalar(
                    #"Hilbert C,C_new,sink_seq_step", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,wass_seq_norm",
                    torch.abs(C - C_new/C_new.max()).max(), # / torch.abs(C_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert C,C_new,wass_seq_norm", utils.hilbert_distance(C, C_new), k
                    #"inf C,C_new,wass_seq_norm",
                    #torch.abs(C - C_new).max() / torch.abs(C_new).max(),
                    #k
                )

            # Normalize C
            C = C_new / C_new.max()

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            # C /= C.max()
            # D /= D.max()
            break

    # C /= C.max()
    # D /= D.max()

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D
    
def wasserstein_singular_vectors_seq_step(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs sequential gradient iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.wasserstein_map(
                A,
                C,
                R=R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref,sink_seq_step", loss_D[-1], k)
                writer.add_scalar(
                    # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                    "inf D,D_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    torch.abs(D - D_new).max(), # / torch.abs(D_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert D,D_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), utils.hilbert_distance(D, D_new), k
                    #"inf D,D_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    #torch.abs(D - D_new).max() / torch.abs(D_new).max(),
                    #k
                )
                if label:
                    writer.add_scalar(
                        #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                        "AWS,D_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                        k
                    )

            # Update D
            D = gamma_D * D_new 

            # Compute C using D
            C_new = distance.wasserstein_map(
                B,
                D,
                R=R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref,wass_seq_step", loss_C[-1], k)
                writer.add_scalar(
                    # "Hilbert C,C_new,sink_seq_step", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    torch.abs(C - C_new).max(), # / torch.abs(C_new).max(),
                    k
                )
                writer.add_scalar(
                    "Hilbert C,C_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), utils.hilbert_distance(C, C_new), k
                    #"inf C,C_new,wass_seq_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    #torch.abs(C - C_new).max() / torch.abs(C_new).max(),
                    #k
                )

            # Update C
            C = (1 - alpha) * C + alpha * gamma_C * C_new

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            # C /= C.max()
            # D /= D.max()
            break

    # C /= C.max()
    # D /= D.max()

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D
       
def sinkhorn_singular_vectors_par_step(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: str,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    log_loss: bool = False,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs parallel gradient iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset. Alternatively, you can give a tuple of tensors (A, B).
        dtype (str): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float): The entropic regularization parameter.
        p (int, optional): The order of the norm R. Defaults to 1.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        log_loss (bool, optional): Whether to return the loss. Defaults to False.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D). If `log_loss`, it returns (C, D, loss_C, loss_D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    D = R_A.clone()
    C = R_B.clone()

    # Initialize loss history.
    loss_C, loss_D = [], []

    # Iterate until `n_iter`.
    for k in range(n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Compute D using C
            D_new = distance.sinkhorn_map(
                A,
                C,
                R=R_A,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(D_ref):
                    loss_D.append(utils.hilbert_distance(D, D_ref))
                    writer.add_scalar("Hilbert D,D_ref,sink_par_step", loss_D[-1], k)
                writer.add_scalar(
                    # "Hilbert D,D_new,sink_par_step", utils.hilbert_distance(D, D_new), k
                    "inf D,D_new,sink_par_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,sink_par_step_%f_%f_%f" %(alpha, gamma_C, gamma_D), silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                    k
                )
             

            # Compute C using D
            C_new = distance.sinkhorn_map(
                B,
                D,
                R=R_B,
                tau=tau,
                eps=eps,
                dtype=dtype,
                device=device,
                progress_bar=progress_bar,
            )

            # Compute Hilbert loss
            if writer:
                if torch.is_tensor(C_ref):
                    loss_C.append(utils.hilbert_distance(C, C_ref))
                    writer.add_scalar("Hilbert C,C_ref,sink_par_step", loss_C[-1], k)
                writer.add_scalar(
                    # "Hilbert C,C_new,sink_par_step", utils.hilbert_distance(C, C_new), k
                    "inf C,C_new,sink_par_step_%f_%f_%f" %(alpha, gamma_C, gamma_D),
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )

            # Update D
            D = (1 - alpha) * D + alpha * gamma_D * D_new
            # Update C
            C = (1 - alpha) * C + alpha * gamma_C * C_new

            # TODO: Try early stopping.

        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            # C /= C.max()
            # D /= D.max()
            break

    # C /= C.max()
    # D /= D.max()

    if log_loss:
        return C, D, loss_C, loss_D
    else:
        return C, D    


def stochastic_wasserstein_singular_vectors_seq(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    sample_prop_C: float = 1e-1,
    sample_prop_D: float = 1e-1,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Wasserstein Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Wasserstein Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop_C <= 1  # a valid proportion
    assert 0 < sample_prop_D <= 1  # a valid proportion
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize the singular vectors as the regularization matrices.
    D = R_A.clone()
    C = R_B.clone()

    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:
            # Update a random subset of the indices.
            D_new, xx, yy = distance.stochastic_wasserstein_map(
                A,
                D,
                C,
                gamma=1,
                sample_prop=sample_prop_D,
                R=R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp(gamma_D * D_new.log())
            else:
                D_new = gamma_D * D_new 

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref,stoch_wasserstein_seq_%f" %(sample_prop_D), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert D,D_new,stoch_wasserstein_seq_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "inf D,D_new,stoch_wasserstein_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,stoch_wasserstein_seq_%f" %(sample_prop_D), silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert D,D_new,stoch_wasserstein_seq_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "2 D,D_new,stoch_wasserstein_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=2),
                    k
                )'''
                
            D = D_new

            # Update a random subset of the indices.
            C_new, xx, yy = distance.stochastic_wasserstein_map(
                B,
                C,
                D,
                gamma=1,
                sample_prop=sample_prop_C,
                R=R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp((1 - alpha) * C.log() + alpha * gamma_C * C_new.log())
            else:
                C_new = (1 - alpha) * C + alpha * gamma_C * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)
            
            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref,stoch_wasserstein_seq_%f" %(sample_prop_C), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert C,C_new,stoch_wasserstein_seq_%f" %(sample_prop_C), 
                    #utils.hilbert_distance(C, C_new), 
                    #k
                    "inf C,C_new,stoch_wasserstein_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert C,C_new,stoch_wasserstein_seq_%f" %(sample_prop_C), 
                    #utils.hilbert_distance(C, C_new), 
                    #k
                    "2 C,C_new,stoch_wasserstein_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=2),
                    k
                )'''

            # Rescale the singular vector.
            C = C_new 

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            break

    # Return the singular vectors.
    
    return C, D

def stochastic_wasserstein_singular_vectors_par(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    sample_prop_C: float = 1e-1,
    sample_prop_D: float = 1e-1,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Wasserstein Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Wasserstein Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop_C <= 1  # a valid proportion
    assert 0 < sample_prop_D <= 1  # a valid proportion
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize the singular vectors as the regularization matrices.
    D = R_A.clone()
    C = R_B.clone()

    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:
            # Update a random subset of the indices.
            D_new, xx, yy = distance.stochastic_wasserstein_map(
                A,
                D,
                C,
                gamma=1,
                sample_prop=sample_prop_D,
                R=R_A,
                tau=tau,
                dtype=dtype,
                device=device,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp((1 - alpha) * D.log() + alpha * gamma_D * D_new.log() )
            else:
                D_new = (1 - alpha) * D + alpha * gamma_D * D_new 

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)

            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref,stoch_wasserstein_par_%f" %(sample_prop_D), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert D,D_new,stoch_wasserstein_par_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "inf D,D_new,stoch_wasserstein_par_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,stoch_wasserstein_par_%f" %(sample_prop_D), silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed'),
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert D,D_new,stoch_wasserstein_par_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "2 D,D_new,stoch_wasserstein_par_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=2),
                    k
                )'''


            # Update a random subset of the indices.
            C_new, xx, yy = distance.stochastic_wasserstein_map(
                B,
                C,
                D,
                gamma=1,
                sample_prop=sample_prop_C,
                R=R_B,
                tau=tau,
                dtype=dtype,
                device=device,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp((1 - alpha) * C.log() + alpha * gamma_C * C_new.log())
            else:
                C_new = (1 - alpha) * C + alpha * gamma_C * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)
            
            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref,stoch_wasserstein_par_%f" %(sample_prop_C), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert C,C_new,stoch_wasserstein_par_%f" %(sample_prop_C), 
                    #utils.hilbert_distance(C, C_new),
                    #k
                    "inf C,C_new,stoch_wasserstein_par_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert C,C_new,stoch_wasserstein_par_%f" %(sample_prop_C), 
                    #utils.hilbert_distance(C, C_new),
                    #k
                    "2 C,C_new,stoch_wasserstein_par_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=2),
                    k
                )'''

            # Rescale the singular vector.
            D = D_new
            C = C_new 

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            break

    # Return the singular vectors.
    
    return C, D


def stochastic_sinkhorn_singular_vectors_seq(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    sample_prop_C: float = 1e-1,
    sample_prop_D: float = 1e-1,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float, optional): The entropic regularization parameter. Defaults to 5e-2.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop_C <= 1  # a valid proportion
    assert 0 < sample_prop_D <= 1  # a valid proportion
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize the singular vectors as the regularization matrices.
    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    D = R_A.clone()
    C = R_B.clone()

    asw_old = 0.0


    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Update a random subset of the indices.
            D_new, xx, yy = stochastic_sinkhorn_map(
                A,
                D,
                C,
                R=R_A,
                gamma=1,
                sample_prop=sample_prop_D,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp(gamma_D * D_new.log())
            else:
                D_new = gamma_D * D_new

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)
            
            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref,stoch_sink_seq_%f" %(sample_prop_D), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert D,D_new,stoch_sink_seq_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "inf D,D_new,stoch_sink_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                asw_new = silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed') 
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,stoch_sink_seq_%f" %(sample_prop_D), asw_new,
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert D,D_new,stoch_sink_seq_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "2 D,D_new,stoch_sink_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=2),
                    k
                )'''
                
            if asw_old < asw_new:
                Dbest = D_new
                # asw_old = asw_new

            D = D_new

            # Update a random subset of the indices.
            C_new, xx, yy = stochastic_sinkhorn_map(
                B,
                C,
                D,
                R=R_B,
                gamma=1,
                sample_prop=sample_prop_C,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

        
            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp( (1 - alpha) * C.log() + alpha * gamma_C * C_new.log())
            else:
                C_new = (1 - alpha) * C + alpha * gamma_C * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)

            
            
            # If we have a writer, compute some losses.
            if writer:
                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref,stoch_sink_seq_%f" %(sample_prop_C), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert C,C_new,stoch_sink_seq_%f" %(sample_prop_C),
                    #utils.hilbert_distance(C, C_new),
                    #k,
                    "inf C,C_new,stoch_sink_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert C,C_new,stoch_sink_seq_%f" %(sample_prop_C),
                    #utils.hilbert_distance(C, C_new),
                    #k,
                    "2 C,C_new,stoch_sink_seq_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=2),
                    k
                )'''

            if asw_old < asw_new:
                Cbest = C_new
                asw_old = asw_new

            C = C_new

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            
            break

    # Projction onto boundary of infinity ball 
        
    return Cbest, Dbest


def stochastic_sinkhorn_singular_vectors_par(
    dataset: torch.Tensor,
    label: torch.Tensor,
    dtype: torch.dtype,
    device: str,
    n_iter: int,
    tau: float = 0,
    eps: float = 5e-2,
    sample_prop_C: float = 1e-1,
    sample_prop_D: float = 1e-1,
    p: int = 1,
    alpha: float = 0.9,
    gamma_C: float = 0.9,
    gamma_D: float = 0.9,
    mult_update: bool = False,
    writer=None,
    small_value: float = 1e-6,
    normalization_steps: int = 1,
    C_ref: torch.Tensor = None,
    D_ref: torch.Tensor = None,
    progress_bar: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Performs stochastic power iterations and returns Sinkhorn Singular Vectors. Early stopping is possible with Ctrl-C.

    Args:
        dataset (torch.Tensor): The input dataset.  Alternatively, you can give a tuple of tensors (A, B).
        dtype (torch.dtype): The dtype
        device (str): The device
        n_iter (int): The number of power iterations.
        tau (float, optional): The regularization parameter for the norm R. Defaults to 0.
        eps (float, optional): The entropic regularization parameter. Defaults to 5e-2.
        sample_prop (float, optional): The proportion of indices to update at each step. Defaults to 1e-1.
        p (int, optional): The order of the norm R. Defaults to 1.
        step_fn (Callable, optional): The function that defines step size from the iteration number (which starts at 1). Defaults to lambdak:1/np.sqrt(k).
        mult_update (bool, optional): If True, use multiplicative update instead of additive update. Defaults to False.
        writer (SummaryWriter, optional): If set, the progress will be written to the Tensorboard writer. Defaults to None.
        small_value (float, optional): A small value for numerical stability. Defaults to 1e-6.
        normalization_steps (int, optional): How many Sinkhorn iterations for the initial normalization of the dataset. Must be > 0. Defaults to 1, which is just regular normalization, along columns for A and along rows for B. For large numbers of steps, A and B are bistochastic.
        C_ref (torch.tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        D_ref (torch.Tensor, optional): If set, Hilbert distances to this reference will be computed. Defaults to None.
        progress_bar (bool, optional): Whether to display a progress bar for individual matrix computations. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Sinkhorn Singular Vectors (C,D)
    """

    # Perform some sanity checks.
    assert n_iter > 0  # at least one iteration
    assert tau >= 0  # a positive regularization
    assert 0 < sample_prop_C <= 1  # a valid proportion
    assert 0 < sample_prop_D <= 1  # a valid proportion
    assert eps >= 0  # a positive entropic regularization
    assert p > 0  # a valid norm
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        A, B = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )

    # Compute the regularization matrices.
    R_A = utils.regularization_matrix(A, p=p, dtype=dtype, device=device)
    R_B = utils.regularization_matrix(B, p=p, dtype=dtype, device=device)

    # Initialize the singular vectors as the regularization matrices.
    # Initialize the approximations of the singular values. Rescaling at each iteration
    # by the (approximated) singular value make convergence super duper fast.
    D = R_A.clone()
    C = R_B.clone()

    asw_old = 0.0


    # Iterate until `n_iter`.
    for k in range(1, n_iter):

        # Try, but expect a KeyboardInterrupt in case of early stopping.
        try:

            # Update a random subset of the indices.
            D_new, xx, yy = stochastic_sinkhorn_map(
                A,
                D,
                C,
                R=R_A,
                gamma=1,
                sample_prop=sample_prop_D,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                D_new = torch.exp((1 - alpha) * D.log() + alpha * gamma_D * D_new.log())
            else:
                D_new = (1 - alpha) * D + alpha * gamma_D * D_new

            # Make sure the diagonal of D is 0.
            D_new.fill_diagonal_(0)

            
            
            # If we have a writer, compute some losses.
            if writer:

                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(D_ref):
                    hilbert = utils.hilbert_distance(D_new, D_ref)
                    writer.add_scalar("Hilbert D,D_ref,stoch_sink_par_%f" %(sample_prop_D), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert D,D_new,stoch_sink_par_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "inf D,D_new,stoch_sink_par_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=torch.inf),
                    k
                )
                asw_new = silhouette_score(D_new.cpu(), label.cpu(), metric = 'precomputed')
                writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,D_new,stoch_sink_par_%f" %(sample_prop_D), asw_new,
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert D,D_new,stoch_sink_par_%f" %(sample_prop_D),
                    #utils.hilbert_distance(D, D_new),
                    #k,
                    "2 D,D_new,stoch_sink_par_%f" %(sample_prop_D),
                    LA.matrix_norm(D - D_new, ord=2),
                    k
                )'''

            # Update a random subset of the indices.
            C_new, xx, yy = stochastic_sinkhorn_map(
                B,
                C,
                D,
                R=R_B,
                gamma=1,
                sample_prop=sample_prop_C,
                tau=tau,
                eps=eps,
                return_indices=True,
                progress_bar=progress_bar,
            )

        
            # Update the singular vector, either multiplicatively or additively.
            if mult_update:
                C_new = torch.exp( (1 - alpha) * C.log() + alpha * gamma_C * C_new.log())
            else:
                C_new = (1 - alpha) * C + alpha * gamma_C * C_new

            # Make sure the diagonal of C is 0.
            C_new.fill_diagonal_(0)
            
            # If we have a writer, compute some losses.
            if writer:
                # If we have a reference, compute Hilbert distance to ref.
                if torch.is_tensor(C_ref):
                    hilbert = utils.hilbert_distance(C_new, C_ref)
                    writer.add_scalar("Hilbert C,C_ref,stoch_sink_par_%f" %(sample_prop_C), hilbert, k)

                # Compute the Hilbert distance to the value of C at the previous step.
                writer.add_scalar(
                    #"Hilbert C,C_new,stoch_sink_par_%f" %(sample_prop_C),
                    #utils.hilbert_distance(C, C_new),
                    #k,
                    "inf C,C_new,stoch_sink_par_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=torch.inf),
                    k
                )
                '''writer.add_scalar(
                    #"Hilbert C,C_new,stoch_sink_par_%f" %(sample_prop_C),
                    #utils.hilbert_distance(C, C_new),
                    #k,
                    "2 C,C_new,stoch_sink_par_%f" %(sample_prop_D),
                    LA.matrix_norm(C - C_new, ord=2),
                    k
                )'''

            if asw_old < asw_new:
                Cbest = C_new
                Dbest = D_new
                asw_old = asw_new

            D = D_new
            C = C_new

        # In case of Ctrl-C, make sure C and D are rescaled properly
        except KeyboardInterrupt:
            print("Stopping early after keyboard interrupt!")
            
            break

    # Projction onto boundary of infinity ball 
        
    return Cbest, Dbest


def stochastic_sinkhorn_map(
    A: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    sample_prop: float,
    gamma: float,
    eps: float,
    R: torch.Tensor = None,
    tau: float = 0,
    progress_bar: bool = False,
    return_indices: bool = False,
    batch_size: int = 50,
    stop_threshold: float = 1e-5,
    num_iter_max: int = 100,
) -> torch.Tensor:
    """Returns the stochastic Sinkhorn divergence map, updating only a random
    subset of indices and leaving the other ones as they are.

    Args:
        A (torch.Tensor): The input dataset.
        D (torch.Tensor): The intialization of the distance matrix
        C (torch.Tensor): The ground cost
        sample_prop (float): The proportion of indices to update
        gamma (float): Rescaling parameter. In practice, one should rescale by an approximation of the singular value.
        eps (float): The entropic regularization parameter
        R (torch.Tensor): The regularization matrix. Defaults to None.
        tau (float): The regularization parameter. Defaults to 0.
        progress_bar (bool): Whether to show a progress bar during the computation. Defaults to False.
        return_indices (bool): Whether to return the updated indices. Defaults to False.
        batch_size (int): Batch size, i.e. how many distances to compute at the same time. Depends on your available GPU memory. Defaults to 50.

    Returns:
        torch.Tensor: The stochastically updated distance matrix.
    """

    # Perform some sanity checks.
    assert tau >= 0 # a positive regularization
    assert 0 < sample_prop <= 1 # a valid proportion
    assert eps >= 0 # a positive entropic regularization

    # Name the dimensions of the dataset (samples x features).
    n_samples, n_features = A.shape

    # Define the sample size from the proportion.
    sampling_size = max(2, int(np.sqrt(sample_prop) * n_samples))

    # Random indices.
    idx = np.random.choice(range(n_samples), size=sampling_size, replace=False)

    # Initialize new distance
    D_new = D.clone()

    # Compute the kernel.
    K = (-C / eps).exp()

    # Initialize the progress bar if we want one.
    if progress_bar:
        pbar = tqdm(total=sampling_size * (sampling_size - 1) // 2, leave=False)

    # Iterate over random indices.
    for k in range(sampling_size):

        i = idx[k]

        for ii in np.array_split(idx[: k + 1], max(1, k // batch_size)):

            # Compute the Sinkhorn dual variables.
            _, wass_log = ot.sinkhorn(
                A[i].contiguous(),  # This is the source histogram.
                A[ii].T.contiguous(),  # These are the target histograms.
                C,  # This is the ground cost.
                eps,  # This is the entropic regularization parameter.
                log=True,  # Return the dual variables.
                stopThr=stop_threshold,
                numItermax=num_iter_max,
            )

            # Compute the exponential dual variables.
            f, g = eps * wass_log["u"].log(), eps * wass_log["v"].log()

            # Compute the Sinkhorn costs.
            # These will be used to compute the Sinkhorn divergences below.
            wass = (
                f * A[[i] * len(ii)].T
                + g * A[ii].T
                - eps * wass_log["u"] * (K @ wass_log["v"])
            ).sum(0)

            # Add them in the distance matrix (including symmetric values).
            D_new[i, ii] = D_new[ii, i] = wass

            # Update the progress bar if we have one.
            if progress_bar:
                pbar.update(len(ii))

    # Close the progress bar if we have one.
    if progress_bar:
        pbar.close()

    # Get the indices for the grid (idx,idx).
    xx, yy = np.meshgrid(idx, idx)

    # Get the diagonal terms OT_eps(a, a)
    d = torch.diagonal(D_new[xx, yy])

    # Sinkhorn divergence OT(a, b) - (OT(a, a) + OT(b, b))/2
    D_new[xx, yy] = D_new[xx, yy] - 0.5 * (d.view(-1, 1) + d.view(1, -1))

    # Make sure there are no negative values.
    D_new[D_new < 0] = 0

    # Make sure the diagonal is zero.
    D_new[xx, yy].fill_diagonal_(0)

    # If the regularization parameter is > 0, regularize.
    if tau > 0:
        D_new[xx, yy] += tau * R[xx, yy]

    # Divide gamma
    D_new[xx, yy] /= gamma

    # Return the distance matrix.
    if return_indices:
        return D_new, xx, yy
    else:
        return D_new

def normalize_dataset(
    dataset: torch.Tensor,
    dtype: str,
    device: str,
    normalization_steps: int = 1,
    small_value: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize the dataset and return the normalized dataset A and the transposed dataset B.

    Args:
        dataset (torch.Tensor): The input dataset, samples as rows.
        normalization_steps (int, optional): The number of Sinkhorn normalization steps. For large numbers, we get bistochastic matrices. Defaults to 1 and should be larger or equal to 1.
        small_value (float): Small addition to the dataset to avoid numerical errors while computing OT distances. Defaults to 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The normalized matrices A and B.
    """

    # Perform some sanity checks.
    assert len(dataset.shape) == 2  # correct shape
    assert torch.sum(dataset < 0) == 0  # positivity
    assert small_value > 0  # a positive numerical offset
    assert normalization_steps > 0  # normalizing at least once

    '''# Do a first normalization pass for A
    A = dataset / dataset.sum(1).reshape(-1, 1)
    A += small_value
    A /= A.sum(1).reshape(-1, 1)

    # Do a first normalization pass for B
    B = dataset.T / dataset.T.sum(1).reshape(-1, 1)
    B += small_value
    B /= B.sum(1).reshape(-1, 1)

    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        A, B = B.T / B.T.sum(1).reshape(-1, 1), A.T / A.T.sum(1).reshape(-1, 1)'''
    
    n,m = dataset.shape
    A = dataset / dataset.sum(1).reshape(-1, 1)
    A += small_value
    D1 = torch.ones(n, dtype=dtype)
    D2 = torch.ones(m, dtype=dtype)
    # Make any additional normalization steps.
    for _ in range(normalization_steps - 1):
        D1 = torch.ones(n) / (A @ D2)
        D2 = torch.ones(m) / (A.T @ D1)
    A = torch.diag(D1) @ A @ torch.diag(D2)
    B = A.T

    return A.to(dtype=dtype, device=device), B.to(dtype=dtype, device=device)
