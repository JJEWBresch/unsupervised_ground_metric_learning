import numpy as np
import torch
import wsingular
from wsingular import utils
import wsingular_new
from wsingular_new import normalize_dataset
import datetime
import time
import pandas as pd
import scanpy as sc

from keras.datasets import mnist
from torch.utils.tensorboard import SummaryWriter

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from time import sleep
from tqdm import tqdm

####
# 
# Code 
#
####

# calculate tau s.t. Psi is definitely contractive.
def calculate_tau_gauss(data: torch.Tensor, lipschitz_const: torch.double):
    n = len(data)
    norms1 = torch.tensor([torch.linalg.norm(data[i] - data[j], ord=1) for i in range(n) for j in range(i, n)])

    return 2 * lipschitz_const * norms1.max() ** 2

def calculate_tau_laplace(data: torch.Tensor, lipschitz_const: torch.double):
    n = len(data)
    norms1 = torch.tensor([torch.linalg.norm(data[i] - data[j], ord=1) for i in range(n) for j in range(i, n)])
    norms2 = torch.tensor([torch.linalg.norm(data[i] - data[j], ord=2) for i in range(n) for j in range(i+1, n)])

    return 2 * lipschitz_const * norms1.max() ** 4 / (norms2.min() ** 2)

def calculate_tau_inv_mult_quad(data: torch.Tensor, lipschitz_const: torch.double):
    n = len(data)
    norms1 = torch.tensor([torch.linalg.norm(data[i] - data[j], ord=1) for i in range(n) for j in range(i, n)])

    return 2 * lipschitz_const * norms1.max() ** 2

# technically rbf_kernel(sqrt())
def rbf_kernel(x: torch.Tensor, y: torch.Tensor, mat: torch.Tensor, sigma: torch.double):
    z = x - y
    dotprod = z.T @ mat @ z
    # dotprod = torch.matmul(torch.matmul(mat, z).T, z)
    return torch.exp(- dotprod / (2 * sigma ** 2))

def laplace_kernel(x: torch.Tensor, y: torch.Tensor, mat: torch.Tensor, sigma: torch.double):
    z = x - y
    dotprod = z.T @ mat @ z
    # dotprod = torch.matmul(torch.matmul(mat, z).T, z)
    return torch.exp(- torch.sqrt(dotprod) / sigma)

def sinc_kernel(x: torch.Tensor, y: torch.Tensor, mat: torch.Tensor, p: torch.int):
    z = x - y
    dotprod = z.T @ mat @ z
    return torch.sinc(dotprod)**p

def inv_mult_quad(x: torch.Tensor, y: torch.Tensor, mat: torch.Tensor, sigma: torch.double):
    z = x - y
    dotprod = z.T @ mat @ z
    # dotprod = torch.matmul(torch.matmul(mat, z).T, z)
    return torch.sqrt(1 - sigma**2 * dotprod**2)

    
def Psi(func_kernel, data: torch.Tensor, learnedmat: torch.Tensor, dtype: str, tau: float=0, norm: int=1):
    n = data.shape[0]
    newlearned = torch.zeros((n, n), dtype=dtype)

    for i in range(n):
        for j in range(i, n):
            newlearned[i,j] = newlearned[j,i] = func_kernel(data[i], data[j], learnedmat) + tau * torch.abs(learnedmat).max() * int(i == j) 

    if norm==1:
        return newlearned / torch.abs(newlearned).max()
    if norm==0:
        return newlearned

# Power Iteration Algorithm
def unsupervised_gauss_kernel_learning_norm(
        dataset: torch.Tensor, 
        dtype: str,
        device: str,
        label: torch.Tensor,
        rbf_sigma_A: float,
        rbf_sigma_B: float,
        func_kernel_A=None, 
        func_kernel_B=None, 
        writer=None,
        iter_max: int = 1000, 
        regularize=False, 
        regularization_param=None, 
        relative_residual: float = 1e-3, 
        normalization_steps: int=1,
        small_value: float=1e-6,
        lipschitz_of_kernel_A=None, 
        lipschitz_of_kernel_B=None, 
        B_0=None):

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        data, dataT = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )
    if label:
        label = torch.tensor(label, dtype=dtype)

    # If Lipschitzconstants are given calculate tau.
    tau_A = calculate_tau_gauss(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    tau_B = calculate_tau_gauss(dataT, lipschitz_of_kernel_B) + regularization_param if regularize else 0
    # tau_A = calculate_tau_inv_mult_quad(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    # tau_B = calculate_tau_inv_mult_quad(data.T, lipschitz_of_kernel_B) + regularization_param if regularize else 0


    # assume pairwise distance of every point is 1
    # B = (torch.ones((data.T.shape[1], data.T.shape[1]), dtype=dtype) - torch.eye(data.T.shape[1], dtype=dtype)) / np.sqrt(2)
    B = torch.eye(dataT.shape[1], dtype=dtype)
    B /= torch.linalg.norm(B, ord=torch.inf)
    A = Psi(func_kernel     = func_kernel_A, 
            dtype           = dtype,
            data            = dataT, 
            learnedmat      = B, 
            tau             = tau_A)

    # power iterations
    for i in range(iter_max):
        old_B = B
        old_A = A

        distA_old = calculate_final_distance_matrix(kernelMat = old_A, data = data, dtype = dtype)

        B = Psi(func_kernel = func_kernel_B, 
                dtype       = dtype,
                data        = data, 
                learnedmat  = A, 
                tau         = tau_B)

        A = Psi(func_kernel = func_kernel_A, 
                dtype       = dtype,
                data        = dataT,
                learnedmat  = B, 
                tau         = tau_A)
        
        # check for fixed point
        res_B = torch.abs(old_B - B).max()
        rel_res_B = res_B / torch.abs(B).max()
        res_A = torch.abs(old_A - A).max()
        rel_res_A = res_A / torch.abs(A).max()

        distA = calculate_final_distance_matrix(kernelMat = A, data = data, dtype = dtype)
        
        # Compute Hilbert loss
        if writer:
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf A,A_old,kernel_gauss_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B),res_A,
                i
            )
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf B,B_old,kernel_gauss_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), res_B,
                i
            )
            writer.add_scalar(
                "Hilbert A,A_old,kernel_gauss_norm_step/sigA_%f_sigB_%f", utils.hilbert_distance(distA_old, distA),
                # "inf A,A_old,kernel_gauss_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B),res_A,
                i
            )

        if writer and label:
            writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,A,kernel_gauss_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), 
                    silhouette_score(distA.cpu(), label.cpu(), metric = 'precomputed'),
                    i
                )
        
        if torch.tensor([rel_res_A, rel_res_B]).max() < relative_residual:
            return A, B, i

    return A, B, iter_max

# Power Iteration Algorithm
def unsupervised_laplace_kernel_learning_norm(
        dataset: torch.Tensor, 
        dtype: str,
        device: str,
        label: torch.Tensor,
        rbf_sigma_A: float,
        rbf_sigma_B: float,
        func_kernel_A=None, 
        func_kernel_B=None, 
        writer=None,
        iter_max: int = 1000, 
        regularize=False, 
        regularization_param=None, 
        relative_residual: float = 1e-3, 
        normalization_steps: int=1,
        small_value: float=1e-6,
        lipschitz_of_kernel_A=None, 
        lipschitz_of_kernel_B=None, 
        B_0=None):

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        data, dataT = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )
    if label:
        label = torch.tensor(label, dtype=dtype)

    # If Lipschitzconstants are given calculate tau.
    tau_A = calculate_tau_laplace(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    tau_B = calculate_tau_laplace(dataT, lipschitz_of_kernel_B) + regularization_param if regularize else 0
    # tau_A = calculate_tau_inv_mult_quad(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    # tau_B = calculate_tau_inv_mult_quad(data.T, lipschitz_of_kernel_B) + regularization_param if regularize else 0


    # assume pairwise distance of every point is 1
    # B = (torch.ones((data.T.shape[1], data.T.shape[1]), dtype=dtype) - torch.eye(data.T.shape[1], dtype=dtype)) / np.sqrt(2)
    B = torch.eye(dataT.shape[1], dtype=dtype)
    B /= torch.linalg.norm(B, ord=torch.inf)
    A = Psi(func_kernel     = func_kernel_A, 
            dtype           = dtype,
            data            = dataT, 
            learnedmat      = B, 
            tau             = tau_A)

    # power iterations
    for i in range(iter_max):
        old_B = B
        old_A = A

        B = Psi(func_kernel = func_kernel_B, 
                dtype       = dtype,
                data        = data, 
                learnedmat  = A, 
                tau         = tau_B)

        A = Psi(func_kernel = func_kernel_A, 
                dtype       = dtype,
                data        = dataT,
                learnedmat  = B, 
                tau         = tau_A)
        
        # check for fixed point
        res_B = torch.abs(old_B - B).max()
        rel_res_B = res_B / torch.abs(B).max()
        res_A = torch.abs(old_A - A).max()
        rel_res_A = res_A / torch.abs(A).max()
        
        # Compute Hilbert loss
        if writer:
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf A,A_old,kernel_laplace_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B),res_A,
                i
            )
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf B,B_old,kernel_laplace_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), res_B,
                i
            )
        
        distA = calculate_final_distance_matrix(kernelMat = A, data = data, dtype = dtype)

        if writer and label:
            writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,A,kernel_laplace_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), 
                    silhouette_score(distA.cpu(), label.cpu(), metric = 'precomputed'),
                    i
                )
        
        if torch.tensor([rel_res_A, rel_res_B]).max() < relative_residual:
            return A, B, i

    return A, B, iter_max

# Seq Iteration Algorithm
def unsupervised_gauss_kernel_learning_seq_step(
        dataset: torch.Tensor, 
        dtype: str,
        device: str,
        label: torch.Tensor,
        rbf_sigma_A: float,
        rbf_sigma_B: float,
        alpha: float=0.9,
        gamma_A: float=0.9,
        gamma_B: float=0.9,
        func_kernel_A=None, 
        func_kernel_B=None, 
        writer=None,
        iter_max: int = 1000, 
        regularize=False, 
        regularization_param=None, 
        relative_residual: float = 1e-3, 
        normalization_steps: int=1,
        small_value: float=1e-6,
        lipschitz_of_kernel_A=None, 
        lipschitz_of_kernel_B=None, 
        B_0=None):

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        data, dataT = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )
    if label:
        label = torch.tensor(label, dtype=dtype)

    # If Lipschitzconstants are given calculate tau.
    tau_A = calculate_tau_gauss(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    tau_B = calculate_tau_gauss(dataT, lipschitz_of_kernel_B) + regularization_param if regularize else 0
    # tau_A = calculate_tau_inv_mult_quad(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    # tau_B = calculate_tau_inv_mult_quad(data.T, lipschitz_of_kernel_B) + regularization_param if regularize else 0


    # assume pairwise distance of every point is 1
    # B = (torch.ones((data.T.shape[1], data.T.shape[1]), dtype=dtype) - torch.eye(data.T.shape[1], dtype=dtype)) / np.sqrt(2)
    B = torch.eye(dataT.shape[1], dtype=dtype)
    B /= torch.linalg.norm(B, ord=torch.inf)
    A = Psi(func_kernel     = func_kernel_A, 
            dtype           = dtype,
            data            = dataT, 
            learnedmat      = B, 
            tau             = tau_A)

    # power iterations
    for i in range(iter_max):
        old_B = B
        old_A = A

        distA_old = calculate_final_distance_matrix(kernelMat = old_A, data = data, dtype = dtype)

        B_new = Psi(func_kernel = func_kernel_B, 
                dtype       = dtype,
                data        = data, 
                learnedmat  = A, 
                tau         = tau_B,
                norm        = 0)
        
        B = gamma_B * B_new

        A_new = Psi(func_kernel = func_kernel_A, 
                dtype       = dtype,
                data        = dataT,
                learnedmat  = B, 
                tau         = tau_A,
                norm        = 0)
        
        A = (1 - alpha) * old_A + alpha * gamma_A * A_new
        
        # check for fixed point
        res_B = torch.abs(old_B - B).max()
        rel_res_B = res_B / torch.abs(B).max()
        res_A = torch.abs(old_A - A).max()
        rel_res_A = res_A / torch.abs(A).max()

        distA = calculate_final_distance_matrix(kernelMat = A, data = data, dtype = dtype)
        
        # Compute Hilbert loss
        if writer:
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf A,A_old,kernel_gauss_seq_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B),res_A,
                i
            )
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf B,B_old,kernel_gauss_seq_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), res_B,
                i
            )
            writer.add_scalar(
                "Hilbert A,A_old,kernel_gauss_seq_step/sigA_%f_sigB_%f", utils.hilbert_distance(distA_old, distA),
                # "inf A,A_old,kernel_gauss_norm_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B),res_A,
                i
            )

        if writer and label:
            writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,A,kernel_gauss_seq_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), 
                    silhouette_score(distA.cpu(), label.cpu(), metric = 'precomputed'),
                    i
                )
        
        if torch.tensor([rel_res_A, rel_res_B]).max() < relative_residual:
            return A, B, i

    return A, B, iter_max

# Seq Iteration Algorithm
def unsupervised_laplace_kernel_learning_seq_step(
        dataset: torch.Tensor, 
        dtype: str,
        device: str,
        label: torch.Tensor,
        rbf_sigma_A: float,
        rbf_sigma_B: float,
        alpha: float=0.9,
        gamma_A: float=0.9,
        gamma_B: float=0.9,
        func_kernel_A=None, 
        func_kernel_B=None, 
        writer=None,
        iter_max: int = 1000, 
        regularize=False, 
        regularization_param=None, 
        relative_residual: float = 1e-3, 
        normalization_steps: int=1,
        small_value: float=1e-6,
        lipschitz_of_kernel_A=None, 
        lipschitz_of_kernel_B=None, 
        B_0=None):

    if type(dataset) is tuple:
        assert len(dataset) == 2  # correct shape

        A, B = dataset  # Recover A and B

        assert torch.sum(A < 0) == 0  # positivity
        assert torch.sum(B < 0) == 0  # positivity

    else:
        assert len(dataset.shape) == 2  # correct shape
        assert torch.sum(dataset < 0) == 0  # positivity

        # Make the transposed datasets A and B from the dataset.
        data, dataT = normalize_dataset(
            dataset,
            normalization_steps=normalization_steps,
            small_value=small_value,
            dtype=dtype,
            device=device,
        )
    if label:
        label = torch.tensor(label, dtype=dtype)

    # If Lipschitzconstants are given calculate tau.
    tau_A = calculate_tau_laplace(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    tau_B = calculate_tau_laplace(dataT, lipschitz_of_kernel_B) + regularization_param if regularize else 0
    # tau_A = calculate_tau_inv_mult_quad(data, lipschitz_of_kernel_A) + regularization_param if regularize else 0
    # tau_B = calculate_tau_inv_mult_quad(data.T, lipschitz_of_kernel_B) + regularization_param if regularize else 0


    # assume pairwise distance of every point is 1
    # B = (torch.ones((data.T.shape[1], data.T.shape[1]), dtype=dtype) - torch.eye(data.T.shape[1], dtype=dtype)) / np.sqrt(2)
    B = torch.eye(dataT.shape[1], dtype=dtype)
    B /= torch.linalg.norm(B, ord=torch.inf)
    A = Psi(func_kernel     = func_kernel_A, 
            dtype           = dtype,
            data            = dataT, 
            learnedmat      = B, 
            tau             = tau_A)

    # power iterations
    for i in range(iter_max):
        old_B = B
        old_A = A

        B_new = Psi(func_kernel = func_kernel_B, 
                dtype       = dtype,
                data        = data, 
                learnedmat  = A, 
                tau         = tau_B,
                norm        = 0)
        
        B = gamma_B * B_new

        A_new = Psi(func_kernel = func_kernel_A, 
                dtype       = dtype,
                data        = dataT,
                learnedmat  = B, 
                tau         = tau_A,
                norm        = 0)
        
        A = (1 - alpha) * old_A + alpha * gamma_A * A_new
        
        # check for fixed point
        res_B = torch.abs(old_B - B).max()
        rel_res_B = res_B / torch.abs(B).max()
        res_A = torch.abs(old_A - A).max()
        rel_res_A = res_A / torch.abs(A).max()
        
        # Compute Hilbert loss
        if writer:
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf A,A_old,kernel_laplace_seq_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B),res_A,
                i
            )
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf B,B_old,kernel_laplace_seq_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), res_B,
                i
            )
        
        distA = calculate_final_distance_matrix(kernelMat = A, data = data, dtype = dtype)

        if writer and label:
            writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,A,kernel_laplace_seq_step/sigA_%f_sigB_%f" %(rbf_sigma_A, rbf_sigma_B), 
                    silhouette_score(distA.cpu(), label.cpu(), metric = 'precomputed'),
                    i
                )
        
        if torch.tensor([rel_res_A, rel_res_B]).max() < relative_residual:
            return A, B, i

    return A, B, iter_max

###

def calculate_final_distance_matrix(kernelMat: torch.Tensor, data: torch.Tensor, dtype: str):
    
    distMat = torch.zeros((data.shape[0], data.shape[0]), dtype=dtype)
    for i in range(len(data)):
        for j in range(i, len(data)):
            distMat[i, j] = distMat[j, i] = torch.sqrt((data[i] - data[j]).T @ kernelMat @ (data[i] - data[j]))
            # distMat[i, j] = distMat[j, i] = torch.sqrt(torch.matmult(torch.matmult(kernelMat, (data[i] - data[j])).T, (data[i] - data[j])))

    return distMat

#####



