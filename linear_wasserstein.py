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

def Psi_lin_lap(data, learnedmat, tau: float=0.0, norm: int=1):
    n = data.shape[0]
    newlearned = torch.zeros((n, n), dtype=torch.double)
    
    for i in range(n):
        for j in range(i, n):
            dif = data[i] - data[j]
            newlearned[i,j] = newlearned[j,i] = -dif @ learnedmat @ dif.reshape(-1,1)
    
    dg = torch.sum(newlearned,1)
    newlearned -= torch.diag(dg)
    
    newlearned += tau * torch.eye(n)

    if norm==1:
        return newlearned / torch.abs(newlearned).max()
    if norm==0:
        return newlearned

# Power Iteration Algorithm
def unsupervised_lin_learning(dataset: torch.Tensor,
		                      label: torch.Tensor,
                              dtype: str,
                              device: str,
                              writer=None,
                              iter_max=1000, 
                              regularize=False, 
                              normalization_steps: int=1,
                              small_value: float=1e-6,
                              tau_A: torch.double=0, 
                              tau_B: torch.double=0, 
                              relative_residual = 1e-3):

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
    if regularize==False:
        tau_A = tau_B = 0

    # assume pairwise distance of every point is 1
    B = -torch.ones((dataT.shape[1], dataT.shape[1]), dtype=torch.double) + torch.eye(dataT.shape[1])
    B /= torch.abs(B).max()
    A = Psi_lin_lap(data=dataT, 
            learnedmat=B, 
            tau=tau_A)


    # power iterations
    for i in range(iter_max):
        old_B = B
        old_A = A

        distA_old = calculate_final_distance_matrix(old_A, data, dtype=torch.double)
        distA_lin_old = distA_old - torch.diag(torch.diag(distA_old))

        B = Psi_lin_lap(data    = data, 
                learnedmat  = A, 
                tau         = tau_B)

        A = Psi_lin_lap(data    = dataT, 
                learnedmat  = B, 
                tau         = tau_A)
        
        print(A.max(), A.min(), B.max(), B.min())
        
        # check for fixed point
        res_B = torch.abs(old_B - B).max()
        rel_res_B = res_B / torch.abs(B).max()
        res_A = torch.abs(old_A - A).max()
        rel_res_A = res_A / torch.abs(A).max()

        distA = calculate_final_distance_matrix(A, data, dtype=torch.double)
        distA_lin = distA - torch.diag(torch.diag(distA))

        # Compute Hilbert loss
        if writer:
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf A,A_old,lin_norm_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                res_A,
                i
            )
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf B,B_old,lin_norm_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                res_B,
                i
            )
            writer.add_scalar(
                "Hilbert A,A_old,lin_norm_lap/tauA_%f_tauB_%f", utils.hilbert_distance(distA_lin_old, distA_lin),
                #"inf A,A_old,lin_norm_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                #res_A,
                i
            )
        
        if writer!=None and label!=None:
            writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,A,lin_norm_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                    silhouette_score(distA_lin.cpu(), label.cpu(), metric = 'precomputed'),
                    i
                )


        if np.max([rel_res_A, rel_res_B]) < relative_residual:
            return A, B, i

    return A, B, iter_max

# Power Iteration Algorithm
def unsupervised_lin_learning_step(dataset: torch.Tensor,
		                      label: torch.Tensor,
                              dtype: str,
                              device: str,
                              alpha: float=0.9,
                              gamma_A: float=0.9,
                              gamma_B: float=0.9,
                              writer=None,
                              iter_max=1000, 
                              regularize=False, 
                              normalization_steps: int=1,
                              small_value: float=1e-6,
                              tau_A: torch.double=0, 
                              tau_B: torch.double=0, 
                              relative_residual = 1e-3):

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
    if regularize==False:
        tau_A = tau_B = 0

    # assume pairwise distance of every point is 1
    B = -torch.ones((dataT.shape[1], dataT.shape[1]), dtype=torch.double) + torch.eye(dataT.shape[1])
    B /= torch.abs(B).max()
    # B = torch.ones((dataT.shape[1], dataT.shape[1]), dtype=torch.double)
    A = Psi_lin_lap(data=dataT, 
            learnedmat=B, 
            tau=tau_A)


    # power iterations
    for i in range(iter_max):
        old_B = B
        old_A = A

        distA_old = calculate_final_distance_matrix(old_A, data, dtype=torch.double)
        distA_lin_old = distA_old - torch.diag(torch.diag(distA_old))

        B_new = Psi_lin_lap(data    = data, 
                learnedmat  = A, 
                tau         = tau_B,
                norm=0)
        
        B = gamma_B * B_new

        A_new = Psi_lin_lap(data    = dataT, 
                learnedmat  = B, 
                tau         = tau_A,
                norm=0)
        
        A = (1 - alpha) * old_A + alpha * gamma_A * A_new
        
        # check for fixed point
        res_B = torch.abs(old_B - B).max()
        rel_res_B = res_B / torch.abs(B).max()
        res_A = torch.abs(old_A - A).max()
        rel_res_A = res_A / torch.abs(A).max()

        distA = calculate_final_distance_matrix(A, data, dtype=torch.double)
        distA_lin = distA - torch.diag(torch.diag(distA))

        # Compute Hilbert loss
        if writer:
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf A,A_old,lin_step_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                res_A,
                i
            )
            writer.add_scalar(
                # "Hilbert D,D_new,sink_seq_step", utils.hilbert_distance(D, D_new), k
                "inf B,B_old,lin_step_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                res_B,
                i
            )
            writer.add_scalar(
                "Hilbert A,A_old,lin_step_lap/tauA_%f_tauB_%f", utils.hilbert_distance(distA_lin, distA_lin_old),
                #"inf A,A_old,lin_norm_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                #res_A,
                i
            )
        
        if writer!=None and label!=None:
            writer.add_scalar(
                    #"Hilbert D,D_new,sink_seq_norm", utils.hilbert_distance(D, D_new), k
                    "AWS,A,lin_step_lap/tauA_%f_tauB_%f" %(tau_A, tau_B), 
                    silhouette_score(distA_lin.cpu(), label.cpu(), metric = 'precomputed'),
                    i
                )


        if np.max([rel_res_A, rel_res_B]) < relative_residual:
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
