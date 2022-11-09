#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 11:18:37 2022

@author: pdavid

SCRIPT FOR THE SINGLE SOURCE AND TO EVALUATE THE NON LINEAR MODEL on a centered position with
both Dirichlet and periodic BCs

"""
#djkflmjaze
import os 
directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Code'
#directory='/home/pdavid/Bureau/Updated_BCs/Code'
os.chdir(directory)

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, plot_sketch, get_MRE, get_position_cartesian_sources
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas
import math
import pdb
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

simulation=1
# =============================================================================
# #0-Set up the sources
# #1-Set up the domain
# alpha=50
# 
# Da_t=10
# D=1
# K0=math.inf
# L=400
# 
# cells=20
# h_coarse=L/cells
# 
# 
# #Metabolism Parameters
# M=Da_t*D/L**2
# phi_0=0.4
# conver_residual=5e-5
# stabilization=0.5
# 
# #Definition of the Cartesian Grid
# x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
# y_coarse=x_coarse
# 
# #V-chapeau definition
# directness=1
# print("directness=", directness)
# 
# S=50
# Rv=L/alpha+np.zeros(S)
# #pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L
# 
# pos_s=np.array([[0.5,0.5]])*L
# random.seed(1)
# for i in range(S-1):
#     theta=random.random()*2*np.pi
#     r=random.random()*0.3*L+0.25*L
#     pos_s=np.concatenate((pos_s, [np.array([np.cos(theta), np.sin(theta)])*r+L/2]),axis=0)
# 
# Rv[1:]/=8
# 
# #ratio=int(40/cells)*2
# ratio=int(100*h_coarse//L/2)*2
# 
# print("h coarse:",h_coarse)
# K_eff=K0/(np.pi*Rv**2)
# 
# p=np.linspace(0,1,100)
# if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")
# 
# 
# C_v_array=np.ones(S) 
# 
# BC_value=np.array([0,0.2,0,0.2])
# BC_type=np.array(['Periodic', 'Periodic', 'Periodic', 'Periodic'])
# 
# 
# #What comparisons are we making 
# COMSOL_reference=1
# non_linear=1
# Peaceman_reference=1
# coarse_reference=1
# directory_COMSOL='../Tool_Validation/COMSOL_output/linear'
# directory_COMSOL_metab='../Tool_Validation/COMSOL_output/metab'
# 
# 
# #%% 2 - Plot source centers and mesh
# #Position image
# plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory)
# #%%
# 
# t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)
# Multi_FV_linear, Multi_q_linear=t.Multi()
# import pdb
# Multi_rec_linear,_,_=t.Reconstruct_Multi(0,1)
# 
# plt.contourf(Multi_rec_linear, levels=100)
# plt.title("bilinear reconstruction \n coupling model Steady State ")
# plt.colorbar(); plt.show()
# #%%
# if non_linear:
#     Multi_FV_metab, Multi_q_metab=t.Multi(M,phi_0)
#     Multi_rec_metab,_,_=t.Reconstruct_Multi(1, 1)
#     plt.imshow(Multi_rec_metab, origin='lower', vmax=np.max(Multi_rec_metab))
#     plt.title("bilinear reconstruction \n coupling model Metabolism")
#     plt.colorbar(); plt.show()
# 
# 
# #%%
# t.Linear_FV_Peaceman(0)
# t.Metab_FV_Peaceman(M, phi_0, 0)
#       
# 
# =============================================================================
#%%
def position_sources(dens, L, cyl_rad):
    """dens -> density in source/square micrometer
       L -> side length of the domain
       cyl_rad -> radius of the capillary free region
       """
    pos_s=np.zeros((0,2))
    elem_square=1/dens
    cells=np.around(L/np.sqrt(elem_square)).astype(int)   
    h=L/cells
    grid_x=np.linspace(h/2, L-h/2,cells) 
    grid_y=grid_x
    
    center=np.array([L/2, L/2])
    
    for i in grid_x: 
        for j in grid_y:
            temp_s=(np.random.rand(1,2)-1/2)*h*0.8+np.array([[i,j]])
            if np.linalg.norm(temp_s-center) > cyl_rad:
                pos_s=np.concatenate((pos_s, temp_s), axis=0)
                
    return(pos_s)

#%% - Do the analysis per angle 
theta_arr=np.linspace(0,2*np.pi,20)
from Reconstruction_extended_space import reconstruction_extended_space
from random import randrange
mean=0.7
mu=0.1 #standard deviation

density=18/240**2
L=400
R_art=L/30
R_cap=R_art/10
K_eff=math.inf
directness=10
CMRO2_max=6*10**-5
def metab_simulation(mean, std_dev, simulations, density, L,  cyl_rad, R_art, R_cap, K_eff, directness, CMRO2_max):
    avg_phi_array=np.zeros((simulations,100))
    for k in range(simulations):
        
        pos_s=np.array([[0.5,0.5]])*L
        pos_s=np.concatenate((pos_s, position_sources(density, L, L/4)), axis=0)
        S=len(pos_s)
        C_v_array=np.array([1])
        Rv=np.array([R_art])
        for i in range(S-1):
            C_v_array=np.append(C_v_array, random.gauss(mean,mu))
            Rv=np.append(Rv,R_cap)
        if np.any(pos_s>L) or np.any(pos_s<0): print("ERROR IN THE POSITIONING")
        
        cells=12
        D=1
        ratio=10
        BC_type=np.array(['Periodic','Periodic','Periodic','Periodic'])
        BC_value=np.zeros(4)
        h_coarse=L/cells
        x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
        y_coarse=x_coarse
        phi_0=0.4
        t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

        if CMRO2_max>10**-3:
            t.stabilization=0.005
        elif CMRO2_max>10**-1:
            t.stabilization=0.02
        else:
            t.stabilization=2
        print(t.stabilization)
        
        plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory)
        C_v_array[C_v_array>1]=1
        C_v_array[C_v_array<0]=0
        Multi_FV_metab, Multi_q_metab=t.Multi(CMRO2_max,phi_0)
# =============================================================================
#         Multi_rec_metab,_,_=t.Reconstruct_Multi(1, 0)
#         
#         
#         plt.imshow(Multi_rec_metab, origin='lower', vmax=np.max(Multi_rec_metab))
#         plt.title("bilinear reconstruction \n coupling model Metabolism")
#         plt.colorbar(); plt.show()
# =============================================================================
        REC_phi_array=np.zeros((0,100))
        REC_x_array=np.zeros((0,100))
        REC_y_array=np.zeros((0,100))
        for i in theta_arr:
            REC_x=np.linspace(0,L*np.cos(i)/2,100)+L/2
            REC_y=np.linspace(0,L*np.sin(i)/2,100)+L/2
            r=reconstruction_extended_space(pos_s, Rv, h_coarse, L,K_eff, D, directness)
            r.s_FV=Multi_FV_metab
            r.q=Multi_q_metab
            r.set_up_manual_reconstruction_space(REC_x,REC_y)
            r.full_rec(C_v_array, BC_value, BC_type)
            REC_phi=r.s+r.SL+r.DL
    # =============================================================================
    #         plt.plot(REC_phi)
    #         plt.show()
    # =============================================================================
            REC_x_array=np.concatenate((REC_x_array, [REC_x]), axis=0)
            REC_y_array=np.concatenate((REC_y_array, [REC_y]), axis=0)
            REC_phi_array=np.concatenate((REC_phi_array, [REC_phi]), axis=0)
        
        avg_phi_REC=np.sum(REC_phi_array, axis=0)/20
        avg_phi_array[k]=avg_phi_REC
        plt.plot(np.linspace(0,L/2,100),avg_phi_REC)
        plt.xlabel('$\   m$')
        plt.title("average of the average")
        plt.show()    
    return(np.sum(avg_phi_array, axis=0)/(k+1))

#%%

#%%
alpha=20
import math
L=400
K_eff=math.inf
directness=20
CMRO2_max=2*10**-5
CMRO_range=np.linspace(2e-4, 6e-3, 4)
CMRO_range=np.linspace(5e-7, 2e-4, 10) #original

CMRO_range=np.linspace(5e-7, 0.5e-2,3)

Da_t_range=CMRO_range*(L/4)**2
print("Damkohler range=", np.around(Da_t_range, decimals=2))
std=0.2
#%%
if simulation:
    for mean in np.array([0.8,0.5,0.2]):
        c=0
        for CMRO2_max in CMRO_range[::-1]:
        #for CMRO2_max in CMRO_range:
            print(c)
            print("CMRO2_max ",CMRO2_max)
            print("mean", mean)
            a=metab_simulation(mean, std, 2, 0.0003,L,L/4,L/alpha,3, K_eff, directness, CMRO2_max)
            np.save('../Figures_and_Tests/Case_metab/average_mean{}_std{}_Dat{}'.format(int(mean*10), int(std*10),Da_t_range[c]), a)
            c+=1   
#%%

CMRO_range=np.linspace(5e-7, 2e-4, 10) #original
for c in range(10):
    d=np.zeros((0,100))
    for mean in np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8]):
        CMRO2_max=CMRO_range[c]
        b=np.load('../Figures_and_Tests/Case_metab/First_tests/average_mean{}_std{}_M{}.npy'.format(int(mean*10), int(std*10),c))
        plt.plot(b,label='mean={}, std={}'.format(mean, std))
        plt.ylim(0,1.1)
        d=np.concatenate((d, [b]), axis=0)
    Damkohler=CMRO_range[len(CMRO_range)-c-1]*(L/4)**2
    #title_met='%.2E' % Damkohler
    plt.title('Da=' + str(Damkohler))
    plt.legend()
    plt.show()
#%%

# import the random module
import random
  
# determining the values of the parameters
mu = 100
sigma = 5
  
# using the gauss() method
arr=np.array([])

for i in range(100000):
    arr=np.append(arr, random.gauss(mu, sigma))

plt.hist(arr, bins=400)