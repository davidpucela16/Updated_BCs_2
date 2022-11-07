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
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs/Code'
directory='/home/pdavid/Bureau/SS/2D_cartesian/Updated_BCs/Code' #Malpighi
os.chdir(directory)

directory_script='/home/pdavid/Bureau/SS/2D_cartesian/Updated_BCs/Figures_and_Tests/Multiple_sources'

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources, plot_sketch
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing, extract_COMSOL_data

import pdb
import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (12,12),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


#0-Set up the sources
#1-Set up the domain
alpha=50

Da_t=10
D=1
K0=1
L=240

cells=10
h_coarse=L/cells


#Metabolism Parameters
M=Da_t*D/L**2
phi_0=0.4
conver_residual=5e-5
stabilization=0.5

#Definition of the Cartesian Grid
x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
y_coarse=x_coarse

#V-chapeau definition
directness=4
print("directness=", directness)


#pos_s=(1-np.array([[0.5,0.05+1/alpha/2]]))*L

pos_s1=np.array([[0.45,0.02],[0.24,0.17],[0.6,0.23],[0.23,0.27],[0.55,0.33],[1.02,0.41],[0.96,0.43]])
pos_s2=np.array([[0.27,0.6],[0.53,0.65],[0.59,0.62],[0.67,0.69],[0.13,0.75],[0.15,0.93],[0.2,0.87],[0.28,0.98],[0.8,0.85],[0.83,0.92]])
pos_s3=np.concatenate((pos_s1, pos_s2))

pos_s3[:,0]-=0.06
pos_s3[:,1]-=0.03

pos_s=(pos_s3*0.8+0.1)*L

pos_s[8,0]=127
pos_s[9,0]=140

S=len(pos_s)
Rv=L/alpha+np.zeros(S)
#ratio=int(40/cells)*2
ratio=int(100*h_coarse//L/2)

print("h coarse:",h_coarse)
K_eff=K0/(np.pi*Rv**2)


p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S) 
C_v_array[[2,5,8,11,14]]=0

BC_value=np.array([0,0,0.3,0.3])
BC_type=np.array(['Neumann',  'Neumann','Dirichlet', 'Dirichlet'])


#What comparisons are we making 
COMSOL_reference=1
non_linear=1
Peaceman_reference=0
coarse_reference=1
directory_COMSOL='../Figures_and_Tests/Multiple_sources/COMSOL_output/linear'
directory_COMSOL_metab='../Figures_and_Tests/Multiple_sources/COMSOL_output/metab'

#%%
plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
#%%

#%%
q_linear, FEM_phi_linear, FEM_x_linear, FEM_y_linear = extract_COMSOL_data(directory_COMSOL, [1,0,0])


#%%

t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

Multi_FV_linear, Multi_q_linear=t.Multi()
Multi_rec_linear,_,_=t.Reconstruct_Multi(0,0, FEM_x_linear, FEM_y_linear)


#%%
diff=Multi_rec_linear-FEM_phi_linear

plt.tricontourf(FEM_x_linear, FEM_y_linear,diff, levels=100)
plt.colorbar()
plt.title('error')

#%%
plt.plot(q_linear)
plt.plot(Multi_q_linear)


#%%

plt.plot(np.abs(q_linear-Multi_q_linear)/np.abs(q_linear))
plt.title("relative error")

plt.plot(np.abs(q_linear-Multi_q_linear))
plt.title("abs error")

#%%
Multi_FV_metab, Multi_q_metab=t.Multi(M,phi_0)


q_metab = extract_COMSOL_data(directory_COMSOL_metab, [1,0,0])

#%%
q_metab=np.squeeze(np.array(q_metab))
plt.plot(q_metab)
plt.plot(Multi_q_metab)
plt.show()

plt.plot(np.abs(q_metab-Multi_q_metab)/np.abs(q_metab))
plt.title("relative error")
plt.show()
plt.plot(np.abs(q_metab-Multi_q_metab))
plt.title("abs error")
plt.show()
