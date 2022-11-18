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
directory='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs_2/Code'
directory='/home/pdavid/Bureau/Updated_BCs_2/Code'
os.chdir(directory)
directory_script='/home/pdavid/Bureau/Code/SS_auto57/2D_cartesian/Updated_BCs_2/Figures_and_Tests/base'
directory_script='/home/pdavid/Bureau/Updated_BCs_2/Figures_and_Tests/base'

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources, plot_sketch
from Reconstruction_extended_space import reconstruction_extended_space
from Testing import Testing, extract_COMSOL_data, FEM_to_Cartesian
from Reconstruction_functions import coarse_cell_center_rec

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8,8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


#0-Set up the sources
#1-Set up the domain
alpha=50

Da_t=1
D=1
K0=1
L=240

cells=5
h_coarse=L/cells



#Metabolism Parameters
M=Da_t*D/L**2
M=6*10**-4
phi_0=0.4
conver_residual=5e-5
stabilization=0.5

#Definition of the Cartesian Grid
x_coarse=np.linspace(h_coarse/2, L-h_coarse/2, int(np.around(L/h_coarse)))
y_coarse=x_coarse

#V-chapeau definition
directness=2
print("directness=", directness)

S=1
Rv=L/alpha+np.zeros(S)
pos_s=np.array([[0.5,0.5]])*L

#ratio=int(40/cells)*2
ratio=int(100*h_coarse//L/2)


print("h coarse:",h_coarse)
K_eff=K0/(np.pi*Rv**2)


p=np.linspace(0,1,100)
if np.min(p-M*(1-phi_0/(phi_0+p)))<0: print("There is an error in the metabolism")


C_v_array=np.ones(S) 

BC_value=np.array([0,0.2,0,0.2])
BC_type=np.array(['Periodic', 'Periodic', 'Neumann', 'Dirichlet'])


#What comparisons are we making 
COMSOL_reference=1
non_linear=1
Peaceman_reference=0
directory_COMSOL= directory_script + '/COMSOL_output/linear'
directory_COMSOL_metab=directory_script + '/COMSOL_output/metab'

#%%
plot_sketch(x_coarse, y_coarse, directness, h_coarse, pos_s, L, directory_script)
#%%
q_linear, FEM_phi_linear, FEM_x_linear, FEM_y_linear, FEM_x_1D_linear, FEM_y_1D_linear, x_1D_linear, y_1D_linear = extract_COMSOL_data(directory_COMSOL, [1,1,1])

q_metab, FEM_phi_metab, FEM_x_metab, FEM_y_metab, FEM_x_1D_metab, FEM_y_1D_metab, x_1D_metab, y_1D_metab = extract_COMSOL_data(directory_COMSOL_metab, [1,1,1])

#%%

t=Testing(pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value)

Multi_FV_linear, Multi_q_linear=t.Multi()
Multi_rec_linear,_,_=t.Reconstruct_Multi(0,1, FEM_x_linear, FEM_y_linear)

#%%
c=0
plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
plt.plot(x_1D_linear, FEM_x_1D_linear, label="COMSOL")
plt.xlabel("x")
plt.legend()
plt.title("linear")
plt.show()

plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
plt.plot(y_1D_linear, FEM_y_1D_linear, label="COMSOL")
plt.xlabel("y")
plt.legend()
plt.title("linear")
plt.show()

#%%
from Testing import Testing, extract_COMSOL_data, FEM_to_Cartesian
from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
if non_linear:
    
    Multi_FV_metab, Multi_q_metab=t.Multi(M,phi_0)
    Multi_rec_metab,_,_=t.Reconstruct_Multi(1, 1)
    
    plt.plot(t.x_fine, t.array_phi_field_x_Multi[c], label="Multi")
    plt.plot(x_1D_metab, FEM_x_1D_metab, label="COMSOL")
    plt.xlabel("x")
    plt.title("Metabolism")
    plt.legend()
    plt.show()
    
    plt.plot(t.y_fine, t.array_phi_field_y_Multi[c], label="Multi")
    plt.plot(y_1D_metab, FEM_y_1D_metab, label="COMSOL")
    plt.xlabel("y")
    plt.legend()
    plt.title("Metabolism")
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%
a,_,_=t.Reconstruct_Multi(0,0)
x_c=np.zeros((cells*t.ratio)**2)
y_c=x_c.copy()

for i in range(cells*t.ratio):
    x_c[i*cells*ratio:(i+1)*cells*ratio]=t.x_fine
    y_c[i*cells*ratio:(i+1)*cells*ratio]=t.y_fine[i]
#%%
toreturn=np.array([np.around(x_c, decimals=2), np.around(y_c, decimals=2),np.around(np.ndarray.flatten(a), decimals=2)]).T

import pandas as pd

b=pd.DataFrame(toreturn)
b.columns=["x", "y", "phi"]

b.to_csv(directory_script + '/try.csv', sep=',', index=None)

