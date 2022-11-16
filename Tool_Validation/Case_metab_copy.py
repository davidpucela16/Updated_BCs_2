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
#directory='/home/pdavid/Bureau/Code/Updated_BCs_2/Code'
directory='/home/pdavid/Bureau/Updated_BCs_2/Code'
os.chdir(directory)

mod_metab='../Tool_Validation'
os.chdir(mod_metab)
from Metab_testing_module import position_sources, metab_simulation, get_met_plateau


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
         'ytick.labelsize':'x-large',
          'text.usetex' : False}
pylab.rcParams.update(params)

from Reconstruction_extended_space import reconstruction_extended_space
from random import randrange

simulation=1



#%% - Do the analysis per angle 


L=400 #According to the experimental data
alpha=20
R_art=L/alpha
R_cap=R_art/10
K_eff=math.inf
directness=10
    
#  Define crucial parameters
L_cap=0.05 #mm
P_max=40 #mmHg
D = 4e-3 # mm^2 s^-1
solubility= 1.39e-6 #micromol mm^-3 mmHg^-3

#Da_t range defined with the values in Natalie's paper:
CMRO2_max=3 #in micromol s^-1 cm^-3 obtained from literature

Da_t_max=CMRO2_max/5.3376

Da_t_range=np.linspace(0,Da_t_max,5)
mean_range=np.array([0.4,0.45,0.5,0.55])
L_char=0.050 #milimeters
solubility= 1.39e-6 #micromol/(mm3*mmHg)
P_max=40 #mmHgµ
D=4e-3 #mm2/s

Prop_Da_M=P_max*D*solubility/L_char**2
M_values=Da_t_range*Prop_Da_M

M_values_min=Da_t_range*5.3376

density_range=np.concatenate(([250],np.linspace(352,528,4)))
real_density=np.concatenate(([250/440],np.linspace(0.8,1.2,4)))
std=0.2
simulations=4


#%%
mean=0.5
for layer in range(len(density_range)):
    #layer represents the cortical layer where we at
    for Da in range(len(Da_t_range)):
        print("layer", layer)
        print("Da pos ", Da)
        #M=M_values[Da]
        M=Da_t_range[Da]*L_char**2/10 #therefore the units are mm-2
        density=density_range[layer]
        #mean=mean_range[layer]
        a=metab_simulation(mean, std, 2, density,L,L/4,L/alpha,3, K_eff, directness, M)
        np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon_test/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
         

#############################################################################################
##############################################################################################
#############################################################################################
##############################################################################################
#%%
mean=mean_range[0]
for layer in range(len(density_range)):
    #layer represents the cortical layer where we at
    for Da in range(len(Da_t_range)):
        print("layer", layer)
        print("Da pos ", Da)
        #M=M_values[Da]
        M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
        density=density_range[layer]
        #mean=mean_range[layer]
        a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
        np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
        

#%%
mean=mean_range[1]
for layer in range(len(density_range)):
    #layer represents the cortical layer where we at
    for Da in range(len(Da_t_range)):
        print("layer", layer)
        print("Da pos ", Da)
        #M=M_values[Da]
        M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
        density=density_range[layer]
        #mean=mean_range[layer]
        a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
        np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
        
#%%
mean=mean_range[2]
for layer in range(len(density_range)):
    #layer represents the cortical layer where we at
    for Da in range(len(Da_t_range)):
        print("layer", layer)
        print("Da pos ", Da)
        #M=M_values[Da]
        M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
        density=density_range[layer]
        #mean=mean_range[layer]
        a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
        np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
        
#%%
mean=mean_range[3]
for layer in range(len(density_range)):
    #layer represents the cortical layer where we at
    for Da in range(len(Da_t_range)):
        print("layer", layer)
        print("Da pos ", Da)
        #M=M_values[Da]
        M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
        density=density_range[layer]
        #mean=mean_range[layer]
        a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
        np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
        

#############################################################################################
##############################################################################################
#############################################################################################
##############################################################################################

#%%

    



#%%
for mean in mean_range:
    for layer in range(len(density_range)):
        #layer represents the cortical layer where we at
        c=0
        for Da in range(len(Da_t_range)):
        #for Da in range(2):
            print("layer", layer)
            print("Da pos ", Da)
            M=M_values[Da]
            density=density_range[layer]
            #mean=mean_range[layer]
            b=np.load('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}.npy'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)))
            b[b>1]=1
            plt.plot(b, label='CMRO2={}'.format(np.around(Da_t_range[c]*5.33))) #The 2.4e3 is to conver it to micromol cm-3 s-1
            c+=1  
        plt.legend()
        plt.title('mean{}, density{}'.format(mean, real_density[layer]))  
        plt.show()
        

    
    
    
    
    
    
    
    
    
    
    
    
# 14 - Nov - 2022
#%%
density_range[-1]=520
layer=4

#%%
mean=mean_range[0]
#layer represents the cortical layer where we at
for Da in range(len(Da_t_range)):
    print("layer", layer)
    print("Da pos ", Da)
    #M=M_values[Da]
    M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
    density=density_range[layer]
    #mean=mean_range[layer]
    a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
    np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
    
#%%

mean=mean_range[1]
#layer represents the cortical layer where we at
for Da in range(len(Da_t_range)):
    print("layer", layer)
    print("Da pos ", Da)
    #M=M_values[Da]
    M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
    density=density_range[layer]
    #mean=mean_range[layer]
    a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
    np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
    
#%%

mean=mean_range[2]
#layer represents the cortical layer where we at
for Da in range(len(Da_t_range)):
    print("layer", layer)
    print("Da pos ", Da)
    #M=M_values[Da]
    M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
    density=density_range[layer]
    #mean=mean_range[layer]
    a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
    np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
    
#%%

mean=mean_range[3]
#layer represents the cortical layer where we at
for Da in range(len(Da_t_range)):
    print("layer", layer)
    print("Da pos ", Da)
    #M=M_values[Da]
    M=Da_t_range[Da]*L_char**2 #therefore the units are mm-2
    density=density_range[layer]
    #mean=mean_range[layer]
    a=metab_simulation(mean, std, simulations, density,L,L/4,L/alpha,3, K_eff, directness, M)
    np.save('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)), a)
    
#%%
    
for mean in mean_range:
    for layer in range(len(density_range)):
        #layer represents the cortical layer where we at
        c=0
        for Da in range(len(Da_t_range)):
            print("layer", layer)
            print("Da pos ", Da)
            M=M_values[Da]
            density=density_range[layer]
            #mean=mean_range[layer]
            b=np.load('../Figures_and_Tests/Case_metab/phys_vals_mean/Da={}_dens={}_layer={}.npy'.format(Da_t_range[Da], density, layer))
            plt.plot(b, label='CMRO2={}'.format(np.around(M_values[c]*2.4e3))) #The 2.4e3 is to conver it to micromol cm-3 s-1
            c+=1  
        plt.legend()
        plt.title('mean{}, density{}'.format(mean, real_density[layer]))  
        plt.show()



#%% - Calculate plateau

plat_mat=np.zeros((len(mean_range), len(density_range), len(Da_t_range)))
for c in range(len(mean_range)):
    mean=mean_range[c]
    for layer in range(len(density_range)):
        #layer represents the cortical layer where we at
        for Da in range(len(Da_t_range)):
            print("layer", layer)
            print("Da pos ", Da)
            M=M_values[Da]
            density=density_range[layer]
            #mean=mean_range[layer]
            b=np.load('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}.npy'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)))
            plateau=np.sum(b[int(len(b)/2):])/int(len(b)/2)
            plat_mat[c, layer, Da]=plateau
    
    
#%%
for layer in range(len(density_range)):
    plt.imshow(plat_mat[:,layer,:], origin='lower', extent=(M_values_min[0], M_values_min[-1],M_values_min[0], M_values_min[-1]))
    plt.ylabel("Capillary mean")
    plt.xlabel("CMRO2_max")
    plt.colorbar()
    plt.title("Cortical layer: {}".format(layer))
    plt.savefig('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/fig_layer{}'.format(layer))
    plt.show()
    
#%%
for Da in range(len(Da_t_range)):
    plt.imshow(plat_mat[:,:,Da], origin='lower', extent=(M_values_min[0], M_values_min[-1],M_values_min[0], M_values_min[-1]))
    matrix=plat_mat[:,:,Da]
    plt.ylabel("Capillary mean")
    plt.xlabel("Density")
    plt.colorbar()
    plt.title("CMRO2_max={}".format(M_values_min[Da]))
    plt.savefig('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/CMRO2_max={}'.format(int(M_values_min[Da]*100)))
    plt.show()
    
#%%
plat_mat=np.zeros((len(mean_range), len(density_range), len(Da_t_range)))
for layer in range(len(density_range)):
    for c in range(len(mean_range)):
        mean=mean_range[c]
        #layer represents the cortical layer where we at
        for Da in range(len(Da_t_range)):
            print("layer", layer)
            print("Da pos ", Da)
            M=M_values[Da]
            density=density_range[layer]
            #mean=mean_range[layer]
            
            b=np.load('../Figures_and_Tests/Case_metab/phys_vals_mean_bon/mean={}_Da={}_layer={}.npy'.format(int(mean*100),int(Da_t_range[Da]*100), int(layer)))
            b[b>1]=1
            plt.plot(b, label="CMRO2_max={}".format(M_values_min[Da]))
        
        plt.legend()
        plt.title("layer={}, mean={}".format(layer, mean))
        plt.show()
        
            
            
            
            
            
#%%
    
    
    
plt.imshow(plat_mat[:,-1,:]-plat_mat[:,0,:],extent=(M_values_min[0], M_values_min[-1],M_values_min[0], M_values_min[-1]))
plt.xlabel("CMRO2_max")
plt.colorbar()
    
    
    
    
    
    
    
    
    
    
    
    
    
#%%
CMRO_range=np.linspace(0,3,5) #in micromol min-1 cm-3
CMRO_range=CMRO_range/2.4e3 #Diff coeff and min -> s
density=352
if simulation:
    for mean in np.array([0.8,0.6,0.4,0.2]):
        c=0
        for CMRO2_max in CMRO_range[::-1]:
        #for CMRO2_max in CMRO_range:
            print("CMRO2_max ",CMRO2_max)
            print("mean", mean)
            a=metab_simulation(mean, std, 2, density,L,L/4,L/alpha,3, K_eff, directness, CMRO2_max)
            pp=int(np.where(CMRO_range==CMRO2_max)[0])
            np.save('../Figures_and_Tests/Case_metab/No_dens/average_mean{}_std{}_met{}'.format(int(mean*10), int(std*10),pp), a)
            c+=1   
            
#%%
CMRO_range=np.linspace(0,3,5) #in micromol min-1 cm-3
CMRO_range=CMRO_range/2.4e3 #Diff coeff and min -> s
density=352
for mean in np.array([0.8,0.6,0.4,0.2]):
    c=0
    for CMRO2_max in CMRO_range:
        #for CMRO2_max in CMRO_range:
        print(c)
        print("CMRO2_max ",CMRO2_max)
        print("mean", mean)
        pp=int(np.where(CMRO_range==CMRO2_max)[0])
        b=np.load('../Figures_and_Tests/Case_metab/No_dens/average_mean{}_std{}_met{}.npy'.format(int(mean*10), int(std*10),pp))
        plt.plot(b, label='CMRO2={}'.format(CMRO_range[c]*2.4e3))
        
        c+=1  
    plt.legend()
    plt.title('mean{}, density{}'.format(mean, real_density[np.where(density_range==density)[0]]))  
    plt.show()





#%% - November - 16th. Keep the metabolism fixed and change the density 

simulations=10

M_value_min=1 #micro mol / (cm^2 s)
Damk=M_value_min/5.3376
M_value_code=Damk*Prop_Da_M

for density in CMRO_range[::-1]:
#for CMRO2_max in CMRO_range:
    print("CMRO2_max ",CMRO2_max)
    print("mean", mean)
    a=metab_simulation(mean, std, 2, density,L,L/4,L/alpha,3, K_eff, directness, CMRO2_max)
    pp=int(np.where(CMRO_range==CMRO2_max)[0])
    np.save('../Figures_and_Tests/Case_metab/No_dens/average_mean{}_std{}_met{}'.format(int(mean*10), int(std*10),pp), a)
    c+=1  
































###### BELOW IS THE STUFF CODED BEFORE NOVEMBER 9th



#%%

CMRO2_max=2*10**-5
CMRO_range=np.linspace(2e-4, 6e-3, 4)
CMRO_range=np.linspace(5e-7, 2e-4, 10) #original

CMRO_range=np.linspace(5e-7, 0.5e-2,3)

Da_t_range=CMRO_range*(L/4)**2
print("Damkohler range=", np.around(Da_t_range, decimals=2))
std=0.2

#%%

mean=0.5

if simulation:
    for density in np.array([0.00026,0.0003,0.00034,0.00038,0.00042]):
        c=0
        for CMRO2_max in CMRO_range:
        #for CMRO2_max in CMRO_range:
            print(c)
            print("CMRO2_max ",CMRO2_max)
            print("mean", mean)
            a=metab_simulation(mean, std, 20, density,L,L/4,L/alpha,3, K_eff, directness, CMRO2_max)
            np.save('../Figures_and_Tests/Case_metab/average_mean{}_density{}_Dat{}'.format(int(mean*10), density*100,Da_t_range[c]), a)
            c+=1   
            
#%% 
for density in np.array([0.00026,0.0003,0.00034,0.00038,0.00042]):
    c=0
    for CMRO2_max in CMRO_range[::-1]:
    #for CMRO2_max in CMRO_range:
        print(c)
        print("CMRO2_max ",CMRO2_max)
        print("mean", mean)
        b=np.load('../Figures_and_Tests/Case_metab/average_mean{}_density{}_Dat{}.npy'.format(int(mean*10), density*100,Da_t_range[c]))
        plt.plot(b)
        plt.title('average_mean{}_density{}_Dat{}'.format(int(mean*10), density*100,Da_t_range[c]))
        plt.show()
        c+=1  
#%% ------------------ First tests
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
            
#%% - Second tests
CMRO_range=np.linspace(5e-7, 0.5e-2,3)
Da_t_range=np.array([0.005,25.0025,50.0])[::-1]
std=0.2
for c in np.arange(len(CMRO_range)):
    d=np.zeros((0,100))
    for mean in np.array([0.8,0.5,0.2]):
        CMRO2_max=CMRO_range[np.array([2,1,0])[c]]
        b=np.load('../Figures_and_Tests/Case_metab/Second_tests/average_mean{}_std{}_Dat{}.npy'.format(int(mean*10), int(std*10),Da_t_range[c]))
        plt.plot(b,label='mean={}'.format(mean, std))
        plt.ylim(0,1.1)
        d=np.concatenate((d, [b]), axis=0)
    Damkohler=Da_t_range[np.array([2,1,0])[c]]
    title_met='%.2E' % Damkohler
    title='M={}'.format(CMRO_range[c]*2.4e3) + '$\mu mol cm-3 min-1$; std= ' + str(std)
    plt.title(title)
    plt.legend()
    plt.savefig('../Figures_and_Tests/Case_metab/' + title + '.pdf')
    plt.show()

#%%
std=0.2
CMRO_range=np.linspace(5e-7, 2e-4, 10) #original


for c in range(10):
    d=np.zeros((0,100))
    for mean in np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8]):
        CMRO2_max=CMRO_range[c]
        b=np.load('../Figures_and_Tests/Case_metab/First_tests/average_mean{}_std{}_M{}.npy'.format(int(mean*10), int(std*10),c))
        plt.plot(b,label='mean={}'.format(mean))
        plt.ylim(0,1.1)
        d=np.concatenate((d, [b]), axis=0)
    Damkohler=CMRO_range[len(CMRO_range)-c-1]/Prop_Da_M
    
    title_Dam='%.2E' % Damkohler
    title='$CMRO2,max =${}'.format(np.around(CMRO_range[len(CMRO_range)-c-1]*2400, decimals=2))
    plt.title(title)
    for xc in np.array([0.2,0.4,0.6,0.8]):
        plt.axhline(y=xc, color='k', linestyle='--')
    plt.legend()
    plt.savefig('../Figures_and_Tests/Case_metab/' + title + '.pdf')
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