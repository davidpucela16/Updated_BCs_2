#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 08:14:58 2022

@author: pdavid

TESTING MUDULE

"""

import numpy as np 
import matplotlib.pyplot as plt
from FV_reference import FV_validation

from Module_Coupling import assemble_SS_2D_FD, non_linear_metab
from reconst_and_test_module import reconstruction_sans_flux
from Small_functions import coord_to_pos,pos_to_coords,get_MAE, get_MRE, get_position_cartesian_sources
from Reconstruction_extended_space import reconstruction_extended_space

import random 
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
import matplotlib.pylab as pylab
import pandas
import copy

import pdb

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6,6 ),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

class Testing():
    def __init__(self,pos_s, Rv, cells, L,  K_eff, D, directness, ratio, C_v_array, BC_type, BC_value):
        """The solution to each case is stored as the solution on the FV grid 
        and an array of the estimation of the vessel tissue exchanges (some type 
        of q array)
        
        It further stores the concentration field on straight vertical and horizontal lines 
        passing through each of the centers of the circular sources"""
        self.h_coarse=L/cells
        self.L=L
        self.ratio=ratio
        self.directness=directness
        self.Rv=Rv
        self.pos_s=pos_s
        self.K0=np.pi*Rv**2*K_eff
        self.D=D
        self.C_v_array=C_v_array
        self.BC_type=BC_type
        self.BC_value=BC_value
        self.cells=cells
        self.K_eff=K_eff
        
        
        #Metabolism Parameters by default
        self.conver_residual=5e-5
        self.stabilization=0.5
        
        #Definition of the Cartesian Grid
        self.x_coarse=np.linspace(self.h_coarse/2, L-self.h_coarse/2, int(np.around(L/self.h_coarse)))
        self.y_coarse=self.x_coarse.copy()
        
        self.x_fine=np.linspace(self.h_coarse/(2*ratio), L-self.h_coarse/(2*ratio), int(np.around(L*ratio/self.h_coarse)))
        self.y_fine=self.x_fine.copy()
        
    def Metab_FV_Peaceman(self, M, phi_0, Peaceman):
        """Peaceman=1 if Peaceman coupling model
        Peaceman=0 if no coupling"""
        L=self.L
        pos_s=self.pos_s
        cells=self.cells
        C_v_array=self.C_v_array
        Rv=self.Rv
        K_eff=self.K0/(np.pi*self.Rv**2)
        BC_type=self.BC_type
        BC_value=self.BC_value 
        x_fine, y_fine=self.x_fine, self.y_fine
        
        FV=FV_validation(L, cells*self.ratio, pos_s, C_v_array, self.D, K_eff, Rv,BC_type, BC_value, Peaceman)
        FV_metab=FV.solve_non_linear_system(phi_0,M, self.stabilization)
        FV_metab=(FV.phi_metab[-1]+FV.Corr_array).reshape(cells*self.ratio, cells*self.ratio)
        q_metab=FV.get_q_metab()
        mat_metab=FV_metab.reshape(cells*self.ratio, cells*self.ratio)
    
        plt.imshow(FV_metab, origin='lower', vmax=np.max(FV_metab))
        plt.title("FV metab reference")
        plt.colorbar(); plt.show()
        
        array_phi_field_x_metab=np.zeros((len(pos_s), len(self.x_fine)))
        array_phi_field_y_metab=np.zeros((len(pos_s), len(self.y_fine)))
        c=0
        for i in pos_s:
            pos=coord_to_pos(x_fine, y_fine, i)
            plt.plot(FV_metab[pos//len(FV.x),:], label="Peac metab")
            plt.legend()
            plt.show()
            array_phi_field_x_metab[c]=mat_metab[pos//len(FV.x),:]
            array_phi_field_y_metab[c]=mat_metab[:,int(pos%len(FV.x))]
            c+=1
            
        if Peaceman:
            self.array_phi_field_x_metab_Peaceman=array_phi_field_x_metab
            self.array_phi_field_y_metab_Peaceman=array_phi_field_y_metab
            self.q_metab_Peaceman=q_metab
            self.FV_metab_Peaceman=FV_metab+FV.get_corr_array(1)
            return(FV_metab, q_metab)
        else:
            self.array_phi_field_x_metab_noPeaceman=array_phi_field_x_metab
            self.array_phi_field_y_metab_noPeaceman=array_phi_field_y_metab
            self.q_metab_noPeaceman=q_metab
            self.FV_metab_noPeaceman=FV_metab  
            return(FV_metab, q_metab)
            
    def Linear_FV_Peaceman(self, Peaceman):
        """Performs the simulation for a refined Peaceman"""
        
        L=self.L
        pos_s=self.pos_s
        cells=self.cells
        C_v_array=self.C_v_array
        Rv=self.Rv
        BC_type=self.BC_type
        BC_value=self.BC_value 
        K_eff=self.K0/(np.pi*self.Rv**2)
        
        FV=FV_validation(L, cells*self.ratio, pos_s, C_v_array, self.D, K_eff, Rv,BC_type, BC_value, Peaceman)
        #####################################
        #####  CORR ARRAY!!
        #####################################
        FV_linear=FV.solve_linear_system()
        q_linear=FV.get_q_linear()
        mat_linear=FV_linear.reshape(cells*self.ratio, cells*self.ratio)
        
        plt.imshow(mat_linear, origin='lower')
        plt.colorbar()
        plt.title("FV Peaceman solution, linear system\n mesh:{}x{}".format(self.ratio*cells, self.ratio*cells))
        plt.show()
        
        array_phi_field_x_linear=np.zeros((len(pos_s), len(self.x_fine)))
        array_phi_field_y_linear=np.zeros((len(pos_s), len(self.y_fine)))
        
        c=0
        for i in pos_s:
            pos=coord_to_pos(FV.x, FV.y, i)
            
            plt.plot(self.x_fine, mat_linear[pos//len(FV.x),:], label="FV")
            plt.xlabel("x $\mu m$")
            plt.legend()
            plt.title("Linear Peaceman solution")
            plt.show()
            
            array_phi_field_x_linear[c]=mat_linear[int(pos//len(FV.x)),:]
            array_phi_field_y_linear[c]=mat_linear[:,int(pos%len(FV.x))]
            c+=1
        if Peaceman:
            self.array_phi_field_x_linear_Peaceman=array_phi_field_x_linear
            self.array_phi_field_y_linear_Peaceman=array_phi_field_y_linear
            self.q_linear_Peaceman=q_linear
            self.FV_linear_Peaceman=FV_linear+FV.get_corr_array()
        else:
            self.array_phi_field_x_linear_noPeaceman=array_phi_field_x_linear
            self.array_phi_field_y_linear_noPeaceman=array_phi_field_y_linear
            self.q_linear_noPeaceman=q_linear
            self.FV_linear_noPeaceman=FV_linear            
        
        return(FV_linear,q_linear)
    
    def Multi(self, *Metab):
        cells=self.cells
        n=non_linear_metab(self.pos_s, self.Rv, self.h_coarse, self.L, self.K_eff, self.D, self.directness)
        n.solve_linear_prob(self.BC_type, self.BC_value, self.C_v_array)
        Multi_FV_linear=n.s_FV_linear
        Multi_q_linear=n.q_linear
        self.Multi_q_linear=Multi_q_linear
        self.Multi_FV_linear=Multi_FV_linear
        self.phi_bar=n.phi_bar
        self.phi_bar2=n.phi_bar2
        
        self.s_blocks=n.s_blocks
        n.phi_0, n.M=1,1
# =============================================================================
#         
#         n.assemble_it_matrices_Sampson(np.ndarray.flatten(Multi_FV_linear), Multi_q_linear)
#         plt.imshow(n.rec_sing.reshape(cells,cells)+Multi_FV_linear, origin='lower', extent=[0,self.L, 0, self.L])
#         plt.title("Average value reconstruction Multi model")
#         plt.colorbar(); plt.show()
#         
#         #self.Multi_linear_object.rec_sing for the potentials averaged per FV cell
# =============================================================================
        self.Multi_linear_object=copy.deepcopy(n)
        #self.Multi_linear_object.rec_sing for the potentials averaged per FV cell
        if Metab:
            M, phi_0=Metab
            n.Full_Newton(np.ndarray.flatten(n.s_FV_linear) , np.ndarray.flatten(n.q_linear), 
                          self.conver_residual, M, phi_0)
            
            self.Multi_metab_object=copy.deepcopy(n)
            Multi_FV_metab=n.s_FV_metab
            Multi_q_metab=n.q_metab
            
            n.assemble_it_matrices_Sampson(n.s_FV_metab, n.q_metab)
            plt.imshow((n.rec_sing+Multi_FV_metab).reshape(cells,cells), origin='lower', extent=[0,self.L, 0, self.L])
            plt.title("Average value reconstruction Multi model Metabolism")
            plt.colorbar(); plt.show()

            self.Multi_q_metab=Multi_q_metab
            self.Multi_FV_metab=Multi_FV_metab
            return(Multi_FV_metab,Multi_q_metab)
        else:
            return(Multi_FV_linear, Multi_q_linear)
    
    def Reconstruct_Multi(self, non_linear, plot_sources,*FEM_args):
        """If non_linear the reconstruction will be made on the latest non-linear
        simulation (for the arrays self.Multi_q_metab, and self.Multi_FV_metab)
        
        Inside FEM_args there are the FEM_x, FEM_y, arrays where to reconstruct the 
        concentration field"""
        if non_linear:
            obj=self.Multi_metab_object
            s_FV=obj.s_FV_metab
            q=obj.q_metab
        else:
            obj=self.Multi_linear_object
            s_FV=np.ndarray.flatten(obj.s_FV_linear)
            q=obj.q_linear
        
        if not FEM_args: #Cartesian reconstruction:
            a=reconstruction_sans_flux(np.concatenate((s_FV, q)),obj,obj.L,self.ratio,obj.directness)
            a.reconstruction()   
            a.reconstruction_boundaries_short(self.BC_type, self.BC_value)
            a.rec_corners()
            plt.imshow(a.rec_final, origin='lower', vmax=np.max(a.rec_final))
            plt.title("bilinear reconstruction \n coupling model Metabolism")
            plt.colorbar(); plt.show()
            
            self.Multi_rec=a.rec_final
            
            
            toreturn=a.rec_final, a.rec_potentials, a.rec_s_FV
        
        
        if FEM_args:
            FEM_x=FEM_args[0]
            FEM_y=FEM_args[1]
            b=reconstruction_extended_space(self.pos_s, self.Rv, self.h_coarse, self.L, 
                                            self.K_eff, self.D, self.directness)
            b.s_FV=s_FV
            b.q=q
            b.set_up_manual_reconstruction_space(FEM_x, FEM_y)
            b.full_rec(self.C_v_array, self.BC_value, self.BC_type)
            if plot_sources:
                plt.tricontourf(b.FEM_x, b.FEM_y, b.s, levels=100)
                plt.colorbar()
                plt.title("s FEM rec")    
                plt.show()
                plt.tricontourf(b.FEM_x, b.FEM_y, b.SL, levels=100)
                plt.colorbar()
                plt.title("SL FEM rec")
                plt.show()
                plt.tricontourf(b.FEM_x, b.FEM_y, b.DL, levels=100)
                plt.colorbar()
                plt.title("DL FEM rec")
                plt.show()
                plt.tricontourf(b.FEM_x, b.FEM_y, b.s+ b.SL+b.DL, levels=100)
                plt.colorbar()
                plt.title("Full FEM rec")
                plt.show()    
            
            self.SL=b.SL
            self.DL=b.DL
            self.s=b.s
            
            toreturn=b.s+ b.SL+b.DL, b.SL, b.s
            
        array_phi_field_x=np.zeros((len(self.pos_s), len(self.x_fine)))
        array_phi_field_y=np.zeros((len(self.pos_s), len(self.y_fine)))   
        c=0
        for i in self.pos_s:
            r=reconstruction_extended_space(self.pos_s, self.Rv, self.h_coarse, self.L, 
            self.K_eff, self.D, self.directness)
            r.s_FV=s_FV
            r.q=q
            r.set_up_manual_reconstruction_space(i[0]+np.zeros(len(self.x_fine)), self.y_fine)
            r.full_rec(self.C_v_array, self.BC_value, self.BC_type)
            array_phi_field_y[c]=r.s+r.SL+r.DL
            
            r.set_up_manual_reconstruction_space(self.x_fine, i[1]+np.zeros(len(self.y_fine)))
            r.full_rec(self.C_v_array, self.BC_value, self.BC_type)
            array_phi_field_x[c]=r.s+r.SL+r.DL
            
            
            if plot_sources: 
                plt.plot(self.x_fine, array_phi_field_x[c], label='Multi_x')
                plt.legend()
                plt.title("Plot through source center")
                plt.xlabel("x")
                plt.ylabel("$\phi$")
                plt.show()
                
                plt.plot(self.y_fine, array_phi_field_y[c], label='Multi_y')
                plt.legend()
                plt.title("Plot through source center")
                plt.xlabel("y")
                plt.ylabel("$\phi$")
                plt.show()

            c+=1
            self.array_phi_field_x_Multi=array_phi_field_x
            self.array_phi_field_y_Multi=array_phi_field_y
            
        return(toreturn)

def extract_COMSOL_data(directory_COMSOL, args):
    
    """args corresponds to which files need to be extracted"""
    toreturn=[]
    if args[0]:
        #Vessel_tissue exchanges reference    
        q_file=directory_COMSOL+ '/q.txt'
        q=np.array(pandas.read_fwf(q_file, infer_rows=500).columns.astype(float))
        toreturn.append(q)
    
    if args[1]:
        #Concentration field data for the linear problem
        field_file=directory_COMSOL + '/contour.txt'
        df=pandas.read_fwf(field_file, infer_nrows=500)
        ref_data=np.array(df).T #reference 2D data from COMSOL
        FEM_x=ref_data[0]*10**6 #in micrometers
        FEM_y=ref_data[1]*10**6
        FEM_phi=ref_data[2]
        toreturn.append(FEM_phi)
        toreturn.append(FEM_x)
        toreturn.append(FEM_y)
        
        
    if args[2]:
      #Plots of the concentration field along a horizontal and vertial lines passing through the center of the source
      x_file=directory_COMSOL + "/plot_x.txt"
      y_file=directory_COMSOL+ '/plot_y.txt'
      
      FEM_x_1D=np.array(pandas.read_fwf(x_file, infer_rows=500)).T[1]
      x_1D=np.array(pandas.read_fwf(x_file, infer_rows=500)).T[0]*10**6
      FEM_y_1D=np.array(pandas.read_fwf(y_file, infer_rows=500)).T[1]
      y_1D=np.array(pandas.read_fwf(y_file, infer_rows=500)).T[0]*10**6
      
      toreturn.append(FEM_x_1D)
      toreturn.append(FEM_y_1D)
      toreturn.append(x_1D)
      toreturn.append(y_1D)
    
    return(toreturn)

def FEM_to_Cartesian(FEM_x, FEM_y, FEM_phi, x_c, y_c):
    phi_Cart=np.zeros((len(y_c), len(x_c)))
    for i in range(len(y_c)):
        for j in range(len(x_c)):
            dist=(FEM_x-x_c[j])**2+(FEM_y-y_c[i])**2
            phi_Cart[i,j]=FEM_phi[np.argmin(dist)]
    return(phi_Cart)            

import os
import csv


def write_parameters_COMSOL(pos_s, L, alpha, K_0, M):
    """Writes the parameter file for COMSOL"""
    rows=[["L", L], ["alpha", alpha],["R","L/alpha"],["K_com","K_0/(2*pi*R)"],["M",M],["phi_0",0.4]]
    
    for i in range(len(pos_s)):
        rows.append(["x_{}".format(i), np.around(pos_s[i,0], decimals=4)])
        rows.append(["y_{}".format(i), np.around(pos_s[i,1], decimals=4)])
    with open('Parameters.txt', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        for i in rows:
            writer.writerow(i)
    

    