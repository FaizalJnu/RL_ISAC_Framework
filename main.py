import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self):
        super().__init__()
        self.fc = 28e9
        self.B = 20e6
        self.Ns = 10
        self.Nt = 4
        self.Nr = 64
        self.Nx = 8
        self.Ny = 8
        self.Mb = 64
        self.minrate = 60
        self.speed = 10
        self.endx = 1000
        self.alpha_nl = 2.2
        self.alpha_l = 3.2
        self.rho_l = 3
        self.rho.nl = 4
        self.starting_pos = [500,500,0]
        self.bs_loc = [900,100,20]
        self.ris_loc = [200,300,40]
        self.step_count = 0;
        self.time = 0;
        self.arrival_threshold = 10
        self.env_dims = [1000,1000,100]
        self.dt = 0.1
        
        # TODO: Calculated parameters
        self.h_l = 0
        self.h_nl = 0
        self.gamma_c = 0
        self.SNR = 0
        self.cc = 0
        self.rate = 0
        self.destination = [np.random.randint(990,1000), np.random.randint(990,1000), 0]
        
        # TODO: Channels
        self.H_bt
        self.H_br
        self.H_rt
        self.phi
    
    def calculate_params(self):
        K_db = 4
        k = 10**(K_db/10)
        sigma = np.sqrt(1/(2*(k+1)));
        mu = np.sqrt(k/(k+1));
        
        self.h_l = sigma*np.random.randn(1) + 1j*sigma*np.random.randn(1) + mu
        self.h_nl = sigma*np.random.randn(1) + 1j*sigma*np.random.randn(1) + mu
        
        HLos = self.generate_HLos(self, self.H_bt, self.Nt, self.Nr, self.Nb)
    
    def generate_HLos(self, H_bt, Nt, Nr, Nb):
        K_db = 4
        k = 10**(K_db/10)
        sigma = np.sqrt(1/(2*(k+1)));
        mu = np.sqrt(k/(k+1));
        
        gamma_l = np.sqrt(Nb*Nt)/np.sqrt(self.rho_l)
        HLos = np.zeros(Nr, Nt, Nb)
        for i in range(Nr):
            for j in range(Nt):
                for k in range(Nb):
                    HLos[i,j,k] = H_bt[i,j,k]
        return HLos
    
    def generate_HLos(self, H_bt, Nt, Nr, Nb):
        K_db = 4
        k = 10**(K_db/10)
        sigma = np.sqrt(1/(2*(k+1)));
        mu = np.sqrt(k/(k+1));
        
        gamma_l = np.sqrt(Nb*Nt)/np.sqrt(self.rho_l)
        
        tau_l 
        HLos = np.zeros(Nr, Nt, Nb)
        for i in range(Nr):
            for j in range(Nt):
                for k in range(Nb):
                    HLos[i,j,k] = H_bt[i,j,k]
        return HLos
    
    def step(self,action):
        self.step_count += 1
        self.time += self.dt
        self.destination = [self.starting_pos[0] + self.speed*self.time, self.starting_pos[1] + self.speed*self.time, 0]
        