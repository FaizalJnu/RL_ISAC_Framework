from flddpg import FLDDPG
import matlab.engine 
import numpy as np
import matplotlib.pyplot as plt
import os

class MultiTargetTrainer:
    def __init__(self):
        print("starting multi target engine..")
        self.eng = matlab.engine.start_matlab()

        