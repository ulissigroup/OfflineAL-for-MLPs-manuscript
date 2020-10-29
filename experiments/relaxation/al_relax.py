import os
import sys
import copy
import numpy as np
import random
import torch

import ase
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS, BFGSLineSearch
from ase.calculators.vasp import Vasp

from amptorch.active_learning.atomistic_methods import MDsimulate, Relaxation
from amptorch.active_learning.learner import AtomisticActiveLearner
from amptorch.active_learning.query_methods import random_query, final_query
from amptorch.model import CustomMSELoss

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    trial = int(sys.argv[1])
    wdir = f"al_relax_final_schedv2_1e-2_trial_{trial}"
    os.makedirs(wdir, exist_ok=True)
    os.chdir(wdir)
    dft_calc = Vasp(
        prec='Normal',
        algo='Normal',
        ncore=4,
        xc='PBE',
        gga='RP',
        lreal=False,
        ediff=1e-4,
        ispin=1,
        nelm=100,
        encut=400,
        lwave=False,
        lcharg=False,
        nsw=0,
        kpts=(4,4,1)
    )

    # Define initial set of images, can be as few as 1. If 1, make sure to
    slab = ase.io.read("/home/mshuaibi/slab_vasp.traj")
    images = [slab]

    # Define symmetry functions
    Gs = {}
    Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
    Gs["G2_rs_s"] = [0] * 4
    Gs["G4_etas"] = [0.005]
    Gs["G4_zetas"] = [1.0, 4.0]
    Gs["G4_gammas"] = [+1.0, -1]
    Gs["cutoff"] = 6.0

    fmax = 0.05

    training_params = {
        "al_convergence": {"method": "final", "force_tol": fmax},
        "samples_to_retrain": 1,
        "Gs": Gs,
        "morse": True,
        "forcetraining": True,
        "cores": 3,
        "optimizer": torch.optim.LBFGS,
        "batch_size": 1000,
        "criterion": CustomMSELoss,
        "num_layers": 3,
        "num_nodes": 20,
        "force_coefficient": 0.04,
        "learning_rate": 1e-2,
        "epochs": 200,
        "test_split": 0,
        "shuffle": False,
        "verbose": 1,
        "filename": f"al_relax_schedulerv2_{fmax}",
        "file_dir": f"./",
        "scheduler": {"policy": "CosineAnnealingWarmRestarts", "params":{"T_0":10, "T_mult":2}}
    }

    # Define AL calculator
    learner = AtomisticActiveLearner(
        training_data=images,
        training_params=training_params,
        parent_calc=dft_calc,
        ensemble=False,
    )
    learner.learn(
        atomistic_method=Relaxation(
            initial_geometry=slab,
            optimizer=BFGS,
            fmax=fmax,
            steps=200,
        ),
        query_strategy=final_query,
    )
    os.system("rm -rf amp-* results*")
