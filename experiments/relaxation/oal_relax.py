import sys
import os
import multiprocessing as mp

import ase
from ase.calculators.emt import EMT
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.optimize import BFGS, QuasiNewton
from ase.calculators.vasp import Vasp2 as vasp
import numpy as np

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from amptorch.model import CustomMSELoss
from amptorch.active_learning.atomistic_methods import MDsimulate, Relaxation
from amptorch.active_learning.oal_calc import AMPOnlineCalc



def main(filename):
    Gs = {}
    Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
    Gs["G2_rs_s"] = [0] * 4
    Gs["G4_etas"] = [0.005]
    Gs["G4_zetas"] = [1.0, 4.0]
    Gs["G4_gammas"] = [+1.0, -1]
    Gs["cutoff"] = 6.0

    dft_calculator = vasp(
        prec='Normal',
        algo='Normal',
        ncore=10,
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
    slab = ase.io.read("/home/mshuaibi/slab_vasp.traj")
    images = [slab]

    training_params = {
        "uncertain_tol": 0.03,
        "Gs": Gs,
        "morse": True,
        "forcetraining": True,
        "cores": 10,
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
        "filename": filename,
        "verbose": 1,
        "scheduler": {"policy": "CosineAnnealingWarmRestarts", "params":{"T_0":10, "T_mult":2}}
    }

    structure_optim = Relaxation(slab, BFGS, fmax=0.05, steps=100)
    online_calc = AMPOnlineCalc(
        parent_dataset=images,
        parent_calc=dft_calculator,
        n_ensembles=5,
        n_cores="max",
        training_params=training_params,
    )
    structure_optim.run(online_calc, filename=filename)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy('file_system')

    trial = int(sys.argv[1])
    # filedir = f"oal_trial_scheduler_0.03_{trial+1}"
    filedir = f"test_{trial+1}"
    os.makedirs(filedir, exist_ok=True)
    os.chdir(filedir)

    name = f"oal_relax_0.03"
    main(name)
    os.system("rm -rf amp-* results*")
