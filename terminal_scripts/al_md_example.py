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
from amptorch.active_learning.query_methods import random_query, max_uncertainty
from amptorch.model import CustomMSELoss

import multiprocessing as mp

if __name__ == "__main__":
    random.seed(1)
    mp.set_start_method("spawn")

    emt_calc = EMT()
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
            kpts=(1,1,1))

    # Define initial set of images, can be as few as 1. If 1, make sure to
    slab = fcc100("Cu", size=(3, 3, 3))
    ads = molecule("CO")
    add_adsorbate(slab, ads, 3, offset=(1, 1))
    cons = FixAtoms(indices=[atom.index for atom in slab if (atom.tag == 3)])
    slab.set_constraint(cons)
    slab.center(vacuum=13.0, axis=2)
    slab.set_pbc(True)
    slab.wrap(pbc=True)
    slab.set_calculator(copy.copy(dft_calc))
    sample_energy = slab.get_potential_energy(apply_constraint=False)
    sample_forces = slab.get_forces(apply_constraint=False)
    slab.set_calculator(sp(atoms=slab, energy=sample_energy, forces=sample_forces))
    ase.io.write("./slab.traj", slab)

    images = [slab]

    # Define symmetry functions
    Gs = {}
    Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
    Gs["G2_rs_s"] = [0] * 4
    Gs["G4_etas"] = [0.005]
    Gs["G4_zetas"] = [1.0, 4.0]
    Gs["G4_gammas"] = [+1.0, -1]
    Gs["cutoff"] = 6.0

    training_params = {
        "al_convergence": {"method": "iter", "num_iterations": 10},
        "samples_to_retrain": 50,
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
        "epochs": 500,
        "test_split": 0,
        "shuffle": False,
        "verbose": 1,
        "filename": "al_md_dft_ex",
        "file_dir": "./al_md_dft_2/",
    }

    # Define AL calculator
    learner = AtomisticActiveLearner(
        training_data=images,
        training_params=training_params,
        parent_calc=dft_calc,
        ensemble=False,
    )
    learner.learn(
        atomistic_method=MDsimulate(
            thermo_ensemble='nvtberendsen',
            dt=1,
            temp=300,
            count=2000,
            initial_geometry=slab
        ),
        query_strategy=random_query,
    )
