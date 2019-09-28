#!/usr/bin/env python3

import time
import h5py

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def mcmc_to_disk(mcmc, **kwargs):
    """Save the MCMC chain to disk with the metadata supplied in kwargs."""

    # FILENAME AND DATASET NAMES
    tmp = f'{mcmc.mcmc_type()}.hdf5'
    filename = kwargs.pop('filename', tmp)
    tmp = f'{mcmc.mcmc_type()}_{timestamp()}'
    dataset_name = kwargs.pop('dataset_name', tmp)

    # OPEN FILE IN TRUNCATE-AND-WRITE MODE
    f = h5py.File(filename, 'w')

    # CREATE DATASET WITH THE CHAIN AS DATA
    dset = f.create_dataset(dataset_name, data=mcmc.chain())

    # ADD ATTRIBUTES
    for key, value in kwargs.items():
        dset.attrs[key] = value
