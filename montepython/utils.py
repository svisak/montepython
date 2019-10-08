#!/usr/bin/env python3

import time
import h5py
import os

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def mcmc_to_disk(mcmc, **kwargs):
    """Save the MCMC chain to disk with the metadata supplied in kwargs."""

    # PATH; CREATE IF IT DOESN'T EXIST
    default_path = 'hdf5'
    path = kwargs.pop('path', default_path)
    try:
        os.mkdir(path)
        print(f'Directory \'{path}\' does not exist, creating it')
    except FileExistsError:
        pass

    # FILENAME
    default_filename = f'{mcmc.mcmc_type()}.hdf5'
    filename = kwargs.pop('filename', default_filename)

    # OPEN FILE IN READ/WRITE/CREATE MODE
    f = h5py.File(f'{path}/{filename}', 'a')

    # DATASET NAME
    default_dataset_name = f'{mcmc.mcmc_type()}_{timestamp()}'
    dataset_name = kwargs.pop('dataset_name', default_dataset_name)

    # CREATE DATASET WITH THE CHAIN AS DATA
    try:
        dset = f.create_dataset(dataset_name, data=mcmc.chain())
    except OSError:
        # DATASET ALREADY EXISTS, USE DEFAULT AS BACKUP
        print(f'Dataset \'{dataset_name}\' already exists, using \'{default_dataset_name}\' instead')
        dataset_name = default_dataset_name
        dset = f.create_dataset(dataset_name, data=mcmc.chain())

    # PRINT INFORMATION
    print(f'Wrote dataset \'{dataset_name}\' to file \'{path}/{filename}\'')

    # ADD ATTRIBUTES
    for key, value in kwargs.items():
        dset.attrs[key] = value
