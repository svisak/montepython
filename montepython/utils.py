#!/usr/bin/env python3

import time
import h5py
import os
import pathlib
import numpy as np

def timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())

def convert_to_seconds(t, unit):
    if unit == 'minutes':
        return t * 60.0
    elif unit == 'hours':
        return t * 3600.0
    elif unit == 'days':
        return t * 86400.0
    else:
        raise ValueError('Unregognized time unit')

def mcmc_to_disk(mcmc, **kwargs):
    """Save the MCMC chain to disk with the metadata supplied in kwargs."""

    # CHECK WHETHER TO TRUNCATE FILE
    mode = kwargs.pop('mode', 'a')

    # PATH; CREATE IF IT DOESN'T EXIST
    default_path = 'h5'
    path = kwargs.pop('path', default_path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    # FILENAME
    default_filename = f'{timestamp()}.h5'
    filename = kwargs.pop('filename', default_filename)

    # OPEN FILE IN READ/WRITE/CREATE MODE
    fullpath = f'{path}/{filename}'
    f = h5py.File(fullpath, mode)

    # DATASET NAME
    dataset_name = kwargs.pop('dataset_name', None)
    if dataset_name is None:
        # NO DATASET NAME SPECIFIED
        # USE DEFAULT DATASET NAME 0,
        # OR "NUMBER OF DATA SETS" IF 0 EXISTS
        tmp = list(f.keys())
        dataset_name = str(len(tmp))

    # CREATE DATASET WITH THE CHAIN AS DATA
    chain = mcmc.chain()
    ndim = chain.shape[1]
    dset = f.get(dataset_name)
    if dset is None:
        dset = f.create_dataset(dataset_name, data=chain, maxshape=(None,ndim))
    else:
        dset.resize(chain.shape)
        dset[...] = chain

    # PRINT INFORMATION
    print(f'Wrote dataset \'{dataset_name}\' to file \'{path}/{filename}\'')

    # ADD ATTRIBUTES
    for key, value in kwargs.items():
        dset.attrs[key] = value

    return (path, filename, dataset_name)

def check_positive_semidefinite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
