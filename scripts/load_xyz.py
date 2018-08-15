import argparse
import logging
import os
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from urllib.error import URLError, HTTPError

import numpy as np
from ase.db import connect
from ase.io.extxyz import read_xyz
from ase.units import Hartree, eV, Bohr, Ang
import ase

from schnet.atoms import collect_neighbors
from tqdm import tqdm


def load_data(xyz_filename, targets):

    logging.info('Parse xyz files...')
    materials = ase.io.read(xyz_filename, index=':')
    targets = np.loadtxt(targets)

    i = 0
    with connect('xyz.db') as con:
        for material in tqdm(materials):
            ats = material

            idx_ik, seg_i, idx_j, idx_jk, seg_j, offset, ratio_j = \
                collect_neighbors(ats, 20.)

            data = {'_idx_ik': idx_ik, '_idx_jk': idx_jk, '_idx_j': idx_j,
                    '_seg_i': seg_i, '_seg_j': seg_j, '_offset': offset,
                    '_ratio_j': ratio_j}
            properties = {'target': targets[i]}
            i += 1
            con.write(ats, key_value_pairs=properties, data=data)
            if i == 100:
                break
    logging.info('Done.')

    return True


if __name__ == '__main__':
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser()
    parser.add_argument('xyz', help='Path to xyz file')
    parser.add_argument('targets', help='Path to targets file')
    args = parser.parse_args()

    load_data(args.xyz, args.targets)
