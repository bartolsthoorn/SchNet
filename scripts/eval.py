import argparse
import logging
import os
from functools import partial

import numpy as np
import tensorflow as tf
from schnet.atoms import stats_per_atom
from schnet.data import ASEReader, DataProvider
from schnet.forces import predict_property, calculate_errors, \
    collect_summaries
from schnet.models import SchNet
from schnet.nn.train import EarlyStopping, build_train_op
from tqdm import tqdm

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def eval(model_path, data_path, indices, property, forces, name, batch_size=100, atomref=None):
    tf.reset_default_graph()
    checkpoint_dir = os.path.join(model_path, 'validation')
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    print('ckpt:', ckpt)

    args = np.load(os.path.join(model_path, 'args.npy')).item()

    atomref = None
    #try:
    #    atomref = np.load(atomref)['atom_ref']
    #    if args.energy == 'energy_U0':
    #        atomref = atomref[:, 1:2]
    #    if args.energy == 'energy_U':
    #        atomref = atomref[:, 2:3]
    #    if args.energy == 'enthalpy_H':
    #        atomref = atomref[:, 3:4]
    #    if args.energy == 'free_G':
    #        atomref = atomref[:, 4:5]
    #    if args.energy == 'Cv':
    #        atomref = atomref[:, 5:6]
    #except Exception as e:
    #    print(e)

    # setup data pipeline
    logging.info('Setup data reader')
    fforces = [forces] if forces != 'none' else []
    data_reader = ASEReader(data_path,
                            [property],
                            fforces, [(None, 3)])
    data_provider = DataProvider(data_reader, batch_size, indices,
                                 shuffle=False)
    data_batch = data_provider.get_batch()

    logging.info('Setup model')
    schnet = SchNet(args.interactions, args.basis, args.filters, args.cutoff,
                    atomref=atomref,
                    intensive=args.intensive,
                    filter_pool_mode=args.filter_pool_mode)

    # apply model
    Et = data_batch[property]
    if forces != 'none':
        Ft = data_batch[forces]
    Ep, Fp = predict_property(schnet, data_batch)

    aids = []
    Epred = []
    Fpred = []
    E = []
    F = []

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        data_provider.create_threads(sess, coord)
        schnet.restore(sess, ckpt)

        for i in tqdm(range(len(data_provider) // batch_size)):
            if forces != 'none':
                e, f, ep, fp, aid = sess.run([Et, Ft, Ep, Fp, data_batch['aid']])
                F.append(f)
                Fpred.append(fp)
            else:
                e, ep, aid = sess.run([Et, Ep, data_batch['aid']])
            E.append(e.ravel())
            Epred.append(ep.ravel())
            aids.append(aid)


    E = np.hstack(E)
    aids = np.hstack(aids)
    Epred = np.hstack(Epred)
    if (Epred.min() < 0.0):
        print('Negative bandgap predicted!!!!', Epred.min())
        Epred = np.clip(Epred, 0, None)
    e_mae = np.mean(np.abs(E - Epred))
    e_rmse = np.sqrt(np.mean(np.square(E - Epred)))

    if forces != 'none':
        F = np.vstack(F)
        Fpred = np.vstack(Fpred)
        f_mae = np.mean(np.abs(F - Fpred[:, 0]))
        f_rmse = np.sqrt(np.mean(np.square(F - Fpred[:, 0])))
    else:
        F = None
        Fpred = None
        f_mae = 0.
        f_rmse = 0.
    np.savez(os.path.join(model_path, 'results_' + name + '.npz'),
             F=F, Fpred=Fpred, E=E, Epred=Epred, aids=aids)
    return e_mae, e_rmse, f_mae, f_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path',
                        help='Path to directory with models')
    parser.add_argument('data', help='Path to data')
    parser.add_argument('splitdir', help='directory with data splits')
    parser.add_argument('split', help='train / val / test')
    parser.add_argument('--batch_size', type=int, help='Batch size',
                        default=10)
    parser.add_argument('--property', help='Name of property',
                        default='target')
    parser.add_argument('--forces', help='Enable forces',
                        default='none')
    parser.add_argument('--atomref', help='Atom reference file (NPZ)',
                        default=None)
    parser.add_argument('--splitname', help='Name of data split',
                        default=None)
    args = parser.parse_args()

    with open(os.path.join(args.path, 'errors_' + args.split + '.csv'),
              'w') as f:
        f.write('model,property MAE,property RMSE,force MAE,force RMSE\n')
        for dir in os.listdir(args.path):
            mdir = os.path.join(args.path, dir)
            if not os.path.isdir(mdir):
                continue
            print(mdir)
            if args.splitname is None:
                split_name = '_'.join(dir.split('_')[9:])
            else:
                split_name = args.splitname
            split_file = os.path.join(args.splitdir, split_name + '.npz')
            indices = np.load(split_file)[args.split]
            print(len(indices)//100)
            try:
                res = eval(mdir, args.data, indices,
                        args.property, args.forces, args.split, atomref=args.atomref, batch_size=args.batch_size)
            except Exception as e:
                print(e)
                continue
            res = [str(np.round(r, 8)) for r in res]
            f.write(dir + ',' + ','.join(res) + '\n')
            print(dir, res)
