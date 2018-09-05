# SchNet - a deep learning architecture for quantum chemistry

SchNet is a deep learning architecture that allows for spatially and chemically 
resolved insights into quantum-mechanical observables of atomistic systems.

This repo is a fork of the original with updates necessary to run with custom structures instead of QM9.

```
module add cudnn/5.1-cuda-8.0
module load anaconda/py35/4.2.0
source activate tensorflow1.1
python scripts/eval.py --split splits10000_64_6 modeldir xyz_5.db . test_idx --batch_size 100
```
 

## Install

    python setup.py install
