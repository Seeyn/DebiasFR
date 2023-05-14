# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline
import os
import sys
sys.path.append('.')
# print(sys.path)
import dbfr.archs
import dbfr.data
import dbfr.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
