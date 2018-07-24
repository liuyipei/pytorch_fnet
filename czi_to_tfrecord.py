
"""Read images Allen Institute's from Carl Zeiss(r) ZISRAW (CZI) from their project:
Label-free prediction of three-dimensional fluorescence images from transmitted light microscopy
https://github.com/AllenCellModeling/pytorch_fnet

CZI is the native image file format of the ZEN(r) software by Carl Zeiss
Microscopy GmbH. It stores multidimensional images and metadata from
microscopy experiments.

This script will:
1. Read their czi image data and csv metadata as is provided for a single datasettype (e.g. dna)
2. Extract each signal/target channel pair of a single czi
3. Apply constant transformations: zoom out spatially by a factor of 0.37241, then 0 mean 1 std normalize
4. Save the results into a tfrecord

python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/LMlow/train.csv --tfrecord_outfile=tfrecords/LMlow-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/LMlow/test.csv --tfrecord_outfile=tfrecords/LMlow-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/registration_2/train.csv --tfrecord_outfile=tfrecords/registration_2-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/registration_2/test.csv --tfrecord_outfile=tfrecords/registration_2-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/sec61_beta/train.csv --tfrecord_outfile=tfrecords/sec61_beta-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/sec61_beta/test.csv --tfrecord_outfile=tfrecords/sec61_beta-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/myosin_iib/train.csv --tfrecord_outfile=tfrecords/myosin_iib-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/myosin_iib/test.csv --tfrecord_outfile=tfrecords/myosin_iib-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/zo1/train.csv --tfrecord_outfile=tfrecords/zo1-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/zo1/test.csv --tfrecord_outfile=tfrecords/zo1-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/dic_lamin_b1/train.csv --tfrecord_outfile=tfrecords/dic_lamin_b1-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/dic_lamin_b1/test.csv --tfrecord_outfile=tfrecords/dic_lamin_b1-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/st6gal1/train.csv --tfrecord_outfile=tfrecords/st6gal1-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/st6gal1/test.csv --tfrecord_outfile=tfrecords/st6gal1-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/lamin_b1/train.csv --tfrecord_outfile=tfrecords/lamin_b1-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/lamin_b1/test.csv --tfrecord_outfile=tfrecords/lamin_b1-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/dna/train.csv --tfrecord_outfile=tfrecords/dna-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/dna/test.csv --tfrecord_outfile=tfrecords/dna-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/at_em/train.csv --tfrecord_outfile=tfrecords/at_em-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/at_em/test.csv --tfrecord_outfile=tfrecords/at_em-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/timelapse_wt2_s2/train.csv --tfrecord_outfile=tfrecords/timelapse_wt2_s2-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/timelapse_wt2_s2/test.csv --tfrecord_outfile=tfrecords/timelapse_wt2_s2-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/desmoplakin/train.csv --tfrecord_outfile=tfrecords/desmoplakin-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/desmoplakin/test.csv --tfrecord_outfile=tfrecords/desmoplakin-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/cross_modal_registration_error_NMCC_v2/train.csv --tfrecord_outfile=tfrecords/cross_modal_registration_error_NMCC_v2-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/cross_modal_registration_error_NMCC_v2/test.csv --tfrecord_outfile=tfrecords/cross_modal_registration_error_NMCC_v2-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/alpha_tubulin/train.csv --tfrecord_outfile=tfrecords/alpha_tubulin-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/alpha_tubulin/test.csv --tfrecord_outfile=tfrecords/alpha_tubulin-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/test_run/train.csv --tfrecord_outfile=tfrecords/test_run-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/test_run/test.csv --tfrecord_outfile=tfrecords/test_run-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/membrane_caax_63x/train.csv --tfrecord_outfile=tfrecords/membrane_caax_63x-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/membrane_caax_63x/test.csv --tfrecord_outfile=tfrecords/membrane_caax_63x-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/LMmed/train.csv --tfrecord_outfile=tfrecords/LMmed-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/LMmed/test.csv --tfrecord_outfile=tfrecords/LMmed-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/fibrillarin/train.csv --tfrecord_outfile=tfrecords/fibrillarin-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/fibrillarin/test.csv --tfrecord_outfile=tfrecords/fibrillarin-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/beta_actin/train.csv --tfrecord_outfile=tfrecords/beta_actin-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/beta_actin/test.csv --tfrecord_outfile=tfrecords/beta_actin-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/tom20/train.csv --tfrecord_outfile=tfrecords/tom20-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/tom20/test.csv --tfrecord_outfile=tfrecords/tom20-test.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/registration/train.csv --tfrecord_outfile=tfrecords/registration-train.tfrecord
python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/registration/test.csv --tfrecord_outfile=tfrecords/registration-test.tfrecord

python3 czi_to_tfrecord.py --path_dataset_csv=data/csvs/dna/train.csv --tfrecord_outfile=temp-dna.tfrecord
python3 ~/jobscripts/general/condacpu_sbatch.py --cmdlist=CMDLIST=czi-to-tfr2.cmds --wd=`pwd`
"""


import argparse
import pytorch_fnet.fnet.data
import pytorch_fnet.fnet.fnet_model
import pytorch_fnet.fnet.transforms as transforms
from pytorch_fnet.fnet.data.czireader import CziReader
import numpy as np
import os
import sys
import warnings
import tensorflow as tf

# from pytorch_fnet.fnet.data.fnetdataset import FnetDataset
import pandas as pd
import numpy as np


class CziNumpyDataset(object):
    """Dataset for CZI files."""

    def __init__(self, dataframe: pd.DataFrame = None,
                 path_csv: str = None, 
                 transform_source = [transforms.normalize],
                 transform_target = None):

        if dataframe is not None:
            self.df = dataframe
        else:
            self.df = pd.read_csv(path_csv)
            
        self.transform_source = transform_source
        self.transform_target = transform_target
        print('transform_source', self.transform_source)
        print('transform_target', self.transform_target)
        
        assert all(i in self.df.columns for i in ['path_czi', 'channel_signal', 'channel_target'])

    def __getitem__(self, index, return_ndarrays=True):
        element = self.df.iloc[index, :]
        has_target = not np.isnan(element['channel_target'])
        czi = CziReader(element['path_czi'])
        
        im_out = list()
        channel_signal = czi.get_volume(element['channel_signal'])
        im_out.append(channel_signal)
        print('channel_signal shape raw', channel_signal.shape)
        if has_target:
            im_out.append(czi.get_volume(element['channel_target']))
        
        if self.transform_source is not None:
            for t in self.transform_source: 
                im_out[0] = t(im_out[0])

        if has_target and self.transform_target is not None:
            for t in self.transform_target:
                im_out[1] = t(im_out[1])
                 
        if not return_ndarrays:
            import torch.utils.data
            import torch
            im_out = [torch.from_numpy(im.astype(float)).float() for im in im_out]

            #unsqueeze to make the first dimension be the channel dimension
            im_out = [torch.unsqueeze(im, 0) for im in im_out]
        
        return im_out
    
    def __len__(self):
        return len(self.df)

    def get_information(self, index: int) -> dict:
        return self.df.iloc[index, :].to_dict()


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    factor_yx = 0.37241 # 0.108 um/px -> 0.29 um/px
    default_resizer_str = 'fnet.transforms.Resizer((1, {:f}, {:f}))'.format(factor_yx, factor_yx)
    parser.add_argument('--path_dataset_csv', type=str, default="data/csvs/dna/train.csv", help='path to csv for constructing Dataset')
    parser.add_argument('--transform_signal', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset signal')
    parser.add_argument('--transform_target', nargs='+', default=['fnet.transforms.normalize', default_resizer_str], help='list of transforms on Dataset target')
    parser.add_argument('--tfrecord_outfile', default='from_czi.tfrecord', help='filename of the tfrecord file to write to')

    opts = parser.parse_args()
    print(opts)
    ds = CziNumpyDataset(
        path_csv = opts.path_dataset_csv,
        transform_source = [eval(t) for t in opts.transform_signal],
        transform_target = [eval(t) for t in opts.transform_target],
    )

    if len(ds) == 0:
        print("empty ds", args.path_dataset_csv, 'quitting..')
    else:
        # open the TFRecords file
        writer = tf.python_io.TFRecordWriter(opts.tfrecord_outfile)

        for i, (signal, target) in enumerate(ds):
            assert signal.dtype == np.float32 or signal.dtype == np.float64
            assert target.dtype == np.float32 or signal.dtype == np.float64
            signal = signal.astype(np.float32)
            target = target.astype(np.float32)
            print(i, signal.shape, signal.dtype, target.shape, target.dtype)
            shape = signal.shape
            assert signal.shape == target.shape
            assert len(shape) == 3
            print(i, ds.df.at[i, 'path_czi'])
            print('signal, target', (signal.mean(), np.var(signal)), (target.mean(), np.var(target)))
            feature = {'train/signal': _bytes_feature([tf.compat.as_bytes(signal.tostring())]), # _bytes_feature(signal),
                       'train/target': _bytes_feature([tf.compat.as_bytes(target.tostring())]), # _bytes_feature(target),
                       'train/shape':  _int64_feature(shape)}

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        
        writer.close()
        sys.stdout.flush()