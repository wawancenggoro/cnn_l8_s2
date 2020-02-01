from os import listdir
from os.path import join
from os import walk
import os
import time

import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import torch
from PIL import Image

import csv
import gdal

import numpy as np
import pandas as pd
import h5py
from progress_bar import format_time

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

class LocalRandomSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        local = 10

        randperm = torch.randperm(local)

        for i in range(int(n/local)):
            if i==int(n/local)-1:
                randperm = torch.cat((randperm, torch.randperm(n%local)+(local*(i+1))))
            else:
                randperm = torch.cat((randperm, torch.randperm(local)+(local*(i+1))))

        return iter(randperm.tolist())

    def __len__(self):
        return len(self.data_source)

class Landsat8Dataset(data.Dataset):
    def __init__(self, csv_file, input_transform=None, target_transform=None):
        super(Landsat8Dataset, self).__init__()
        # csv_file = "../dataset/landsat8-sentinel2.csv"
        # csv_reader = csv.reader(open(csv_file), delimiter=',')
        # next(csv_reader)
        # self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.data = pd.read_csv(csv_file)

        # for row in csv_reader:
        #     self.l8_filenames.append(row[0])
        #     self.s2_filenames.append(row[1])
        print("Use Dataloader with open from directory")

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):

        # get l8 data
        idx = 1
        path = self.data['landsat8_path'][index]
        print("--Get L8--")
        start = time.time()
        for (dirpath, dirnames, filenames) in walk(os.path.join(os.path.dirname(os.path.realpath(__file__)), path)): 
            for filename in filenames:        
                xstart = self.data['pix_start_x'][index]
                ystart = self.data['pix_start_y'][index]
                patch_size = self.data['patch_size'][index]

                ds = gdal.Open(os.path.join(os.path.dirname(os.path.realpath(__file__)), path, filename))
                img = np.array(ds.GetRasterBand(1).ReadAsArray())
                img = img[ystart:ystart+patch_size, xstart:xstart+patch_size]

                if idx==1:
                    input_image = np.expand_dims(img, axis=2)
                else:
                    input_image = np.concatenate((input_image, np.expand_dims(img, axis=2)), axis=2)

                idx+=1

        if self.input_transform:
            input_image = self.input_transform(input_image)

        end = time.time()
        print('  Time: %s' % format_time(end-start))

        # get s2 data  
        idx = 1
        target = []
        path = self.data['sentinel2_path'][index]
        for (dirpath, dirnames, filenames) in walk(os.path.join(os.path.dirname(os.path.realpath(__file__)), path)): 
            filenames.sort()
            for filename in filenames:  
                xstart = self.data['pix_start_x_'+filename[-7:-4]][index]
                ystart = self.data['pix_start_y_'+filename[-7:-4]][index]
                patch_size = self.data['patch_size_'+filename[-7:-4]][index]

                print("--Get S2 Band {}--".format(idx))
                start = time.time()
                ds = gdal.Open(os.path.join(os.path.dirname(os.path.realpath(__file__)), path, filename))
                img = np.array(ds.GetRasterBand(1).ReadAsArray())
                img = img[ystart:ystart+patch_size, xstart:xstart+patch_size]

                target.append(img)
                end = time.time()
                print('  Time: %s' % format_time(end-start))

                idx+=1

        temp = target[12]
        for i in range(4):
            idx = 12-i
            target[idx] = target[idx-1]
        target[8] = temp

        if self.target_transform:
            target = self.target_transform(target)

        return input_image, target

    def __len__(self):
        return self.data.shape[0]


class Landsat8DatasetHDF5(data.Dataset):
    def __init__(self, csv_file, input_transform=None, target_transform=None):
        super(Landsat8DatasetHDF5, self).__init__()
        print("Use Dataloader with HDF5")

        self.data = pd.read_csv(csv_file)
        self.hfl8 = h5py.File('../dataset/landsat8.h5', 'r')
        # self.dfl8 = self.hfl8['bands']
        self.hfs2 = h5py.File('../dataset/sentinel2.h5', 'r')

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.band_names = [
            '01',
            '02',
            '03',
            '04',
            '05',
            '06',
            '07',
            '08',
            '8A',
            '09',
            '10',
            '11',
            '12'
        ]

        bands_dataset = {
            '01':'band50',
            '02':'band300',
            '03':'band300',
            '04':'band300',
            '05':'band150',
            '06':'band150',
            '07':'band150',
            '08':'band300',
            '8A':'band150',
            '09':'band50',
            '10':'band50',
            '11':'band150',
            '12':'band150'
        }

        bands_num_dataset = {
            '01':0,
            '02':0,
            '03':1,
            '04':2,
            '05':0,
            '06':1,
            '07':2,
            '08':3,
            '8A':3,
            '09':1,
            '10':2,
            '11':4,
            '12':5
        }

    def __getitem__(self, index):

        # get l8 data
        idx = 1
        idx_l8 = self.data['index_l8'][index]
        xstart = self.data['pix_start_x_l8'][index]
        ystart = self.data['pix_start_y_l8'][index]
        patch_size = self.data['patch_size_l8'][index]

        # print("--Get L8--")
        # start = time.time()
        input_image = self.hfl8['bands'][idx_l8][:, ystart:ystart+patch_size, xstart:xstart+patch_size]
        # input_image = self.hfl8['bands'][idx_l8]
        # end = time.time()
        # print('  Index: {} Time: {}'.format(index,format_time(end-start)))
        

        # get s2 data
        idx_s2 = self.data['index_s2'][index]

        # print("--Get S2--")
        # start = time.time()

        # get band50
        xstart = self.data['pix_start_x_s2_60'][index]
        ystart = self.data['pix_start_y_s2_60'][index]
        patch_size = self.data['patch_size_s2_60'][index]
        band50 = self.hfs2['band50'][idx_s2][:, ystart:ystart+patch_size, xstart:xstart+patch_size]
        b01 = np.expand_dims(band50[0], axis=0)
        b09 = np.expand_dims(band50[1], axis=0)
        b10 = np.expand_dims(band50[2], axis=0)
        
        # get band300
        xstart = self.data['pix_start_x_s2_10'][index]
        ystart = self.data['pix_start_y_s2_10'][index]
        patch_size = self.data['patch_size_s2_10'][index]
        bands = self.hfs2['band300'][idx_s2][:, ystart:ystart+patch_size, xstart:xstart+patch_size]
        b02 = np.expand_dims(bands[0], axis=0)
        b03 = np.expand_dims(bands[1], axis=0)
        b04 = np.expand_dims(bands[2], axis=0)
        b08 = np.expand_dims(bands[3], axis=0)

        # get band150
        xstart = self.data['pix_start_x_s2_20'][index]
        ystart = self.data['pix_start_y_s2_20'][index]
        patch_size = self.data['patch_size_s2_20'][index]
        bands = self.hfs2['band150'][idx_s2][:, ystart:ystart+patch_size, xstart:xstart+patch_size]
        b05 = np.expand_dims(bands[0], axis=0)
        b06 = np.expand_dims(bands[1], axis=0)
        b07 = np.expand_dims(bands[2], axis=0)
        b8A = np.expand_dims(bands[3], axis=0)
        b11 = np.expand_dims(bands[4], axis=0)
        b12 = np.expand_dims(bands[5], axis=0)

        target = [
            b01,
            b02,
            b03,
            b04,
            b05,
            b06,
            b07,
            b08,
            b8A,
            b09,
            b10,
            b11,
            b12
        ]

        # end = time.time()
        # print('  Time: %s' % format_time(end-start))
            
        return input_image, target

    def __len__(self):
        return self.data.shape[0]

class Landsat8DatasetHDF5FIm(data.Dataset):
    def __init__(self, csv_file, input_transform=None, target_transform=None):
        super(Landsat8DatasetHDF5FIm, self).__init__()
        print("Use Dataloader with HDF5")

        self.data = pd.read_csv(csv_file)
        self.hfl8 = h5py.File('../dataset/landsat8.h5', 'r')
        # self.dfl8 = self.hfl8['bands']
        self.hfs2 = h5py.File('../dataset/sentinel2.h5', 'r')

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.band_names = [
            '01',
            '02',
            '03',
            '04',
            '05',
            '06',
            '07',
            '08',
            '8A',
            '09',
            '10',
            '11',
            '12'
        ]

        bands_dataset = {
            '01':'band50',
            '02':'band300',
            '03':'band300',
            '04':'band300',
            '05':'band150',
            '06':'band150',
            '07':'band150',
            '08':'band300',
            '8A':'band150',
            '09':'band50',
            '10':'band50',
            '11':'band150',
            '12':'band150'
        }

        bands_num_dataset = {
            '01':0,
            '02':0,
            '03':1,
            '04':2,
            '05':0,
            '06':1,
            '07':2,
            '08':3,
            '8A':3,
            '09':1,
            '10':2,
            '11':4,
            '12':5
        }

    def __getitem__(self, index):

        # get l8 data
        # print("--Get L8--")
        # start = time.time()
        idx = 1
        idx_l8 = self.data['index_l8'][index]
        input_image = self.hfl8['bands'][idx_l8][:, :1000, :1000]
        # end = time.time()
        # print('  Index: {} Time: {}'.format(index,format_time(end-start)))
        
        # get s2 data
        # print("--Get S2--")
        # start = time.time()
        idx_s2 = self.data['index_s2'][index]
        
        # get band50
        bands = self.hfs2['band50'][idx_s2][:, :500, :500]
        b01 = np.expand_dims(bands[0], axis=0)
        b09 = np.expand_dims(bands[1], axis=0)
        b10 = np.expand_dims(bands[2], axis=0)
        
        # get band300
        bands = self.hfs2['band300'][idx_s2][:, :3000, :3000]
        b02 = np.expand_dims(bands[0], axis=0)
        b03 = np.expand_dims(bands[1], axis=0)
        b04 = np.expand_dims(bands[2], axis=0)
        b08 = np.expand_dims(bands[3], axis=0)

        # get band150
        bands = self.hfs2['band150'][idx_s2][:, :1500, :1500]
        b05 = np.expand_dims(bands[0], axis=0)
        b06 = np.expand_dims(bands[1], axis=0)
        b07 = np.expand_dims(bands[2], axis=0)
        b8A = np.expand_dims(bands[3], axis=0)
        b11 = np.expand_dims(bands[4], axis=0)
        b12 = np.expand_dims(bands[5], axis=0)

        target = [
            b01,
            b02,
            b03,
            b04,
            b05,
            b06,
            b07,
            b08,
            b8A,
            b09,
            b10,
            b11,
            b12
        ]
        # end = time.time()
        # print('  Time: %s' % format_time(end-start))
            
        return input_image, target

    def __len__(self):
        return self.data.shape[0]