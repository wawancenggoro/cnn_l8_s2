import torch
from torch.utils.data import DataLoader
from dataset.dataset import Landsat8Dataset, Landsat8DatasetHDF5
from dataset.dataset import LocalRandomSampler
from dataset.customTransform import DenormalizeS2, ConvertS2WithoutDenormalization, DenormalizeL8
from torchvision import transforms

import argparse
from IPython import embed

import gc
import gdal
from gdalconst import GA_ReadOnly
from osgeo import osr

import numpy as np
from PIL import Image
from math import log10

import pickle

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='sub', help='choose which model is going to use')

args = parser.parse_args()

def nn_interpolate(A, new_size):
    """Vectorized Nearest Neighbor Interpolation"""

    old_size = A.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)

    # row wise interpolation 
    row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)

    # column wise interpolation
    col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)

    final_matrix = A[:, row_idx][col_idx, :]

    return final_matrix

def main():
    train_csv = "../dataset/l8s2-train.csv"
    val_csv = "../dataset/l8s2-val.csv"
    test_csv = "../dataset/l8s2-test.csv"

    #====================================================================================================
    # Dataloader with HDF5
    #====================================================================================================
    input_transform = transforms.Compose([
                            transforms.ToTensor()
                        ])

    target_transform = transforms.Compose([
                            transforms.Lambda(lambda x: [x[i].astype('float32') for i in range(13)]),
                            transforms.Lambda(lambda x: [transforms.ToTensor()(x[i]) for i in range(13)])
                        ])

    train_set = Landsat8DatasetHDF5(train_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.testBatchSize, shuffle=False)

    val_set = Landsat8DatasetHDF5(val_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    val_data_loader = DataLoader(dataset=val_set, batch_size=args.testBatchSize, shuffle=False)

    test_set = Landsat8DatasetHDF5(test_csv,
        input_transform = input_transform,
        target_transform=target_transform)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)
    #====================================================================================================

    # L8
    means_l8 = [0.04847158, 0.05862545, 0.08196981, 0.09406061, 0.18542677, 0.18565391, 0.13468906]
    sds_l8 = [0.03634661, 0.04239037, 0.05852175, 0.07904375, 0.12151979, 0.14369792, 0.11404569]

    # S2
    means = [0.1440263, 0.12583447, 0.12149246, 0.13250135, 0.14868644, 0.18663958, 0.20851527, 0.20700881, 0.22721754, 0.09312758, 0.00213068, 0.23704098, 0.1701286]
    sds = [0.03666846, 0.03787366, 0.05120519, 0.07712212, 0.07912124, 0.08743614, 0.0989072,  0.10019914, 0.10937765, 0.05528788, 0.0028293, 0.1379629, 0.10973044]

    denormL8 = DenormalizeL8(means_l8, sds_l8)
    denormS2 = DenormalizeS2(means, sds)


    s2_path = 'sentinel2fim/la2017/S2A_MSIL1C_20171230T183751_N0206_R027_T11SLU_20171230T202151/T11SLU_20171230T183751'

    modelname = 'TransConvMaxPoolCNN'
    model = torch.load('save/'+modelname+'/model_path.pth')
    model.eval()

    for set_name in ['test', 'val', 'train']:
        if set_name=="train":
            iter_loader = iter(train_data_loader)
            setlen = 269
        elif set_name=="val":
            iter_loader = iter(val_data_loader)
            setlen = 89
        elif set_name=="test":
            iter_loader = iter(test_data_loader)
            setlen = 90

        cc = np.zeros((setlen,13))
        
        for i in range(setlen):
            print(set_name+": "+str(i)+" of "+str(setlen))
            input, target = next(iter_loader)

            target_denorm = denormS2(target)
            target_denorm[0] = target_denorm[0].reshape(480, 480) 
            target_denorm[9] = target_denorm[9].reshape(480, 480)
            target_denorm[10] = target_denorm[10].reshape(480, 480)
            target_denorm[4] = target_denorm[4].reshape(1440, 1440) 
            target_denorm[5] = target_denorm[5].reshape(1440, 1440) 
            target_denorm[6] = target_denorm[6].reshape(1440, 1440) 
            target_denorm[8] = target_denorm[8].reshape(1440, 1440)
            target_denorm[11] = target_denorm[11].reshape(1440, 1440)
            target_denorm[12] = target_denorm[12].reshape(1440, 1440)
            target_denorm[1] = target_denorm[1].reshape(2880, 2880) 
            target_denorm[2] = target_denorm[2].reshape(2880, 2880) 
            target_denorm[3] = target_denorm[3].reshape(2880, 2880) 
            target_denorm[7] = target_denorm[7].reshape(2880, 2880)

            geotransform = (300000.0, 20.0, 0.0, 3900000.0, 0.0, -20.0)
            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B05.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('check_data/'+set_name+'/'+str(i)+'gt.tif', 1440, 1440, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(target_denorm[4])   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            out = model(input.cuda())
            out_denorm = denormS2(out)
            out_denorm[0] = out_denorm[0].reshape(480, 480) 
            out_denorm[9] = out_denorm[9].reshape(480, 480)
            out_denorm[10] = out_denorm[10].reshape(480, 480)
            out_denorm[4] = out_denorm[4].reshape(1440, 1440) 
            out_denorm[5] = out_denorm[5].reshape(1440, 1440) 
            out_denorm[6] = out_denorm[6].reshape(1440, 1440) 
            out_denorm[8] = out_denorm[8].reshape(1440, 1440)
            out_denorm[11] = out_denorm[11].reshape(1440, 1440)
            out_denorm[12] = out_denorm[12].reshape(1440, 1440)
            out_denorm[1] = out_denorm[1].reshape(2880, 2880) 
            out_denorm[2] = out_denorm[2].reshape(2880, 2880) 
            out_denorm[3] = out_denorm[3].reshape(2880, 2880) 
            out_denorm[7] = out_denorm[7].reshape(2880, 2880) 

            dst_ds = gdal.GetDriverByName('GTiff').Create('check_data/'+set_name+'/'+str(i)+'_pred.tif', 1440, 1440, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(out_denorm[4])   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            geotransform = (300000.0, 30.0, 0.0, 3900000.0, 0.0, -30.0)
            input_denorm = denormL8(input)
            dst_ds = gdal.GetDriverByName('GTiff').Create('check_data/'+set_name+'/'+str(i)+'_input.tif', 960, 960, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(input_denorm[0,4].data.numpy())   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            for j in range(13):
                pred = out_denorm[j]
                gt = target_denorm[j]
                var_pred = (pred-pred.mean())
                var_gt = (gt-gt.mean())
                cc[i,j] = np.multiply(var_pred, var_gt).sum() / np.sqrt((var_pred**2).sum() * (var_gt**2).sum())

            # import IPython; IPython.embed()
            del out_denorm
            del out

        np.savetxt("check_data/"+set_name+"/cc.csv", cc, delimiter=",")

if __name__ == '__main__':
    main()