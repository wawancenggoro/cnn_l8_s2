import torch
from torch.utils.data import DataLoader
from dataset.dataset import Landsat8Dataset, Landsat8DatasetHDF5
from dataset.dataset import LocalRandomSampler
from dataset.customTransform import DenormalizeS2
from torchvision import transforms

import argparse
from IPython import embed

import gc
import gdal
from gdalconst import GA_ReadOnly
from osgeo import osr

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# hyper-parameters
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')

# model configuration
parser.add_argument('--upscale_factor', '-uf',  type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--model', '-m', type=str, default='sub', help='choose which model is going to use')

args = parser.parse_args()

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
    train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, sampler = LocalRandomSampler(train_set))
    # train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False)

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
    # means = [489.7118, 591.63416, 826.2221, 948.7332, 1858.4872, 1864.6527, 1355.4669]
    # sds = [338.75378, 403.48727, 572.8161, 784.2508, 1208.3722, 1436.1204, 1138.7588]

    # S2
    means = [1440.2627, 1258.3445, 1214.9252, 1325.0135, 1486.8649, 1866.3961, 2085.1528, 2070.0884, 2272.1758, 931.276, 21.306807, 2370.4104, 1701.286]
    sds = [366.68463, 378.73654, 512.0519, 771.2212, 791.2124, 874.36127, 989.072, 1001.9915, 1093.7765, 552.87885, 28.292986, 1379.6288, 1097.3044]

    model = torch.load('model_path.pth')
    model.eval()

    input, target = next(iter(val_data_loader))
    out = model(input.cuda())
    denorm = DenormalizeS2(means, sds)
    out_denorm = denorm(out)
    patch = out_denorm[0].reshape(500,500) 

    nx = patch.shape[0]
    ny = patch.shape[1]

    ds = gdal.Open('dataset/sentinel2/la2017/S2A_MSIL1C_20170404T184421_N0204_R027_T11SLU_20170404T184424/T11SLU_20170404T184421_B01.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    geotransform = (300000.0+(60.0*975), 60.0, 0.0, 3900000.0-(60.0*0), 0.0, -60.0)

    dst_ds = gdal.GetDriverByName('GTiff').Create('test.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    embed()

if __name__ == '__main__':
    main()