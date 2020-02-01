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
from PIL import Image
from numba import cuda

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

def main():
    train_csv = "../dataset/l8s2-train.csv"
    val_csv = "../dataset/l8s2-val.csv"
    test_csv = "../dataset/l8s2-test.csv"
    for area in ['sf','la']:
        print(area)
        single_csv = "../dataset/l8s2-predict-single-"+area+".csv"

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

        # train_set = Landsat8DatasetHDF5(train_csv,
        #     input_transform = input_transform,
        #     target_transform=target_transform)
        # train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, sampler = LocalRandomSampler(train_set))
        # train_data_loader = DataLoader(dataset=train_set, batch_size=args.batchSize, shuffle=False)

        # val_set = Landsat8DatasetHDF5(val_csv,
        #     input_transform = input_transform,
        #     target_transform=target_transform)
        # val_data_loader = DataLoader(dataset=val_set, batch_size=args.testBatchSize, shuffle=False)

        # test_set = Landsat8DatasetHDF5(test_csv,
        #     input_transform = input_transform,
        #     target_transform=target_transform)
        # test_data_loader = DataLoader(dataset=test_set, batch_size=args.testBatchSize, shuffle=False)

        single_set = Landsat8DatasetHDF5(single_csv,
            input_transform = input_transform,
            target_transform=target_transform)
        single_data_loader = DataLoader(dataset=single_set, batch_size=args.testBatchSize, shuffle=False)
        #====================================================================================================

        # L8
        # means = [0.04847158, 0.05862545, 0.08196981, 0.09406061, 0.18542677, 0.18565391, 0.13468906]
        # sds = [0.03634661, 0.04239037, 0.05852175, 0.07904375, 0.12151979, 0.14369792, 0.11404569]

        # S2
        means = [0.1440263, 0.12583447, 0.12149246, 0.13250135, 0.14868644, 0.18663958, 0.20851527, 0.20700881, 0.22721754, 0.09312758, 0.00213068, 0.23704098, 0.1701286]
        sds = [0.03666846, 0.03787366, 0.05120519, 0.07712212, 0.07912124, 0.08743614, 0.0989072,  0.10019914, 0.10937765, 0.05528788, 0.0028293, 0.1379629, 0.10973044]

        if area == 'la':
            min_val = [0.1106, 0.074, 0.0456, 0.0277, 0.0265, 0.0321, 0.0334, 0.0304, 0.0312, 0.0201, 0.0012, 0.0147, 0.0079]
            max_val = [0.188, 0.1999, 0.2211, 0.2986, 0.3234, 0.3566, 0.3931, 0.4089, 0.4353, 0.2474, 0.0101, 0.5478, 0.4203]
            min_val_l8 = [0.0031000000000000003, 0.0063, 0.0083, 0.0063, 0.021, 0.011600000000000001, 0.006500000000000001]
            max_val_l8 = [0.11840100000000094, 0.1499, 0.2217, 0.29660000000000003, 0.439, 0.5736, 0.4456]

        elif area == 'sf':
            min_val = [0.1151, 0.0776, 0.0494, 0.0255, 0.0207, 0.016800000000000002, 0.0142, 0.0111, 0.0094, 0.0048, 0.0005, 0.0011, 0.0004]
            max_val = [0.1948, 0.185, 0.1744, 0.1801, 0.1787, 0.2761, 0.3276, 0.3404, 0.36, 0.11750100000000092, 0.0042, 0.2739, 0.1931]
            min_val_l8 = [0.0001, 0.007, 0.012400000000000001, 0.0048000000000000004, 0.0015, 0.0007, 0.0002]
            max_val_l8 = [0.1131, 0.1293, 0.155, 0.165, 0.3501, 0.2786, 0.19920000000000002]

        # modelname = 'SubPixelCNN'
        # modelname = 'SubPixelMaxPoolCNN'
        # modelname = 'TransConvCNN'
        # modelname = 'TransConvMaxPoolCNN'
        for modelname in ['SubPixelCNN','SubPixelMaxPoolCNN','TransConvCNN','TransConvMaxPoolCNN']:
            print(modelname)
            model = torch.load('save/'+modelname+'/model_path.pth')

            if modelname=='SubPixelCNN':
                modelShortName = 'sub-stride'
            elif modelname == 'SubPixelMaxPoolCNN':
                modelShortName = 'sub-max'
            elif modelname == 'TransConvCNN':
                modelShortName = 'trans-stride'
            elif modelname == 'TransConvMaxPoolCNN':
                modelShortName = 'trans-max'

            # model = torch.load('save/SubPixelCNN/model_path.pth')
            # model = torch.load('save/SubPixelMaxPoolCNN/model_path.pth')
            # model = torch.load('save/TransConvCNN/model_path.pth')
            # model = torch.load('save/TransConvMaxPoolCNN/model_path.pth')
            if area == 'la':
                s2_path = 'sentinel2fim/la2017/S2A_MSIL1C_20171230T183751_N0206_R027_T11SLU_20171230T202151/T11SLU_20171230T183751'
            elif area == 'sf':
                s2_path = 'sentinel2fim/sf2018/S2B_MSIL1C_20180209T185519_N0206_R113_T10SEG_20180209T204300/T10SEG_20180209T185519'

            model.eval()

            iter_loader = iter(single_data_loader)
            for i in range(1):
                input, target = next(iter_loader)
            out = model(input.cuda())
            denorm = DenormalizeS2(means, sds)
            out_denorm = denorm(out)
            patch01 = out_denorm[0].reshape(480, 480) 
            patch09 = out_denorm[9].reshape(480, 480)
            patch10 = out_denorm[10].reshape(480, 480)

            patch05 = out_denorm[4].reshape(1440, 1440) 
            patch06 = out_denorm[5].reshape(1440, 1440) 
            patch07 = out_denorm[6].reshape(1440, 1440) 
            patch8A = out_denorm[8].reshape(1440, 1440)
            patch11 = out_denorm[11].reshape(1440, 1440)
            patch12 = out_denorm[12].reshape(1440, 1440)

            patch02 = out_denorm[1].reshape(2880, 2880) 
            patch03 = out_denorm[2].reshape(2880, 2880) 
            patch04 = out_denorm[3].reshape(2880, 2880) 
            patch08 = out_denorm[7].reshape(2880, 2880) 

            # max_val = [patch01.max(), patch02.max(), patch03.max(), patch04.max(), patch05.max(), patch06.max(), patch07.max(), patch08.max(), patch8A.max(), patch09.max(), patch10.max(), patch11.max(), patch12.max()]
            # min_val = [patch01.min(), patch02.min(), patch03.min(), patch04.min(), patch05.min(), patch06.min(), patch07.min(), patch08.min(), patch8A.min(), patch09.min(), patch10.min(), patch11.min(), patch12.min()]

            #====================================================================================================
            # 01, 09, 10
            #====================================================================================================
            
            #la
            xstart = 450
            ystart = 450
            
            #sf
            xstart = 450
            ystart = 450

            geotransform = (300000.0+(60.0*xstart), 60.0, 0.0, 3900000.0-(60.0*ystart), 0.0, -60.0)

            # 01
            print("Predicting B01")
            nx = patch01.shape[0]
            ny = patch01.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B01.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B01.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch01)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch01.min()
            maxval = patch01.max()
            patch01 = ((patch01.astype(np.float32)-min_val[0])/(max_val[0]-min_val[0]))*255
            patch01[patch01>255] = 255
            patch01[patch01<0] = 0
            patch01 = patch01.round().astype(np.uint8)
            pil_img = Image.fromarray(patch01)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B01.jpg')


            # 09
            print("Predicting B09")
            nx = patch09.shape[0]
            ny = patch09.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B09.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B09.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch09)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch09.min()
            maxval = patch09.max()
            patch09 = ((patch09.astype(np.float32)-min_val[9])/(max_val[9]-min_val[9]))*255
            patch09[patch09>255] = 255
            patch09[patch09<0] = 0
            patch09 = patch09.round().astype(np.uint8)
            pil_img = Image.fromarray(patch09)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B09.jpg')



            # 10
            print("Predicting B10")
            nx = patch10.shape[0]
            ny = patch10.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B10.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B10.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch10)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch10.min()
            maxval = patch10.max()
            patch10 = ((patch10.astype(np.float32)-min_val[10])/(max_val[10]-min_val[10]))*255
            patch10[patch10>255] = 255
            patch10[patch10<0] = 0
            patch10 = patch10.round().astype(np.uint8)
            pil_img = Image.fromarray(patch10)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B10.jpg')



            #====================================================================================================
            # 02, 03, 04, 08
            #====================================================================================================
            
            #la
            xstart = 2700
            ystart = 2700

            #sf
            xstart = 2700
            ystart = 2700

            geotransform = (300000.0+(10.0*xstart), 10.0, 0.0, 3900000.0-(10.0*ystart), 0.0, -10.0)

            # 02
            print("Predicting B02")
            nx = patch02.shape[0]
            ny = patch02.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B02.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B02.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch02)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch02.min()
            maxval = patch02.max()
            patch02 = ((patch02.astype(np.float32)-min_val[1])/(max_val[1]-min_val[1]))*255
            patch02[patch02>255] = 255
            patch02[patch02<0] = 0
            patch02 = patch02.round().astype(np.uint8)
            pil_img = Image.fromarray(patch02)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B02.jpg')

            # 03
            print("Predicting B03")
            nx = patch03.shape[0]
            ny = patch03.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B03.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B03.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch03)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch03.min()
            maxval = patch03.max()
            patch03 = ((patch03.astype(np.float32)-min_val[2])/(max_val[2]-min_val[2]))*255
            patch03[patch03>255] = 255
            patch03[patch03<0] = 0
            patch03 = patch03.round().astype(np.uint8)
            pil_img = Image.fromarray(patch03)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B03.jpg')


            # 04
            print("Predicting B04")
            nx = patch04.shape[0]
            ny = patch04.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B04.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B04.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch04)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch04.min()
            maxval = patch04.max()
            patch04 = ((patch04.astype(np.float32)-min_val[3])/(max_val[3]-min_val[3]))*255
            patch04[patch04>255] = 255
            patch04[patch04<0] = 0
            patch04 = patch04.round().astype(np.uint8)
            pil_img = Image.fromarray(patch04)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B04.jpg')


            # 08
            print("Predicting B08")
            nx = patch08.shape[0]
            ny = patch08.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B08.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B08.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch08)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch08.min()
            maxval = patch08.max()
            patch08 = ((patch08.astype(np.float32)-min_val[7])/(max_val[7]-min_val[7]))*255
            patch08[patch08>255] = 255
            patch08[patch08<0] = 0
            patch08 = patch08.round().astype(np.uint8)
            pil_img = Image.fromarray(patch08)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B08.jpg')


            #====================================================================================================
            # 05, 06, 07, 8A, 11, 12
            #====================================================================================================
            
            #la
            xstart = 1350
            ystart = 1350

            #sf
            xstart = 1350
            ystart = 1350

            geotransform = (300000.0+(20.0*xstart), 20.0, 0.0, 3900000.0-(20.0*ystart), 0.0, -20.0)


            # 05
            print("Predicting B05")
            nx = patch05.shape[0]
            ny = patch05.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B05.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B05.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch05)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch05.min()
            maxval = patch05.max()
            patch05 = ((patch05.astype(np.float32)-min_val[4])/(max_val[4]-min_val[4]))*255
            patch05[patch05>255] = 255
            patch05[patch05<0] = 0
            patch05 = patch05.round().astype(np.uint8)
            pil_img = Image.fromarray(patch05)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B05.jpg')


            # 06
            print("Predicting B06")
            nx = patch06.shape[0]
            ny = patch06.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B06.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B06.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch06)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch06.min()
            maxval = patch06.max()
            patch06 = ((patch06.astype(np.float32)-min_val[5])/(max_val[5]-min_val[5]))*255
            patch06[patch06>255] = 255
            patch06[patch06<0] = 0
            patch06 = patch06.astype(np.uint8)
            pil_img = Image.fromarray(patch06)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B06.jpg')



            # 07
            print("Predicting B07")
            nx = patch07.shape[0]
            ny = patch07.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B07.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B07.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch07)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch07.min()
            maxval = patch07.max()
            patch07 = ((patch07.astype(np.float32)-min_val[6])/(max_val[6]-min_val[6]))*255
            patch07[patch07>255] = 255
            patch07[patch07<0] = 0
            patch07 = patch07.round().astype(np.uint8)
            pil_img = Image.fromarray(patch07)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B07.jpg')



            # 8A
            print("Predicting B8A")
            nx = patch8A.shape[0]
            ny = patch8A.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B8A.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B8A.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch8A)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch8A.min()
            maxval = patch8A.max()
            patch8A = ((patch8A.astype(np.float32)-min_val[8])/(max_val[8]-min_val[8]))*255
            patch8A[patch8A>255] = 255
            patch8A[patch8A<0] = 0
            patch8A = patch8A.round().astype(np.uint8)
            pil_img = Image.fromarray(patch07)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B8A.jpg')



            # 11
            print("Predicting B11")
            nx = patch11.shape[0]
            ny = patch11.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B11.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B11.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch11)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch11.min()
            maxval = patch11.max()
            patch11 = ((patch11.astype(np.float32)-min_val[11])/(max_val[11]-min_val[11]))*255
            patch11[patch11>255] = 255
            patch11[patch11<0] = 0
            patch11 = patch11.round().astype(np.uint8)
            pil_img = Image.fromarray(patch11)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B11.jpg')



            # 12
            print("Predicting B12")
            nx = patch12.shape[0]
            ny = patch12.shape[1]

            ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B12.tif')
            img = np.array(ds.GetRasterBand(1).ReadAsArray())
            projection = ds.GetProjection()

            dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+area+'-'+modelShortName+'-B12.tif', ny, nx, 1, gdal.GDT_Float64)
            dst_ds.SetGeoTransform(geotransform)    # specify coords
            srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            dst_ds.GetRasterBand(1).WriteArray(patch12)   # write band to the raster            
            dst_ds.FlushCache()                     # write to disk
            dst_ds = None                           # save, close  

            minval = patch12.min()
            maxval = patch12.max()
            # if area=='sf':
            #     import IPython; IPython.embed()
            patch12 = ((patch12.astype(np.float32)-min_val[12])/(max_val[12]-min_val[12]))*255
            patch12[patch12>255] = 255
            patch12[patch12<0] = 0
            patch12 = patch12.round().astype(np.uint8)
            pil_img = Image.fromarray(patch12)
            pil_img.save('../save/jpg/'+area+'-'+modelShortName+'-B12.jpg')

            del model
            del input
            del out
            torch.cuda.empty_cache()
            # cuda.select_device(0)
            # cuda.close()


if __name__ == '__main__':
    main()