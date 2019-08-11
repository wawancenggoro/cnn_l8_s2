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
    single_csv = "../dataset/l8s2-predict-single.csv"

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
    # means = [489.7118, 591.63416, 826.2221, 948.7332, 1858.4872, 1864.6527, 1355.4669]
    # sds = [338.75378, 403.48727, 572.8161, 784.2508, 1208.3722, 1436.1204, 1138.7588]

    # S2
    means = [1440.2627, 1258.3445, 1214.9252, 1325.0135, 1486.8649, 1866.3961, 2085.1528, 2070.0884, 2272.1758, 931.276, 21.306807, 2370.4104, 1701.286]
    sds = [366.68463, 378.73654, 512.0519, 771.2212, 791.2124, 874.36127, 989.072, 1001.9915, 1093.7765, 552.87885, 28.292986, 1379.6288, 1097.3044]

    modelname = 'SubPixelCNN'
    modelname = 'SubPixelMaxPoolCNN'
    modelname = 'TransConvCNN'
    modelname = 'TransConvMaxPoolCNN'
    model = torch.load('save/'+modelname+'/model_path.pth')

    # model = torch.load('save/SubPixelCNN/model_path.pth')
    # model = torch.load('save/SubPixelMaxPoolCNN/model_path.pth')
    # model = torch.load('save/TransConvCNN/model_path.pth')
    # model = torch.load('save/TransConvMaxPoolCNN/model_path.pth')

    s2_path = 'S2A_MSIL1C_20171230T183751_N0206_R027_T11SLU_20171230T202151/T11SLU_20171230T183751'

    model.eval()

    iter_loader = iter(single_data_loader)
    for i in range(1):
        input, target = next(iter_loader)
    out = model(input.cuda())
    denorm = DenormalizeS2(means, sds)
    out_denorm = denorm(out)
    patch01 = out_denorm[0].reshape(500,500) 
    patch09 = out_denorm[9].reshape(500,500)
    patch10 = out_denorm[10].reshape(500,500)

    patch05 = out_denorm[4].reshape(1500,1500) 
    patch06 = out_denorm[5].reshape(1500,1500) 
    patch07 = out_denorm[6].reshape(1500,1500) 
    patch8A = out_denorm[8].reshape(1500,1500)
    patch11 = out_denorm[11].reshape(1500,1500)
    patch12 = out_denorm[12].reshape(1500,1500)

    patch02 = out_denorm[1].reshape(3000,3000) 
    patch03 = out_denorm[2].reshape(3000,3000) 
    patch04 = out_denorm[3].reshape(3000,3000) 
    patch08 = out_denorm[7].reshape(3000,3000) 

    #====================================================================================================
    # 01, 09, 10
    #====================================================================================================
    xstart = 475
    ystart = 475
    geotransform = (300000.0+(60.0*xstart), 60.0, 0.0, 3900000.0-(60.0*ystart), 0.0, -60.0)

    # 01
    print("Predicting B01")
    nx = patch01.shape[0]
    ny = patch01.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B01.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B01.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch01)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch01.min()
    maxval = patch01.max()
    patch01 = ((patch01.astype(np.float32)-minval)/(maxval-minval))*256
    patch01 = patch01.astype(np.uint8)
    pil_img = Image.fromarray(patch01)
    pil_img.save('../save/jpg/'+modelname+'/pred_B01.jpg')


    # 09
    print("Predicting B09")
    nx = patch09.shape[0]
    ny = patch09.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B09.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B09.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch09)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch09.min()
    maxval = patch09.max()
    patch09 = ((patch09.astype(np.float32)-minval)/(maxval-minval))*256
    patch09 = patch09.astype(np.uint8)
    pil_img = Image.fromarray(patch09)
    pil_img.save('../save/jpg/'+modelname+'/pred_B09.jpg')



    # 10
    print("Predicting B10")
    nx = patch10.shape[0]
    ny = patch10.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B10.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B10.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch10)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch10.min()
    maxval = patch10.max()
    patch10 = ((patch10.astype(np.float32)-minval)/(maxval-minval))*256
    patch10 = patch10.astype(np.uint8)
    pil_img = Image.fromarray(patch10)
    pil_img.save('../save/jpg/'+modelname+'/pred_B10.jpg')



    #====================================================================================================
    # 02, 03, 04, 08
    #====================================================================================================
    xstart = 2850
    ystart = 2850
    geotransform = (300000.0+(10.0*xstart), 10.0, 0.0, 3900000.0-(10.0*ystart), 0.0, -10.0)

    # 02
    print("Predicting B02")
    nx = patch02.shape[0]
    ny = patch02.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B02.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B02.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch02)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch02.min()
    maxval = patch02.max()
    patch02 = ((patch02.astype(np.float32)-minval)/(maxval-minval))*256
    patch02 = patch02.astype(np.uint8)
    pil_img = Image.fromarray(patch02)
    pil_img.save('../save/jpg/'+modelname+'/pred_B02.jpg')

    # 03
    print("Predicting B03")
    nx = patch03.shape[0]
    ny = patch03.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B03.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B03.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch03)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch03.min()
    maxval = patch03.max()
    patch03 = ((patch03.astype(np.float32)-minval)/(maxval-minval))*256
    patch03 = patch03.astype(np.uint8)
    pil_img = Image.fromarray(patch03)
    pil_img.save('../save/jpg/'+modelname+'/pred_B03.jpg')


    # 04
    print("Predicting B04")
    nx = patch04.shape[0]
    ny = patch04.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B04.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B04.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch04)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch04.min()
    maxval = patch04.max()
    patch04 = ((patch04.astype(np.float32)-minval)/(maxval-minval))*256
    patch04 = patch04.astype(np.uint8)
    pil_img = Image.fromarray(patch04)
    pil_img.save('../save/jpg/'+modelname+'/pred_B04.jpg')


    # 08
    print("Predicting B08")
    nx = patch08.shape[0]
    ny = patch08.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B08.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B08.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch08)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch08.min()
    maxval = patch08.max()
    patch08 = ((patch08.astype(np.float32)-minval)/(maxval-minval))*256
    patch08 = patch08.astype(np.uint8)
    pil_img = Image.fromarray(patch08)
    pil_img.save('../save/jpg/'+modelname+'/pred_B08.jpg')


    #====================================================================================================
    # 05, 06, 07, 8A, 11, 12
    #====================================================================================================
    xstart = 1425
    ystart = 1425
    geotransform = (300000.0+(20.0*xstart), 20.0, 0.0, 3900000.0-(20.0*ystart), 0.0, -20.0)


    # 05
    print("Predicting B05")
    nx = patch05.shape[0]
    ny = patch05.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B05.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B05.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch05)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch05.min()
    maxval = patch05.max()
    patch05 = ((patch05.astype(np.float32)-minval)/(maxval-minval))*256
    patch05 = patch05.astype(np.uint8)
    pil_img = Image.fromarray(patch05)
    pil_img.save('../save/jpg/'+modelname+'/pred_B05.jpg')


    # 06
    print("Predicting B06")
    nx = patch06.shape[0]
    ny = patch06.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B06.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B06.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch06)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch06.min()
    maxval = patch06.max()
    patch06 = ((patch06.astype(np.float32)-minval)/(maxval-minval))*256
    patch06 = patch06.astype(np.uint8)
    pil_img = Image.fromarray(patch06)
    pil_img.save('../save/jpg/'+modelname+'/pred_B06.jpg')



    # 07
    print("Predicting B07")
    nx = patch07.shape[0]
    ny = patch07.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B07.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B07.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch07)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch07.min()
    maxval = patch07.max()
    patch07 = ((patch07.astype(np.float32)-minval)/(maxval-minval))*256
    patch07 = patch07.astype(np.uint8)
    pil_img = Image.fromarray(patch07)
    pil_img.save('../save/jpg/'+modelname+'/pred_B07.jpg')



    # 8A
    print("Predicting B8A")
    nx = patch8A.shape[0]
    ny = patch8A.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B8A.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B8A.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch8A)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch07.min()
    maxval = patch07.max()
    patch07 = ((patch07.astype(np.float32)-minval)/(maxval-minval))*256
    patch07 = patch07.astype(np.uint8)
    pil_img = Image.fromarray(patch07)
    pil_img.save('../save/jpg/'+modelname+'/pred_B8A.jpg')



    # 11
    print("Predicting B11")
    nx = patch11.shape[0]
    ny = patch11.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B11.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B11.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch11)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch11.min()
    maxval = patch11.max()
    patch11 = ((patch11.astype(np.float32)-minval)/(maxval-minval))*256
    patch11 = patch11.astype(np.uint8)
    pil_img = Image.fromarray(patch11)
    pil_img.save('../save/jpg/'+modelname+'/pred_B11.jpg')



    # 12
    print("Predicting B12")
    nx = patch12.shape[0]
    ny = patch12.shape[1]

    ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/sentinel2fim/la2017/'+s2_path+'_B12.tif')
    img = np.array(ds.GetRasterBand(1).ReadAsArray())
    projection = ds.GetProjection()

    dst_ds = gdal.GetDriverByName('GTiff').Create('../save/tif/'+modelname+'/pred_B12.tif', ny, nx, 1, gdal.GDT_Int16)
    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
    dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(patch12)   # write band to the raster            
    dst_ds.FlushCache()                     # write to disk
    dst_ds = None                           # save, close  

    minval = patch12.min()
    maxval = patch12.max()
    patch12 = ((patch12.astype(np.float32)-minval)/(maxval-minval))*256
    patch12 = patch12.astype(np.uint8)
    pil_img = Image.fromarray(patch12)
    pil_img.save('../save/jpg/'+modelname+'/pred_B12.jpg')


if __name__ == '__main__':
    main()