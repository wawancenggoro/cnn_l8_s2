import torch
from torch.utils.data import DataLoader
from dataset.dataset import Landsat8Dataset, Landsat8DatasetHDF5
from dataset.dataset import LocalRandomSampler
from dataset.customTransform import DenormalizeS2, ConvertS2WithoutDenormalization
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
    test_csv = "../dataset/l8s2-test-clean.csv"
    datestr='20200106'
    test_num_rows = 84

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

    # modelname = 'SubPixelCNN'
    # modelname = 'SubPixelMaxPoolCNN'
    # modelname = 'TransConvCNN'
    # modelname = 'TransConvMaxPoolCNN'
    for modelname in ['sub', 'submax', 'trans', 'transmax']:
        print(modelname)
        model = torch.load('save/'+datestr+'/'+modelname+'/model_path.pth')

        # model = torch.load('save/SubPixelCNN/model_path.pth')
        # model = torch.load('save/SubPixelMaxPoolCNN/model_path.pth')
        # model = torch.load('save/TransConvCNN/model_path.pth')
        # model = torch.load('save/TransConvMaxPoolCNN/model_path.pth')

        model.eval()

        pred_min = np.zeros((test_num_rows,13))
        pred_max = np.zeros((test_num_rows,13))
        pred_mean = np.zeros((test_num_rows,13))
        pred_std = np.zeros((test_num_rows,13))

        gt_min = np.zeros((test_num_rows,13))
        gt_max = np.zeros((test_num_rows,13))
        gt_mean = np.zeros((test_num_rows,13))
        gt_std = np.zeros((test_num_rows,13))

        mae = np.zeros((test_num_rows,13))
        mse = np.zeros((test_num_rows,13))
        rmse = np.zeros((test_num_rows,13))
        psnr = np.zeros((test_num_rows,13))
        ssim = np.zeros((test_num_rows,13))
        cc = np.zeros((test_num_rows,13))
        sam = np.zeros((test_num_rows,2880,2880))
        ergas = np.zeros((test_num_rows,))
        ergas_no10 = np.zeros((test_num_rows,))

        s2_path = 'sentinel2fim/la2017/S2A_MSIL1C_20171230T183751_N0206_R027_T11SLU_20171230T202151/T11SLU_20171230T183751'

        iter_loader = iter(test_data_loader)
        # import IPython; IPython.embed()
        for i in range(test_num_rows):
        # for i in range(1):
            print(i)
            input, target = next(iter_loader)
            # import pdb; pdb.set_trace()
            out = model(input.cuda())

            denorm = DenormalizeS2(means, sds)
            # denorm = ConvertS2WithoutDenormalization()

            out_denorm = denorm(out)
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

            target_denorm = denorm(target)
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

            # geotransform = (300000.0+(20.0*0), 20.0, 0.0, 3900000.0-(20.0*0), 0.0, -20.0)
            # ds = gdal.Open('/mnt/Storage2/Projects/dikti2019PakSani/dataset/'+s2_path+'_B05.tif')
            # img = np.array(ds.GetRasterBand(1).ReadAsArray())
            # projection = ds.GetProjection()

            # dst_ds = gdal.GetDriverByName('GTiff').Create('gt.tif', 1440, 1440, 1, gdal.GDT_Float64)
            # dst_ds.SetGeoTransform(geotransform)    # specify coords
            # srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            # dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            # dst_ds.GetRasterBand(1).WriteArray(target_denorm[4])   # write band to the raster            
            # dst_ds.FlushCache()                     # write to disk
            # dst_ds = None                           # save, close  

            # dst_ds = gdal.GetDriverByName('GTiff').Create('pred.tif', 1440, 1440, 1, gdal.GDT_Float64)
            # dst_ds.SetGeoTransform(geotransform)    # specify coords
            # srs = osr.SpatialReference(wkt=ds.GetProjection())            # establish encoding
            # dst_ds.SetProjection(srs.ExportToWkt()) # export coords to file
            # dst_ds.GetRasterBand(1).WriteArray(out_denorm[4])   # write band to the raster            
            # dst_ds.FlushCache()                     # write to disk
            # dst_ds = None                           # save, close  

            # import IPython; IPython.embed()

            pred_vecs = np.zeros((13,2880,2880))
            gt_vecs = np.zeros((13,2880,2880))

            pred_vecs[1] = out_denorm[1]
            pred_vecs[2] = out_denorm[2]
            pred_vecs[3] = out_denorm[3]
            pred_vecs[7] = out_denorm[7]

            gt_vecs[1] = target_denorm[1]
            gt_vecs[2] = target_denorm[2]
            gt_vecs[3] = target_denorm[3]
            gt_vecs[7] = target_denorm[7]

            ergas_inroot = 0
            ergas_inroot_no10 = 0
            for j in range(13):
                pred = out_denorm[j]
                gt = target_denorm[j]

                pred_min[i,j] = pred.min()
                pred_max[i,j] = pred.max()
                pred_mean[i,j] = pred.mean()
                pred_std[i,j] = pred.std()

                gt_min[i,j] = gt.min()
                gt_max[i,j] = gt.max()
                gt_mean[i,j] = gt.mean()
                gt_std[i,j] = gt.std()

                pred_gt_cov = np.mean((pred-pred_mean[i,j])*(gt-gt_mean[i,j]))

                mae[i,j] = np.abs(pred-gt).mean()
                mse[i,j] = np.square(pred-gt).mean()
                rmse[i,j] = np.sqrt(mse[i,j])

                L = 1.            
                psnr[i,j] = 10 * log10(L**2 / mse[i,j])
                
                k1 = 0.01
                k2 = 0.03
                c1 = (k1*L)**2
                c2 = (k2*L)**2
                ssim[i,j] = ((2*pred_mean[i,j]*gt_mean[i,j] + c1) * (pred_gt_cov + c2)) / ((pred_mean[i,j]**2 + gt_mean[i,j]**2 + c1) * (pred_std[i,j]**2 + gt_std[i,j]**2 + c2))

                var_pred = (pred-pred_mean[i,j])
                var_gt = (gt-gt_mean[i,j])
                cc[i,j] = np.multiply(var_pred, var_gt).sum() / np.sqrt((var_pred**2).sum() * (var_gt**2).sum())

                ergas_inroot += np.square(rmse[i,j]/gt_mean[i,j])
                if j!=10:
                    ergas_inroot_no10 += np.square(rmse[i,j]/gt_mean[i,j])
                # print(i)
                # print(np.square(rmse[i,j]/gt_mean[i,j]))
                # import IPython; IPython.embed()

            # import IPython; IPython.embed()

            # In ergas, ratio = h/l, where h and l is the pixel size of gt and pred. 
            # Because the gt and pred in this case has the same resolution, then the ratio = 1
            # Thus, the actual code is:
            # ================================================================================
            # ratio = 1
            # ergas[i] = 100*ratio*np.sqrt(ergas_inroot/13)
            # ================================================================================
            # But, for efficient computation, we omit the ratio.
            ergas[i] = 100*np.sqrt(ergas_inroot/13)
            ergas_no10[i] = 100*np.sqrt(ergas_inroot_no10/12)

            new_size = (2880, 2880)
            old_size = out_denorm[0].shape
            row_ratio, col_ratio = np.array((new_size))/np.array(old_size)
            row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)
            col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)

            pred_vecs[0] = out_denorm[0][:, row_idx][col_idx, :]
            pred_vecs[9] = out_denorm[9][:, row_idx][col_idx, :]
            pred_vecs[10] = out_denorm[10][:, row_idx][col_idx, :]

            gt_vecs[0] = target_denorm[0][:, row_idx][col_idx, :]
            gt_vecs[9] = target_denorm[9][:, row_idx][col_idx, :]
            gt_vecs[10] = target_denorm[10][:, row_idx][col_idx, :]

            old_size = out_denorm[4].shape
            row_ratio, col_ratio = np.array((new_size))/np.array(old_size)
            row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)
            col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)

            pred_vecs[4] = out_denorm[4][:, row_idx][col_idx, :]
            pred_vecs[5] = out_denorm[5][:, row_idx][col_idx, :]
            pred_vecs[6] = out_denorm[6][:, row_idx][col_idx, :]
            pred_vecs[8] = out_denorm[8][:, row_idx][col_idx, :]
            pred_vecs[11] = out_denorm[11][:, row_idx][col_idx, :]
            pred_vecs[12] = out_denorm[12][:, row_idx][col_idx, :]

            gt_vecs[4] = target_denorm[4][:, row_idx][col_idx, :]
            gt_vecs[5] = target_denorm[5][:, row_idx][col_idx, :]
            gt_vecs[6] = target_denorm[6][:, row_idx][col_idx, :]
            gt_vecs[8] = target_denorm[8][:, row_idx][col_idx, :]
            gt_vecs[11] = target_denorm[11][:, row_idx][col_idx, :]
            gt_vecs[12] = target_denorm[12][:, row_idx][col_idx, :]

            sam[i] = np.arccos((pred_vecs * gt_vecs).sum(axis=0)/(np.sqrt(np.square(pred_vecs).sum(axis=0))*np.sqrt(np.square(gt_vecs).sum(axis=0))))

            # import IPython; IPython.embed()

            del out

        np.savetxt("save/"+datestr+'/'+modelname+"/pred_min.csv", pred_min, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/pred_max.csv", pred_max, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/pred_mean.csv", pred_mean, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/pred_std.csv", pred_std, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/gt_min.csv", gt_min, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/gt_max.csv", gt_max, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/gt_mean.csv", gt_mean, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/gt_std.csv", gt_std, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_mae.csv", mae, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_rmse.csv", rmse, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_psnr.csv", psnr, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_ssim.csv", ssim, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_cc.csv", cc, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_ergas.csv", ergas, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_ergas_no10.csv", ergas_no10, delimiter=",")
        np.savetxt("save/"+datestr+'/'+modelname+"/metric_sam.csv", np.mean(sam, axis=(1,2)), delimiter=",")

        with open("save/"+datestr+'/'+modelname+"/sam.pkl", 'wb') as f:
            pickle.dump(sam, f, protocol=4)


if __name__ == '__main__':
    main()
