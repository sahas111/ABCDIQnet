__author__= 'Susmita Saha'

'''Slice data preparation file'''

import SimpleITK as sitk
import h5py
import glob2
import numpy as np
import tarfile
import shutil
import os
import sys

#Define the command line arguments
slice_start_no=int(sys.argv[1])
slice_end_no=int(sys.argv[2])

#Define the functions
def read_scan_image(img_file, np_dtype):
    # Read SimpleITK scan image
    img_sitk = sitk.ReadImage(img_file)
    # Convert to a NumPy array
    img_np = np.array(sitk.GetArrayFromImage(img_sitk)).astype(np_dtype)
    return img_sitk, img_np

def extract_slice(img_np, lims_slice):
    ''' image informations are stored in y,z,x sequence '''
    slice = (img_np[
             lims_slice['z'][0]: lims_slice['z'][1],
             lims_slice['x'][0]: lims_slice['x'][1],
             lims_slice['y'][0]: lims_slice['y'][1]
             ])
    slice_reshaped = np.transpose(slice, (1, 2, 0))

    return slice_reshaped

#Define the patch dimensions
DimX=240
DimY=240
DimZ=3

#Define the strides
stride_X=int(DimX/2)
stride_Y=int(DimY/2)
stride_Z=DimZ
stride =[stride_Z,stride_X,stride_Y]

#Read the train data
T1_data_train=glob2.glob('/Volumes/{cci-msk}-1/reference/Projects/ABCD/final/ABCDChallenge2019Train/fmriresults01/image03/training/*.tgz')

''' Reading actual/age-corrected Fluid IQ score '''

f=open('/Users/sah012/clinicalData/abcd_tbss01.txt',"r")
lines=f.readlines()
all_ID_FIQ=[]
for x in lines:
    all_ID_FIQ.append(x.split('\t'))
f.close()

all_ID_FIQ=np.array(all_ID_FIQ)

f=open('/Users/sah012/clinicalData/abcd_tbss01.txt',"r")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split('\t'))
f.close()

#Cognition Fluid Composite Age-Corrected Standard Score"
print(result[2][46])
#Cognition Fluid Composite unCorrected Standard Score"
print(result[2][45])

'''Split the data if needed for training'''
# T1_data_train=T1_data_train[0:3739]
#Read the data for all slices
for slice_start_no in range(slice_start_no, slice_end_no, 3):
    actual_FIQ_train_uncorrected = []
    slice_data_train_actualFIQ=[]
    count=0
    for item in T1_data_train:
        count+=1
        print(count)
        tar = tarfile.open(item,'r')
        if not os.path.isdir('/Users/sah012/Desktop/output1/'):
            os.makedirs('/Users/sah012/Desktop/output1/')
        for item1 in tar:
            tar.extract(item1, '/Users/sah012/Desktop/output1/')
            if (item1.name.find("t1_brain.nii"))!=-1:
                img = '/Users/sah012/Desktop/output1/' + str(item1.name)
                img_sitk, img_np = read_scan_image(img, np.float32)

                lims_img = {
                    'z': [60 + slice_start_no, 177],
                    'x': [0 + int(stride[1]), img_np.shape[1] - int(stride[1]) + 1],
                    'y': [0 + int(stride[2]), img_np.shape[2] - int(stride[2]) + 1]}
                j = 0
                for x in range(lims_img['x'][0], lims_img['x'][1], stride[1]):
                    for y in range(lims_img['y'][0], lims_img['y'][1], stride[2]):
                        for z in range(lims_img['z'][0], lims_img['z'][1], DimZ):
                            lims_slice = {'z': [z, z+DimZ],
                                          'x': [x - int(stride[1]), x + int(stride[1])],
                                          'y': [y - int(stride[2]), y + int(stride[2])]
                                          }
                            slice = extract_slice(img_np, lims_slice)

                            for index, i in enumerate(list(all_ID_FIQ[2:-1, 3])):
                                if str(img [30:46]) in str(i) and all_ID_FIQ[index + 2, 45] != '""':
                                    # actual_FIQ_train_uncorrected.append(all_ID_FIQ[index + 2, 46])
                                    actual_FIQ_train_uncorrected.append(all_ID_FIQ[index + 2, 45])
                                    slice_data_train_actualFIQ.append(slice)

                            j = j + 1

                            if j == 1:
                                break
                filelist = glob2.glob('/Users/sah012/Desktop/output1/*/*/*/*')
                for f in filelist:
                    os.remove(os.path.abspath(f))
                shutil.rmtree('/Users/sah012/Desktop/output1/')
                break
            # else:
            #     print('not found')

    slice_data_train_actualFIQ_data = np.array(slice_data_train_actualFIQ, dtype=float)
    #Read the clean data
    fluid_comp_score_train_data = []
    for x in actual_FIQ_train_uncorrected:
        fluid_comp_score_train_data.append(float(''.join(x.replace('"', '').split(','))))
    actual_FIQ_train_uncorrected_label_regression = np.array(fluid_comp_score_train_data,dtype=float)
    #Save the data in HDf5 format
    h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/slice_train_data/ABCD_T1_IQ_train3Sliceactual_FIQ_uncorrected_Split2_'+
                    str(60+slice_start_no)+'_'+str(60+slice_start_no+DimZ-1)+'.h5', 'w')
    h5f.create_dataset('x_T1_train_actual', data=slice_data_train_actualFIQ_data)
    h5f.create_dataset('y_train_actual_FIQ_train_uncorrectedScore_regression', data=actual_FIQ_train_uncorrected_label_regression)
    h5f.close()




