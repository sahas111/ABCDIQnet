__author__= 'Susmita Saha'

'''Slice+Volume data prep file'''

import SimpleITK as sitk
import h5py
import glob2
import numpy as np
import pandas as pd
import tarfile
import shutil
import os

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
#Read the images
T1_data_val=glob2.glob('/Volumes/{cci-msk}/reference/Projects/ABCD/final/ABCDChallenge2019Val/fmriresults01/image03/validation/*.tgz')
#Read the score file
f=open('/Users/sah012/clinicalData/abcd_tbss01.txt',"r")
lines=f.readlines()
all_ID_FIQ=[]
for x in lines:
    all_ID_FIQ.append(x.split('\t'))
f.close()
all_ID_FIQ=np.array(all_ID_FIQ)

#Read the volume info
seg_file = pd.read_csv('/Volumes/{cci-msk}/reference/Projects/ABCD/final/ABCDChallenge2019Val/btsv01.csv')
seg_file_subID = seg_file['subjectkey'][1:-1]
# pons_WM_vol = seg_file['sri24ponswm'][1:-1]
# pons_WM_vol=seg_file['sri24precentralrgm'][1:-1]
right_hippocampus_GM_vol = seg_file['sri24thalamusrgm'][1:-1]
left_hippocampus_GM_vol = seg_file['sri24thalamuslgm'][1:-1]
right_thalamus_GM_vol = seg_file['sri24thalamusrgm'][1:-1]
left_thalamus_GM_vol = seg_file['sri24thalamuslgm'][1:-1]

#Split the data if needed to overcome memory limit
T1_data_val = T1_data_val[0:415]

#Prepare the data for the best slice

actual_FIQ_val = []
slice_data_val_actualFIQ = []
# pons_WM_vol_val = []
right_hippocampus_GM_vol_val=[]
left_hippocampus_GM_vol_val=[]
right_thalamus_GM_vol_val = []
left_thalamus_GM_vol_val = []

for slice_start_no in range(36, 38, 3):
    count=0
    for item in T1_data_val:
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
                print(str(img [30:46]))
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
                                if str(img [30:46]) in str(i) and all_ID_FIQ[index + 2, 46] != '""' and str(img [30:46]) in list(seg_file_subID):
                                    index_WM=list(seg_file_subID).index(str(img [30:46]))
                                    if list(right_hippocampus_GM_vol)[index_WM] is not None\
                                            and list(left_hippocampus_GM_vol)[index_WM] is not None\
                                            and list(right_thalamus_GM_vol)[index_WM] is not None\
                                            and list(left_hippocampus_GM_vol)[index_WM] is not None:
                                        actual_FIQ_val.append(all_ID_FIQ[index + 2, 46])
                                        slice_data_val_actualFIQ.append(slice)
                                        left_hippocampus_GM_vol_val.append(list(left_hippocampus_GM_vol)[index_WM])
                                        right_hippocampus_GM_vol_val.append(list(right_hippocampus_GM_vol)[index_WM])
                                        left_thalamus_GM_vol_val.append(list(left_thalamus_GM_vol)[index_WM])
                                        right_thalamus_GM_vol_val.append(list(right_thalamus_GM_vol)[index_WM])

                            j = j + 1

                            if j == 1:
                                break
                filelist = glob2.glob('/Users/sah012/Desktop/output1/*/*/*/*')
                # print(filelist)
                for f in filelist:
                    os.remove(os.path.abspath(f))
                shutil.rmtree('/Users/sah012/Desktop/output1/')
                break
            # else:
            #     print('not found')

    slice_data_val_actualFIQ_data = np.array(slice_data_val_actualFIQ, dtype=float)

    fluid_comp_score_val_data = []
    for x in actual_FIQ_val:
       fluid_comp_score_val_data.append(float(''.join(x.replace('"', '').split(','))))
    actual_FIQ_val_label_regression = np.array(fluid_comp_score_val_data,dtype=float)

    left_hippocampus_GM_vol_val_data=np.array(left_hippocampus_GM_vol_val,dtype=float)
    right_hippocampus_GM_vol_val_data=np.array(right_hippocampus_GM_vol_val,dtype=float)
    left_thalamus_GM_vol_val_data=np.array(left_thalamus_GM_vol_val,dtype=float)
    right_thalamus_GM_vol_val_data=np.array(right_thalamus_GM_vol_val,dtype=float)
    #Save the data in HDf5 format
    h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/slice_val_data/ABCD_T1_IQ_val3Sliceactual_LR_thalamus_hippo_FIQ_'+
                    str(60+slice_start_no)+'_'+str(60+slice_start_no+DimZ-1)+'.h5', 'w')

    h5f.create_dataset('left_hippocampus_GM_vol_val', data=left_hippocampus_GM_vol_val_data)
    h5f.create_dataset('right_hippocampus_GM_vol_val', data=right_hippocampus_GM_vol_val_data)
    h5f.create_dataset('left_thalamus_GM_vol_val', data=left_thalamus_GM_vol_val_data)
    h5f.create_dataset('right_thalamus_GM_vol_val', data=right_thalamus_GM_vol_val_data)
    h5f.create_dataset('actual_FIQ_val_label_regression', data=actual_FIQ_val_label_regression)
    h5f.create_dataset('x_T1_val_actual', data=slice_data_val_actualFIQ_data)
    h5f.close()

#Add the splits if necessary

# h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/slice_train_data/' + 'ABCD_T1_IQ_train3Sliceactual_LR_thalamus_hippo_FIQ_Split1_96_98.h5', 'r')
# X_train_actual1 = h5f['x_T1_train_actual'][:]
# Y_train_regress_actual1=h5f['actual_FIQ_train_label_regression'][:]
# left_hippocampus_GM_vol_train1=h5f['left_hippocampus_GM_vol_train'][:]
# right_hippocampus_GM_vol_train1=h5f['right_hippocampus_GM_vol_train'][:]
# left_thalamus_GM_vol_train1=h5f['left_thalamus_GM_vol_train'][:]
# right_thalamus_GM_vol_train1=h5f['right_thalamus_GM_vol_train'][:]
# h5f.close()
#
# h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/slice_train_data/' + 'ABCD_T1_IQ_train3Sliceactual_LR_thalamus_hippo_FIQ_Split2_96_98.h5', 'r')
# X_train_actual2 = h5f['x_T1_train_actual'][:]
# Y_train_regress_actual2=h5f['actual_FIQ_train_label_regression'][:]
# left_hippocampus_GM_vol_train2=h5f['left_hippocampus_GM_vol_train'][:]
# right_hippocampus_GM_vol_train2=h5f['right_hippocampus_GM_vol_train'][:]
# left_thalamus_GM_vol_train2=h5f['left_thalamus_GM_vol_train'][:]
# right_thalamus_GM_vol_train2=h5f['right_thalamus_GM_vol_train'][:]
# h5f.close()

# X_train_actual=np.vstack((X_train_actual1,X_train_actual2))
# Y_train_regress_actual1=np.asarray(Y_train_regress_actual1)
# Y_train_regress_actual2=np.asarray(Y_train_regress_actual2)
# left_hippocampus_GM_vol_train1=np.asarray(left_hippocampus_GM_vol_train1)
# left_hippocampus_GM_vol_train2=np.asarray(left_hippocampus_GM_vol_train2)
# right_hippocampus_GM_vol_train1=np.asarray(right_hippocampus_GM_vol_train1)
# right_hippocampus_GM_vol_train2=np.asarray(right_hippocampus_GM_vol_train2)
# left_thalamus_GM_vol_train1=np.asarray(left_thalamus_GM_vol_train1)
# left_thalamus_GM_vol_train2=np.asarray(left_thalamus_GM_vol_train2)
# right_thalamus_GM_vol_train1=np.asarray(right_thalamus_GM_vol_train1)
# right_thalamus_GM_vol_train2=np.asarray(right_thalamus_GM_vol_train2)
# Y_train_regress_actual=np.concatenate((Y_train_regress_actual1,Y_train_regress_actual2))
# Y_train_regress_actual=np.array(Y_train_regress_actual)
# left_hippocampus_GM_vol_train=np.concatenate((left_hippocampus_GM_vol_train1,left_hippocampus_GM_vol_train2))
# left_hippocampus_GM_vol_train=np.array(left_hippocampus_GM_vol_train)
# right_hippocampus_GM_vol_train=np.concatenate((right_hippocampus_GM_vol_train1,left_hippocampus_GM_vol_train2))
# right_hippocampus_GM_vol_train=np.array(right_hippocampus_GM_vol_train)
# left_thalamus_GM_vol_train=np.concatenate((left_thalamus_GM_vol_train1,left_thalamus_GM_vol_train2))
# left_thalamus_GM_vol_train=np.array(left_thalamus_GM_vol_train)
# right_thalamus_GM_vol_train=np.concatenate((right_thalamus_GM_vol_train1,right_thalamus_GM_vol_train2))
# right_thalamus_GM_vol_train=np.array(right_thalamus_GM_vol_train)

#Save the final training data
# h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/slice_train_data/' + 'ABCD_T1_IQ_train3Sliceactual_LR_thalamus_hippo_FIQ_96_98.h5', 'w')
# h5f.create_dataset('x_T1_train_actual', data=X_train_actual)
# h5f.create_dataset('y_train_actual_FIQ_trainScore_regression', data=Y_train_regress_actual)
# h5f.create_dataset('left_hippocampus_GM_vol_train', data=left_hippocampus_GM_vol_train)
# h5f.create_dataset('right_hippocampus_GM_vol_train', data=right_hippocampus_GM_vol_train)
# h5f.create_dataset('left_thalamus_GM_vol_train', data=left_thalamus_GM_vol_train)
# h5f.create_dataset('right_thalamus_GM_vol_train', data=right_thalamus_GM_vol_train)
# h5f.close()

