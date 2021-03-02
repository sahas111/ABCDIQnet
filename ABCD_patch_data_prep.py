__author__= 'Susmita Saha'

'''Patch input data preparation file'''

import SimpleITK as sitk
import glob2
import numpy as np
import tarfile
import os
import sys
import h5py

#Read the command line arguments: best slice range
slice_start_no=int(sys.argv[1])
slice_end_no=int(sys.argv[2])
split_no=sys.argv[3]

#Read IQ scores
f=open('/Users/sah012/clinicalData/abcd_tbss01.txt',"r")
lines=f.readlines()
result=[]
for x in lines:
    result.append(x.split('\t'))
f.close()
result1=np.array(result)

"Crystallized Composite Age-Corrected Standard Score"
print(result1[1][50])
"Cognition Fluid Composite Age-Corrected Standard Score"
print(result1[2][46])

#Functions for preparing image data
def read_scan_image(img_file, np_dtype):
    # Read SimpleITK scan image
    img_sitk = sitk.ReadImage(img_file)
    # Convert to a NumPy array
    img_np = np.array(sitk.GetArrayFromImage(img_sitk)).astype(np_dtype)
    return img_sitk, img_np
def extract_patch(img_np, lims_patch):
    ''' image informations are stored in y,z,x sequence '''
    patch = (img_np[
             lims_patch['z'][0]: lims_patch['z'][1],
             lims_patch['x'][0]: lims_patch['x'][1],
             lims_patch['y'][0]: lims_patch['y'][1]
             ])
    patch_reshaped = np.transpose(patch, (1, 2, 0))
    return patch_reshaped
## Define patch size
PatchX=60
PatchY=60
PatchZ=3
## Define the stride
stride_X=int(PatchX)
stride_Y=int(PatchY)
stride_Z=int(PatchZ)
stride =[stride_Z,stride_X,stride_Y]
#Read the data
T1_data_train=glob2.glob('/Volumes/{cci-msk}/reference/Projects/ABCD/final/ABCDChallenge2019Train/fmriresults01/image03/training/*.tgz')
mask_train=glob2.glob('/Volumes/{cci-msk}/reference/Projects/ABCD/final/ABCDChallenge2019Train/fmriresults01/image03/training_mask/*mask.nii.gz')
#Train data generation
patch_data_per_sub=[]
patch_scores_per_sub=[]
count=0
for idx,item in enumerate(T1_data_train):
    patch_data_train = []
    patch_fluid_comp_score_train = []
    x_y_z_train=[]
    print('{0}').format(count)
    tar = tarfile.open(item, 'r')
    if not os.path.isdir('/Users/sah012/Desktop/output1/'):
        os.makedirs('/Users/sah012/Desktop/output1/')
    for item1 in tar:
        tar.extract(item1, '/Users/sah012/Desktop/output1/')
        if (item1.name.find("t1_brain.nii")) != -1:
            img = '/Users/sah012/Desktop/output1/' + str(item1.name)
            img_sitk, img_np = read_scan_image(img, np.float32)
            mask_sitk, mask_np = read_scan_image(mask_train[idx], np.float32)
            lims_img = {
                'z': [slice_start_no, slice_end_no],
                'x': [0 + int(stride[1]/2), img_np.shape[1] - int(stride[1]/2)+1],
                'y': [0 + int(stride[2]/2), img_np.shape[2] - int(stride[2]/2)+1]}

            for x in range(lims_img['x'][0], lims_img['x'][1], int(stride[1]/2)):
                for y in range(lims_img['y'][0], lims_img['y'][1], int(stride[2]/2)):
                    for z in range(lims_img['z'][0], lims_img['z'][1], PatchZ):
                        lims_patch = {'z': [z, z+PatchZ],
                                      'x': [x - int(stride[1]/2), x + int(stride[1]/2)],
                                      'y': [y - int(stride[2]/2), y + int(stride[2]/2)]}

                        patch = extract_patch(img_np, lims_patch)
                        patch_mask = extract_patch(mask_np, lims_patch)
                        if np.count_nonzero(patch_mask != 0) > np.count_nonzero(patch_mask == 0):
                            for index,i in enumerate(list(result1[2:-1,3])):
                                if str(item[107:118]) in str(i) and result1[index+2, 45] != '""':
                                    patch_fluid_comp_score_train.append(result1[index+2, 45])
                                    patch_data_train.append(patch)
                                    x_y_z_train.append(str(x)+str('_')+str(y)+str('_')+str(z))

    if len(patch_fluid_comp_score_train)!=0:
        patch_data_train = np.array(patch_data_train)
        patch_data_train=patch_data_train[0:4]
        x_y_z_train=x_y_z_train[0:4]
        print (patch_data_train.shape)
        print('xyz',x_y_z_train[1])
        patch_data_per_sub.append(patch_data_train)
        patch_scores_per_sub.append(patch_fluid_comp_score_train[0])

    count+=1


patch_scores_per_sub_cleaned = []
for x in patch_scores_per_sub:
    patch_scores_per_sub_cleaned.append(float(''.join(x.replace('"', '').split(','))))

patch_scores_per_sub_cleaned_final = np.array(patch_scores_per_sub_cleaned,dtype=float)
patch_data_per_sub=np.array(patch_data_per_sub)
print ({0}).format(patch_data_per_sub.shape)
print ({0}).format(patch_scores_per_sub_cleaned_final.shape)


#Save the patch data in HDf5 binary data format
h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/patch_train_data/' + 'ABCD_T1_IQ_train_patch_60_60_3_'+str(slice_start_no)+'_'+str(slice_end_no)+'_'+str(split_no)+'.h5','w')
h5f.create_dataset('x_T1_train_actual', data=patch_data_per_sub)
h5f.create_dataset('y_train_actual_FIQ_trainScore_regression', data=patch_scores_per_sub_cleaned_final)
h5f.close()
#Add splits
h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/patch_train_data/' + 'ABCD_T1_IQ_train_patch_60_60_3_87_89_2_1.h5', 'r')
X_train_actual1 = h5f['x_T1_train_actual'][:]
Y_train_regress_actual1=h5f['y_train_actual_FIQ_trainScore_regression'][:]
h5f.close()
h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/patch_train_data/' + 'ABCD_T1_IQ_train_patch_60_60_3_87_89_2_2.h5', 'r')
X_train_actual2 = h5f['x_T1_train_actual'][:]
Y_train_regress_actual2=h5f['y_train_actual_FIQ_trainScore_regression'][:]
h5f.close()

X_train_actual=np.vstack((X_train_actual1,X_train_actual2))
Y_train_regress_actual1=np.asarray(Y_train_regress_actual1)
Y_train_regress_actual2=np.asarray(Y_train_regress_actual2)
Y_train_regress_actual=np.concatenate((Y_train_regress_actual1,Y_train_regress_actual2))
Y_train_regress_actual=np.array(Y_train_regress_actual)

#save the added splits in HDf5 binary data format
h5f = h5py.File('/Users/sah012/PycharmProjects/ABCD-test-data/patch_train_data/' + 'ABCD_T1_IQ_train_patch_60_60_3_'+str(slice_start_no)+'_'+str(slice_end_no)+'.h5', 'w')
h5f.create_dataset('x_T1_train_actual', data=X_train_actual)
h5f.create_dataset('y_train_actual_FIQ_trainScore_regression', data=Y_train_regress_actual)
h5f.close()

