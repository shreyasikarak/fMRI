
import nilearn
from nilearn import plotting
import nibabel as nib
from nilearn.image import threshold_img, index_img, concat_imgs
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_regions
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

from brain_energy_image import brain_energy_image

import pdb

MNI_template_path = '/home/silp150/shreyashi/'
# MNI_template = nib.load(MNI_template_name)

t1_file_name = 'tfMRI_EMOTION_RL'
t1_path = '/home/silp150/shreyashi/100307_preproc/100307/T1w/Results/' + t1_file_name
os.chdir(t1_path)
t1_image_files = glob.glob("SBRef_dc.nii.gz")
t1_image = nib.load(t1_image_files[0])		
structural_data = t1_image.get_data()
original_data = structural_data.copy()

# Enlisting Directories
# dirs = glob.glob("*")
#dirs = ["tfMRI_EMOTION_RL.nii.gz"]

# Debugger

#for dir_ in dirs:
#os.chdir(dirs[0])

file_name = 'tfMRI_EMOTION_RL'
source_path = '/home/silp150/shreyashi/100307_preproc/100307/MNINonLinear/Results/'+ file_name
target_path = ''
os.chdir(source_path)
image_files = glob.glob("%s.nii.gz" %file_name)


# pdb.set_trace()
#	for image_file in image_files[0]:		
data_file = nib.load(image_files[0])		


data = data_file.get_data()
max_data_val = np.max(data)
min_data_val = np.min(data)


if not os.path.isfile('every_values_fMRI.npy'):
	energy_values = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
	for x_coord in range(data.shape[0]):
		print(x_coord)
		for y_coord in range(data.shape[1]):
			for z_coord in range(data.shape[2]):
				for slice_id in range(data.shape[3]):

					# vall = (np.abs(data[x_coord, y_coord, z_coord, slice_id])/max_data_val)*255

					#energy_values[x_coord, y_coord, z_coord] = energy_values[x_coord, y_coord, z_coord] + vall
					energy_values[x_coord, y_coord, z_coord] = energy_values[x_coord, y_coord, z_coord] + np.abs(data[x_coord, y_coord, z_coord, slice_id])

	energy_values = energy_values / data.shape[3]
	np.save('every_values_fMRI.npy', energy_values)

else:
	energy_values = np.load('every_values_fMRI.npy')


if not os.path.isfile('recreated_t2_image.nii'):
	brain_energy_image(list([MNI_template_path, energy_values]))


'''print("Started filling up the values")
for x_coord in range(energy_values.shape[0]):
    print(x_coord)
    for y_coord in range(energy_values.shape[1]):

        for z_coord in range(energy_values.shape[2]):        

           
            structural_data[x_coord, y_coord, z_coord] = energy_values[x_coord, y_coord, z_coord]

            #t2_val = max_val+(float(energy_values[x_coord, y_coord, z_coord])/float(40))*(200-max_val)

            #structural_data[x_coord, y_coord, z_coord] = t2_val

file_name = 'recreated_gm_image'+'.nii'
nib.save(t1_image, file_name) 

recreated_image = nib.load(file_name)                   

pdb.set_trace()
nilearn.plotting.plot_anat(anat_img=recreated_image, cut_coords=[-17], output_file=os.path.join('Created_Images','image'+'.png'), draw_cross=False,display_mode='z')'''

gm_image = nib.load('recreated_gm_image.nii')
recreated_data = gm_image.get_data()
#t2_image = nib.load('recreated_t2_image.nii')

# pdb.set_trace()
# nilearn.plotting.plot_anat(anat_img=gm_image, cut_coords=[-17], output_file=os.path.join('MNI'+'image'+'.png'), draw_cross=False,display_mode='z')
maxm = np.max(recreated_data)
minm = np.min(recreated_data)
range_ = (maxm-minm)/6

upper_1 = minm + range_
upper_2 = upper_1 + range_
upper_3 = upper_2 + range_
upper_4 = upper_3 + range_
upper_5 = upper_4 + range_
upper_6 = upper_5 + range_

touple_1 = np.where((recreated_data > minm) & (recreated_data < upper_1))
touple_2 = np.where((recreated_data > upper_1) & (recreated_data < upper_2))
touple_3 = np.where((recreated_data > upper_2) & (recreated_data < upper_3))
touple_4 = np.where((recreated_data > upper_3) & (recreated_data < upper_4))
touple_5 = np.where((recreated_data > upper_4) & (recreated_data < upper_5))
touple_6 = np.where((recreated_data > upper_5) & (recreated_data < upper_6))


aft = gm_image.affine
fmri_coord = [[]]
counter = 0

for touple in zip(touple_6[0],touple_6[1],touple_6[2]):
	coord = [touple[0], touple[1], touple[2]]
	coord.extend([1])
	#fmri_coord = np.append(fmri_coord, aft.dot(coord))
	fmri_coord = aft.dot(coord)
	print(fmri_coord)	
	nilearn.plotting.plot_anat(anat_img=gm_image, cut_coords=[fmri_coord[2]], 
		output_file=os.path.join('Created_Images','image'+str(counter)+'.png'),
		draw_cross=False,display_mode='z')
	counter += 1
	print("=================================================================================")
	pdb.set_trace()
# iaft = np.linalg.pinv(aft)


# np.where((data_file.get_data()==np.max(data_file.get_data()))==True)
#nilearn.plotting.plot_anat(anat_img=data_file, cut_coords=[68.19861416,  79.35950654, -25.78495284], output_file='image6.png',draw_cross=False,display_mode='ortho')

