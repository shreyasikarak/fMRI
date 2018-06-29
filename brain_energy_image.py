
import nilearn
from nilearn import plotting
import nibabel as nib
from nilearn.image import threshold_img, index_img, concat_imgs
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import connected_regions
import os
import numpy as np
import random
import pdb


def brain_energy_image(t):

    energy_values = t[1]
    points_VAC = [[0,0,0,1],[1,0,0,1],[0,1,0,1],[0,0,1,1],
                  [1,1,0,1],[0,1,1,1],[1,0,1,1],[1,1,1,1]]
    

    mni_template_file = os.path.join(t[0],'mni_icbm152_gm_tal_nlin_asym_09c.nii.gz')
    mni_template = nib.load(mni_template_file)
    data = mni_template.get_data()
    aft = mni_template.affine
    iaft = np.linalg.pinv(aft)

    mni_template_file_t2 = os.path.join(t[0],'mni_icbm152_t2_tal_nlin_asym_09c.nii.gz')
    mni_template_t2 = nib.load(mni_template_file_t2)
    data_t2 = mni_template_t2.get_data()
    aft_t2 = mni_template_t2.affine
    iaft_t2 = np.linalg.pinv(aft_t2)


    '''pdb.set_trace()
    nilearn.plotting.plot_anat(anat_img=data, cut_coords=[-17], output_file=os.path.join('MNI'+'image'+'.png'), draw_cross=False,display_mode='z')'''
    
    max_val = np.max(data_t2)
    
    ''' all_indexes = (csv_file['reg_rename']==unq_regions[unq_reg_ind]).nonzero()
    
    for indexes in range(np.size(all_indexes)):
        xyz = [csv_file['x'][all_indexes[0][indexes]],csv_file['y'][all_indexes[0][indexes]],
               csv_file['z'][all_indexes[0][indexes]]]
        
        xyz.extend([1])
        voxel_image_index=iaft.dot(xyz)
        voxel_image_index_t2=iaft_t2.dot(xyz)'''


    print("Started filling up the values")
    for x_coord in range(data.shape[0]):
        print(x_coord)
        for y_coord in range(data.shape[1]):

            for z_coord in range(data.shape[2]):        


                '''voxel_image_index = [x_coord, y_coord, z_coord]            
                voxel_image_index_t2 = [x_coord, y_coord, z_coord]            

                voxel_image_index.extend([1]) 
                voxel_image_index_t2.extend([1])'''
                chance = random.randint(0,50)

                if chance == 1:
                    number_to_add = random.randrange(10, 50)
                    data[x_coord, y_coord, z_coord] = data[x_coord, y_coord, z_coord] + number_to_add
                else:                    
                    data[x_coord, y_coord, z_coord] = data[x_coord, y_coord, z_coord]

                    #t2_val = max_val+(float(energy_values[x_coord, y_coord, z_coord])/float(40))*(200-max_val)

                    #data_t2[x_coord, y_coord, z_coord] = t2_val

                    '''for k_ind in range(2):
                        for j_ind in range(2):
                            for i_ind in range(2):
                                try:
                                    # Sort out the problem of np.round. Use no.round instead of int.
                                    data[int(voxel_image_index[0])+i_ind,   
                                    int(voxel_image_index[1])+j_ind,
                                    int(voxel_image_index[2])+k_ind] = energy_values[x_coord, y_coord, z_coord]

                                    t2_val = max_val+(float(energy_values[x_coord, y_coord, z_coord])/float(40))*(200-max_val)
                                    
                                    data_t2[int(voxel_image_index_t2[0])+i_ind,
                                    int(voxel_image_index_t2[1])+j_ind,
                                    int(voxel_image_index_t2[2])+k_ind] = t2_val

                                except Exception as e:
                                    print(e)
                                    pdb.set_trace()'''
                                    
    print("Saving the image")                                    
    file_name = 'recreated_gm_image'+'.nii'
    nib.save(mni_template,file_name)                    

    '''file_name = 'recreated_t2_image'+'.nii'
    nib.save(mni_template_t2,file_name)                    '''

    '''mni_template_file = os.path.join(t[0],'mni_icbm152_gm_tal_nlin_asym_09c.nii.gz')
    mni_template = nib.load(mni_template_file)
    data = mni_template.get_data()
    aft = mni_template.affine

    voxel_image_index = iaft.dot([0,0,0,1]) # Vertical Anterior Commisure
    for k_ind in range(5):
        for j_ind in range(5):
            for i_ind in range(5):
                data[int(voxel_image_index[0])-2+i_ind,
                int(voxel_image_index[1])-2+j_ind,
                int(voxel_image_index[2])-2+k_ind] = 10

    file_name = 'VAC.nii'
    nib.save(mni_template,file_name)'''
