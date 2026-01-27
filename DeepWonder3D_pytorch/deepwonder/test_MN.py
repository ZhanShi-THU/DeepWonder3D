import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import numpy as np
from skimage import io
import math
import tifffile as tiff

from deepwonder.MN.MergeNeuron_SEG import merge_neuron_SEG_mul_inten
from deepwonder.MN.merge_neuron_f import joint_neuron
from deepwonder.MN.merge_neuron_f import listAddcontours_Laplacian_pytorch, list2contours
from deepwonder.MN.merge_neuron_f import listAddtrace, listAdd_remain_trace
import scipy.io as scio

import warnings

warnings.filterwarnings('ignore')
from deepwonder.utils import save_img_train, save_para_dict, UseStyle, save_img, get_netpara


#############################################################################
#############################################################################
class calculate_neuron:
    def __init__(self, mn_para):
        """
        Initialize the neuron calculation and merging class.

        This constructor sets up the configuration for the Neuron Merging (MN) 
        module by resetting parameters, creating necessary output directories, 
        and scanning dataset folders for input patches.

        Args:
            mn_para (dict): A dictionary containing configuration parameters. 
                Common keys include:
                - ``RMBG_datasets_folder``, ``RMBG_datasets_path``
                - ``SEG_datasets_folder``, ``SEG_datasets_path``
                - ``SR_datasets_folder``, ``SR_datasets_path``
                - ``MN_output_dir``, ``MN_output_folder``
        """
        self.RMBG_datasets_folder = ''
        self.RMBG_datasets_path = ''

        self.SEG_datasets_folder = ''
        self.SEG_datasets_path = ''

        self.SR_datasets_folder = ''
        self.SR_datasets_path = ''

        self.MN_output_dir = ''
        self.MN_output_folder = ''
        
        self.reset_para(mn_para)

        self.make_folder()
        self.save_para()
        self.generate_patch()

    #########################################################################
    #########################################################################
    def reset_para(self, MN_para):
        """
        Update class attributes from a parameter dictionary.

        Iterates through the provided dictionary and sets the corresponding 
        instance attributes if they exist.

        Args:
            MN_para (dict): Dictionary of parameters to be updated.
        """
        for key, value in MN_para.items():
            if hasattr(self, key):
                setattr(self, key, value)

    #########################################################################
    #########################################################################
    def make_folder(self):
        """
        Create the output directory structure for neuron merging results.

        Constructs the main output path based on the input dataset names and 
        ensures the directories exist on the disk.
        """
        current_time = 'MN_' + self.RMBG_datasets_folder + '_' + self.SEG_datasets_folder
        self.MN_output_path = self.MN_output_dir + '//' + self.MN_output_folder  
        if not os.path.exists(self.MN_output_dir):
            os.mkdir(self.MN_output_dir)
        if not os.path.exists(self.MN_output_path):
            os.mkdir(self.MN_output_path)

    #########################################################################
    #########################################################################
    def save_para(self):
        """
        Save the current instance parameters to a YAML file.

        Serializes the class attributes into a configuration file for 
        reproducibility and logging purposes.
        """
        yaml_dict = self.__dict__.copy()
        yaml_name = 'MN_para' + '.yaml'
        save_MN_para_path = self.MN_output_path + '//' + yaml_name
        save_para_dict(save_MN_para_path, yaml_dict)

    #########################################################################
    #########################################################################
    def generate_patch(self):
        """
        Scan dataset directories and build file lists for processing.

        This method walks through the RMBG, SEG, and SR directories, identifying 
        TIFF files and mapping their filenames to full file paths.

        Returns:
            tuple: Contains lists and dictionaries for matching files:
                - RMBG_name_list (list of str): List of filenames for RMBG data.
                - RMBG_list (dict): Mapping of filenames to full RMBG file paths.
                - SEG_name_list (list of str): List of filenames for SEG data.
                - SEG_list (dict): Mapping of filenames to full SEG file paths.
                - SR_name_list (list of str): List of filenames for SR data.
                - SR_list (dict): Mapping of filenames to full SR file paths.
        """
        RMBG_folder = self.RMBG_datasets_path + '//' + self.RMBG_datasets_folder
        RMBG_name_list = []
        RMBG_list = {}
        for RMBG_name in list(os.walk(RMBG_folder, topdown=False))[-1][-1]:
            if '.tif' in RMBG_name:
                RMBG_name_list.append(RMBG_name)
                RMBG_dir = RMBG_folder + '//' + RMBG_name
                RMBG_list[RMBG_name] = RMBG_dir

        SEG_folder = self.SEG_datasets_path + '//' + self.SEG_datasets_folder
        SEG_name_list = []
        SEG_list = {}
        for SEG_name in list(os.walk(SEG_folder, topdown=False))[-1][-1]:
            if '.tif' in SEG_name:
                SEG_name_list.append(SEG_name)
                SEG_dir = SEG_folder + '//' + SEG_name
                SEG_list[SEG_name] = SEG_dir

        self.RMBG_name_list = RMBG_name_list
        self.RMBG_list = RMBG_list
        self.SEG_name_list = SEG_name_list
        self.SEG_list = SEG_list

        SR_folder = self.SR_datasets_path + '//' + self.SR_datasets_folder
        SR_name_list = []
        SR_list = {}
        for SR_name in list(os.walk(SR_folder, topdown=False))[-1][-1]:
            if '.tif' in SR_name:
                SR_name_list.append(SR_name)
                SR_dir = SR_folder + '//' + SR_name
                SR_list[SR_name] = SR_dir
                
        self.SR_name_list = SR_name_list
        self.SR_list = SR_list

        return self.RMBG_name_list, self.RMBG_list, self.SEG_name_list, self.SEG_list, self.SR_name_list, self.SR_list

    #########################################################################
    #########################################################################
    def get_neuron_mask(self, SEG_dir, RMBG_dir, SR_dir, im_name):
        """
        Process individual image sets to extract and merge neuron masks.

        Loads segmented, background-removed, and super-resolved images, then 
        performs neuron merging based on intensity and correlation. It 
        calculates Laplacian contours, extracts traces, and exports results 
        as TIFF files and MATLAB structures.

        Args:
            SEG_dir (str): Full path to the segmentation TIFF file.
            RMBG_dir (str): Full path to the background-removed TIFF file.
            SR_dir (str): Full path to the super-resolution TIFF file.
            im_name (str): The base name/identifier of the current image.
        """
        img_SEG = tiff.imread(SEG_dir)
        print('img_SEG ---> ', img_SEG.shape)
        img_RMBG = tiff.imread(RMBG_dir)
        img_SR = tiff.imread(SR_dir)

        img_SEG1 = img_SEG.copy()
        img_RMBG1 = img_RMBG.copy()
        
        self.inten_thres = 0
        whole_mask_list, test_list, all_neuron_mask, all_neuron_remain_mask, \
            mask_stack_filted = merge_neuron_SEG_mul_inten(img_SEG1,
                                                           img_RMBG1,
                                                           quit_round_rate=0.5,
                                                           good_round_rate=0.8,
                                                           good_round_size_rate=0.5,
                                                           corr_mark=0.9,
                                                           max_value=1000,
                                                           if_nmf=False,
                                                           inten_thres=self.inten_thres,
                                                           edge_value=10)

        whole_mask_list = listAddcontours_Laplacian_pytorch(whole_mask_list, img_RMBG.shape[1], img_RMBG.shape[2])

        whole_mask_list = listAddtrace(whole_mask_list, img_RMBG, mode='update', trace_mode='all')
        whole_mask_list = listAdd_remain_trace(whole_mask_list, img_RMBG, mode='update', trace_mode='all')
        whole_mask_list = listAdd_remain_trace(whole_mask_list, img_SR, dict_name='remain_sr_trace', mode='update',
                                               trace_mode='all')

        final_contours, whole_contours = list2contours(whole_mask_list, img_RMBG.shape[1], img_RMBG.shape[2])

        f_con_output_path = self.MN_output_path + '//f_con' + '_' + str(self.inten_thres)
        if not os.path.exists(f_con_output_path):
            os.mkdir(f_con_output_path)
        img_f_contours_name = f_con_output_path + '//' + im_name + '_f_con.tif'
        final_contours = final_contours.clip(0, 65535).astype('uint16')
        io.imsave(img_f_contours_name, final_contours)

        from deepwonder.MN.merge_neuron_f import list2masks
        f_mask_bina_output_path = self.MN_output_path + '//f_mask_bina' + '_' + str(self.inten_thres)
        if not os.path.exists(f_mask_bina_output_path):
            os.mkdir(f_mask_bina_output_path)

        final_masks, whole_masks = list2masks(whole_mask_list, img_RMBG.shape[1], img_RMBG.shape[2])
        final_masks_bina = final_masks
        final_masks_bina[final_masks_bina > 0] = 1
        img_f_masks_bina_name = f_mask_bina_output_path + '//' + im_name + '_f_mask_bina.tif'
        final_masks_bina = final_masks_bina.clip(0, 65535).astype('uint16')
        io.imsave(img_f_masks_bina_name, final_masks_bina)

        w_con_output_path = self.MN_output_path + '//w_con' + '_' + str(self.inten_thres)
        if not os.path.exists(w_con_output_path):
            os.mkdir(w_con_output_path)
        img_w_contours_name = w_con_output_path + '//' + im_name + '_w_con.tif'
        whole_contours = whole_contours.clip(0, 65535).astype('uint16')

        mat_output_path = self.MN_output_path + '//mat' + '_' + str(self.inten_thres)
        if not os.path.exists(mat_output_path):
            os.mkdir(mat_output_path)
        mat_save_name = mat_output_path + '//' + im_name + '.mat'
        
        for i in range(0, len(whole_mask_list)):
            single_neuron = whole_mask_list[i]
            del single_neuron['mask']
            
        scio.savemat(mat_save_name, {'final_mask_list': whole_mask_list, 'final_contours': final_contours})

        img_w_contours_name = self.MN_output_path + '//' + im_name + '.tif'
        whole_contours = whole_contours.clip(0, 65535).astype('uint16')

    #########################################################################
    #########################################################################
    def run(self):
        """
        Iterate through the identified image datasets and perform neuron merging.

        Loops over the list of background-removed image filenames, matches 
        them with corresponding segmentation and super-resolution results, 
        and executes the mask extraction process.
        """
        for im_index in range(0, len(self.RMBG_name_list)):
            im_name = self.RMBG_name_list[im_index]
            img_RMBG = self.RMBG_list[im_name]
            img_SEG = self.SEG_list[im_name]
            img_SR = self.SR_list[im_name]
            self.get_neuron_mask(img_SEG, img_RMBG, img_SR, im_name)


if __name__ == '__main__':
    MN_parameters_test = {'RMBG_datasets_folder': 'RMBG',
                          'RMBG_datasets_path': 'test_results//SR_test_07192023_2_113_up12',
                          ###########################
                          'SEG_datasets_folder': 'SEG',
                          'SEG_datasets_path': 'test_results//SR_test_07192023_2_113_up12',
                          ###########################
                          'SR_datasets_folder': 'pred_signal',
                          'SR_datasets_path': 'test_results//SR_test_07192023_2_113_up12',
                          'MN_output_dir': 'test_results',
                          }

    MN_parameters_test['MN_output_folder'] = 'MN_' + MN_parameters_test['RMBG_datasets_folder'] + \
                                             '_' + MN_parameters_test['SEG_datasets_folder']
    MN_model = calculate_neuron(MN_parameters_test)
    MN_model.run()
