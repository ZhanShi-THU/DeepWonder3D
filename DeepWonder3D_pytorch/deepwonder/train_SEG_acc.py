import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import datetime
import functools
import numpy as np

from deepwonder.SEG.SEG_data_process_train import train_preprocess_lessMemory_SEG, trainset_SEG
from deepwonder.SEG.SEG_network import SEG_Network_3D_Unet
from deepwonder.SEG.SEG_utils import FFDrealign4, inv_FFDrealign4

from skimage import io

from deepwonder.utils import save_img_train, save_para_dict, UseStyle, get_netpara
import warnings
warnings.filterwarnings("ignore")
import yaml
import sys
########################################################################################################
class train_seg_net_acc():
    def __init__(self, SEG_para):
        """
        Training wrapper for the 3D segmentation (SEG) network.

        This class configures a 3D UNet-based segmentation model that operates
        on temporal stacks, constructs training datasets, runs the training
        loop, and manages saving of configuration, intermediate outputs, and
        checkpoints.

        Args:
            SEG_para (dict): Configuration dictionary for training. Typical
                keys:
                - ``GPU``: Comma-separated GPU indices.
                - ``SEG_n_epochs``, ``SEG_batch_size``: Training schedule.
                - ``SEG_img_*``: 3D patch size.
                - ``SEG_output_dir`` and dataset-related paths/folders.
                - ``SEG_use_pretrain`` and pretrain-related fields.
        """
        self.GPU = '0,1'
        self.SEG_n_epochs = 100
        self.SEG_batch_size = 4

        self.SEG_img_w = 128
        self.SEG_img_h = 128
        self.SEG_img_s = 32

        self.SEG_lr = 0.00005
        self.SEG_b1 = 0.5
        self.SEG_b2 = 0.999
        self.SEG_norm_factor = 1

        self.SEG_f_maps = 32
        self.SEG_in_c = 4
        self.SEG_out_c = 4
        self.SEG_down_num = 4

        self.SEG_output_dir = './results'
        self.SEG_datasets_folder = 'train'
        self.SEG_GT_folder = 'GT'
        self.SEG_input_folder = 'Input'  
        self.SEG_datasets_path = 'datasets'  

        self.SEG_use_pretrain = 0
        self.SEG_pretrain_index = 'signal_SR_220.pth'
        self.SEG_pretrain_model = 'SR_signal_up15_down5_20230302-2250'
        self.SEG_pretrain_path = 'pth'

        self.SEG_pth_path = 'pth'
        self.SEG_select_img_num = 10000  
        self.SEG_train_datasets_size = 2000  

        self.reset_para(SEG_para)
        self.make_folder()


    #########################################################################
    #########################################################################
    def make_folder(self):
        """
        Create output and checkpoint folders for SEG training.

        Uses a timestamped subdirectory under ``SEG_output_dir`` and
        ``SEG_pth_path`` to store configuration, logs, sample outputs, and
        model weights.
        """
        current_time = 'SEG_'+datetime.datetime.now().strftime("%Y%m%d%H%M")
        self.SEG_output_path = self.SEG_output_dir + '/' + current_time
        if not os.path.exists(self.SEG_output_dir): 
            os.mkdir(self.SEG_output_dir)
        if not os.path.exists(self.SEG_output_path): 
            os.mkdir(self.SEG_output_path)

        self.SEG_pth_save_path = self.SEG_pth_path+'//'+ current_time
        if not os.path.exists(self.SEG_pth_path): 
            os.mkdir(self.SEG_pth_path)
        if not os.path.exists(self.SEG_pth_save_path): 
            os.mkdir(self.SEG_pth_save_path)


    #########################################################################
    #########################################################################
    def reset_para(self, SEG_para):
        """
        Reset instance attributes from parameter dictionary and optional pretrain YAML.

        First updates attributes based on ``SEG_para`` (if attributes exist),
        then, when ``SEG_use_pretrain`` is enabled and the pretrain YAML is
        found, overrides core model hyperparameters to align with the
        pretraining setup.

        Args:
            SEG_para (dict): Parameter dictionary including training and
                dataset options.
        """
        for key, value in SEG_para.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if self.SEG_use_pretrain:
            pretrain_yaml = self.SEG_pretrain_path+'//'+self.SEG_pretrain_model+'//'+'SEG_para'+'.yaml'
            print('pretrain_yaml -----> ', pretrain_yaml, os.path.exists(pretrain_yaml))
            if os.path.exists(pretrain_yaml):
                # with open(pretrain_yaml, 'r') as f:
                f = open(pretrain_yaml)
                pretrain_para = yaml.load(f.read(), Loader = yaml.FullLoader)
                setattr(self, 'SEG_f_maps', pretrain_para['SEG_f_maps'])
                setattr(self, 'SEG_img_s', pretrain_para['SEG_img_s'])
                setattr(self, 'SEG_in_c', pretrain_para['SEG_in_c'])
                setattr(self, 'SEG_out_c', pretrain_para['SEG_out_c'])
                setattr(self, 'SEG_norm_factor', pretrain_para['SEG_norm_factor'])

        print(UseStyle('Training parameters ----->', mode = 'bold', fore  = 'red'))
        print(self.__dict__)


    #########################################################################
    #########################################################################
    def save_para(self):
        """
        Save SEG training configuration to YAML and TXT files.

        Serializes the current instance attributes (excluding the network and
        optimizer) to YAML and TXT files in both the main output folder and
        the checkpoint folder for reproducibility.
        """
        yaml_dict = self.__dict__.copy()
        del yaml_dict['SEG_net'] 
        del yaml_dict['optimizer'] 
        yaml_name = 'SEG_para.yaml'
        save_SEG_para_path = self.SEG_output_path+ '//'+yaml_name
        save_para_dict(save_SEG_para_path, yaml_dict)
        save_SEG_para_path = self.SEG_pth_save_path+ '//'+yaml_name
        save_para_dict(save_SEG_para_path, yaml_dict)

        txt_dict = self.__dict__.copy()
        txt_name = 'SEG_para_'+str(get_netpara(self.SEG_net))+'.txt'
        save_SEG_para_path = self.SEG_output_path+ '//'+txt_name
        save_para_dict(save_SEG_para_path, txt_dict)
        save_SEG_para_path = self.SEG_pth_save_path+ '//'+txt_name
        save_para_dict(save_SEG_para_path, txt_dict)


    #########################################################################
    #########################################################################
    def initialize_model(self):
        """
        Initialize the SEG model, loss functions, and optimizer.

        Constructs the 3D UNet segmentation model, wraps it in
        ``torch.nn.DataParallel``, moves it to GPU, optionally loads a
        pretrained checkpoint, sets up BCE and MSE losses, initializes the
        Adam optimizer, and saves configuration via ``save_para``.
        """
        GPU_list = self.GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list #str(opt.GPU)

        self.SEG_net = SEG_Network_3D_Unet(UNet_type= 'TS_UNet3D',
                                            in_channels = self.SEG_in_c,
                                            out_channels = self.SEG_out_c,
                                            frame_num = self.SEG_img_s,
                                            final_sigmoid = True,
                                            f_maps = self.SEG_f_maps)

        self.SEG_net = torch.nn.DataParallel(self.SEG_net) 
        self.SEG_net.cuda()
        # self.Dnet.cuda()
        L1_pixelwise = torch.nn.L1Loss().cuda() 
        L2_pixelwise = torch.nn.MSELoss().cuda()
        BCEloss_function = torch.nn.BCELoss().cuda()
        if self.SEG_use_pretrain:
            SEG_net_pth_name = self.SEG_pretrain_index
            SEG_net_model_path = self.SEG_pretrain_path+'//'+self.SEG_pretrain_model+'//'+SEG_net_pth_name
            self.SEG_net.load_state_dict(torch.load(SEG_net_model_path))

        self.optimizer = torch.optim.Adam(self.SEG_net.parameters(),
                                        lr=self.SEG_lr, betas=(self.SEG_b1, self.SEG_b2))

        print('\r Parameters of SEG_net -----> ' , get_netpara(self.SEG_net) )
        self.save_para()


    #########################################################################
    #########################################################################
    def generate_patch(self):
        """
        Build training patch metadata for segmentation.

        Uses ``train_preprocess_lessMemory_SEG`` to obtain:
        - ``name_list``: list of volume IDs.
        - ``coor_list``: patch coordinates.
        - ``GT_list``: ground-truth label stacks.
        - ``raw_list``: raw input image stacks.
        """
        self.name_list, self.coor_list, self.GT_list, self.raw_list = train_preprocess_lessMemory_SEG(self)

    #########################################################################
    #########################################################################
    def train(self):
        """
        Run the SEG training loop.

        For each epoch, iterates over training patches, computes BCE and MSE
        losses between predictions and ground truth, updates the model, prints
        progress, periodically saves sample inputs, labels, and outputs, and
        writes model checkpoints to disk.
        """
        Tensor = torch.cuda.FloatTensor
        per_epoch_len = len(self.name_list)
        L1_pixelwise = torch.nn.L1Loss().cuda() 
        L2_pixelwise = torch.nn.MSELoss().cuda()
        BCEloss_function = torch.nn.BCELoss().cuda()

        prev_time = time.time()
        ########################################################################################################################
        # torch.multiprocessing.set_start_method('spawn')
        ########################################################################################################################
        time_start=time.time()
        for epoch in range(0, self.SEG_n_epochs):
            train_data = trainset_SEG(self.name_list, self.coor_list, self.GT_list, self.raw_list)
            trainloader = DataLoader(train_data, batch_size=self.SEG_batch_size, shuffle=True, num_workers=0)
            for index, (input, GT, input_name) in enumerate(trainloader):
                real_A = FFDrealign4(input).cuda()
                real_B = FFDrealign4(GT).cuda()
                fake_B = self.SEG_net(real_A)

                BCE_loss = BCEloss_function(fake_B, real_B)
                L2_loss = L2_pixelwise(fake_B, real_B)
                ################################################################################################################
                self.optimizer.zero_grad() 
                Total_loss = BCE_loss + L2_loss #+ Dloss1 
                Total_loss.backward() 
                self.optimizer.step() 
                ################################################################################################################
                ################################################################################################################
                batches_done = epoch * per_epoch_len + index + 1
                batches_left = self.SEG_n_epochs * per_epoch_len - batches_done
                time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
                prev_time = time.time()
                ################################################################################################################
                if index%(100//self.SEG_batch_size) == 0:
                    time_end=time.time()
                    print_head = 'SEG_TRAIN'
                    print_head_color = UseStyle(print_head, mode = 'bold', fore  = 'red')
                    print_body = '    [Epoch %d/%d]   [Batch %d/%d]   [Total_loss: %f]   [Time_Left: %s]'% (
                        epoch,
                        self.SEG_n_epochs,
                        index,
                        per_epoch_len,
                        Total_loss, 
                        time_left,
                    )
                    print_body_color = UseStyle(print_body, fore  = 'blue')
                    sys.stdout.write("\r  "+print_head_color+print_body_color)
                ################################################################################################################
                if (index+1)%(2000//self.SEG_batch_size) == 0:
                    norm_factor = self.SEG_norm_factor
                    image_name = input_name

                    SEG_out = inv_FFDrealign4(fake_B)
                    save_img_train(input, self.SEG_output_path, epoch, index, input_name, norm_factor, 'SEG_in')
                    save_img_train(GT, self.SEG_output_path, epoch, index, input_name, norm_factor, 'SEG_gt')
                    save_img_train(SEG_out, self.SEG_output_path, epoch, index, input_name, norm_factor, 'SEG_out')

            torch.save(self.SEG_net.state_dict(), self.SEG_pth_save_path +'//'+'SEG_' + str(epoch) + '.pth')


    #########################################################################
    #########################################################################
    def run(self):
        print_head = '==========initialize model=========='
        print_head_color = UseStyle(print_head.upper(), mode = 'bold', fore  = 'red')
        sys.stdout.write("\r  "+print_head_color)
        self.initialize_model()

        print_head = '==========generate patch=========='
        print_head_color = UseStyle(print_head.upper(), mode = 'bold', fore  = 'red')
        sys.stdout.write("\r  "+print_head_color)
        self.generate_patch()

        print_head = '==========train=========='
        print_head_color = UseStyle(print_head.upper(), mode = 'bold', fore  = 'red')
        sys.stdout.write("\r  "+print_head_color)
        self.train()


if __name__ == '__main__':
    SEG_parameters={'GPU':'0,1',
                    'SEG_n_epochs':100,
                    'SEG_batch_size':1,
                    ############################
                    'SEG_img_w':512,
                    'SEG_img_h':512,
                    'SEG_img_s':64,
                    ############################
                    'SEG_lr':0.00005,
                    'SEG_b1':0.5,
                    'SEG_b2':0.999,
                    'SEG_norm_factor':1,
                    ############################
                    'SEG_f_maps':16,
                    'SEG_in_c':4,
                    'SEG_out_c':4,
                    'SEG_down_num':4,
                    ############################
                    'SEG_output_dir':'./results',
                    'SEG_datasets_folder':'seg_circle_mov_wo_bg_0.03_64_300_10_2',
                    'SEG_GT_folder':'mask',
                    'SEG_input_folder':'image',
                    'SEG_datasets_path':'..//datasets',
                    ############################
                    'SEG_use_pretrain':0,
                    'SEG_pretrain_index':'signal_SR_220.pth',
                    'SEG_pretrain_model':'SR_signal_up15_down5_20230302-2250',
                    'SEG_pretrain_path':'pth',
                    ############################
                    'SEG_pth_path':'pth',
                    'SEG_select_img_num':10000,
                    'SEG_train_datasets_size':2000}

    SEG_model = train_seg_net_acc(SEG_parameters)
    SEG_model.run()
