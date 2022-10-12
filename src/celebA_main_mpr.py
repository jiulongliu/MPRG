from __future__ import division
import os
from argparse import ArgumentParser
import numpy as np
import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy import linalg as LA
import tensorflow as tf

import numpy as np
import time
def main(hparams):
    hparams.n_input = np.prod(hparams.image_shape)
    
    maxiter = hparams.max_outer_iter
    utils.print_hparams(hparams)
    
    

    xs_dict = model_input(hparams) # returns the images
    time_elapsed =np.zeros([len(hparams.num_outer_measurement_ls),hparams.num_experiments,len(hparams.noise_std_ls),len(xs_dict.keys()),len(hparams.method_ls)])
    
    # print('len xs_dict',len(xs_dict))
    # meai=0
    for n_measurement in hparams.num_outer_measurement_ls:

        hparams.num_outer_measurements = n_measurement
        hparams.model_type='dcgan'
        hparams.model_types=['dcgan']
        utils.setup_checkpointing(hparams)  
        # for beta in hparams.beta_ls:
        
        for ri in range(0,hparams.num_experiments):
            A_outer = utils.get_outer_A(hparams) # Created the random matric A
            
            for ni in range(len(hparams.noise_std_ls)):
                hparams.noise_std = hparams.noise_std_ls[ni]
                noise_batch = hparams.noise_std * np.random.randn(hparams.batch_size, hparams.num_outer_measurements)
                
                
                x_hats_dict = {}
                y_dict = {} 
                for method in hparams.method_ls:
                    x_hats_dict[method] = {}  
                
                for key, x in xs_dict.items():
                    x_batch_dict = {}
                    x_batch_dict[key] = x #placing images in dictionary

                    x_coll = [x.reshape(1, hparams.n_input) for _, x in x_batch_dict.items()] #Generates the columns of input x
                    x_batch = np.concatenate(x_coll) # Generates entire X   
                    
                    if hparams.nonlinear_model == 'abs(Ax)+eta':
                        y_batch_outer = np.abs(np.matmul(x_batch/LA.norm(x_batch, axis=(1),keepdims=True), A_outer))+noise_batch
                    elif hparams.nonlinear_model == 'abs(Ax+eta)+5tanh(abs(Ax))':
                        Ax_abs = np.abs(np.matmul(x_batch/LA.norm(x_batch, axis=(1),keepdims=True), A_outer)+noise_batch)
                        y_batch_outer = Ax_abs+5.0*np.tanh(Ax_abs)                         
                    elif hparams.nonlinear_model == '4sq(Ax)+3sin(abs(Ax))Ax+eta':
                        Ax_abs = np.abs(np.matmul(x_batch/LA.norm(x_batch, axis=(1),keepdims=True), A_outer))  
                        y_batch_outer = 4*(Ax_abs**2)+3.0*Ax_abs*np.sin(Ax_abs) + noise_batch                            
                    else:
                        print(hparams.nonlinear_model+'is not in [abs(Ax)+eta,abs(Ax+eta)+5tanh(abs(Ax)),4sq(Ax)+3sin(abs(Ax))Ax+eta]!')
                        break                    
                    
                    
                   
                    Vdiag_batch = np.zeros((x_batch.shape[0],x_batch.shape[1]),dtype=np.float32)
                    for si in range(y_batch_outer.shape[1]):
                        aat_diag=A_outer[:,si]**2#-np.eye(x_batch.shape[1], dtype=np.float32)
                        for bi in range(y_batch_outer.shape[0]):                                 
                                Vdiag_batch[bi,:]=Vdiag_batch[bi,:]+(y_batch_outer[bi,si])*aat_diag
                    Vdiag_batch = Vdiag_batch/y_batch_outer.shape[1]                     
                    xcidx=Vdiag_batch.argmax(1)                     
                    x_main_batch = 0.0 * x_batch        

                        
                    for si in range(y_batch_outer.shape[1]):
                        #-np.eye(x_batch.shape[1], dtype=np.float32)
                        for bi in range(y_batch_outer.shape[0]):  
                                aat_p=A_outer[:,si]*A_outer[xcidx[bi],si]
                                x_main_batch[bi,:]=x_main_batch[bi,:]+(y_batch_outer[bi,si])*aat_p
                    x_main_batch = x_main_batch/y_batch_outer.shape[1]   
                           
                        
                        
                    # x_hat_ini0_batch =x_main_batch/LA.norm(x_main_batch, axis=(1),keepdims=True)
    
                    # MPRG Step 1
                    hparams.model_type='dcgan'
                    hparams.model_types=['dcgan']
                    tf.compat.v1.reset_default_graph()
                    estimators = utils.get_estimators(hparams)
                    measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                    estimator = estimators['dcgan']  
                    z_opt_batch = np.random.randn(hparams.batch_size, 100)
                    for k in range(20):   

                        x_est_batch = np.matmul(y_batch_outer*np.matmul(x_main_batch, A_outer), A_outer.T)/hparams.num_outer_measurements
                     
                        x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                        x_hat_batch = estimator(x_est_batch,z_opt_batch, hparams)                        
                        x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True)
                        x_main_batch = x_hat_batch
                    x_hat_ini1_batch = x_hat_batch
                    
                    y_batch_mean = np.mean(y_batch_outer, axis=1, keepdims=True)

                    nv_tilde_batch = np.mean((y_batch_outer-y_batch_mean)*(np.matmul(x_main_batch, A_outer)**2), axis=1, keepdims=True)#.reshape(y_batch_outer.shape[0],1)
                    y_tilde_batch  = (y_batch_outer-y_batch_mean)*np.matmul(x_main_batch, A_outer)
                    

               
                    for method in hparams.method_ls:                   
                        if method == 'PPower':
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.compat.v1.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            estimator = estimators['dcgan']                            
                            x_main_batch = x_hat_ini1_batch
                            for k in range(30):            

                                x_est_batch = np.matmul(y_batch_outer*np.matmul(x_main_batch, A_outer), A_outer.T)/hparams.num_outer_measurements
                                print('current:',method,'iter:',k+20)    
                              
                                x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)   
                                x_hat_batch = estimator(x_est_batch,z_opt_batch, hparams)
                                x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True)#*LA.norm(x_batch, axis=(1),keepdims=True) #x_batch
                                x_main_batch = x_hat_batch                                       

    
                
                        # MPRG step 2  
                        elif method == 'MPRG':
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.compat.v1.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            estimator = estimators['dcgan']  
                            x_main_batch = x_hat_ini1_batch    
                            for k in range(30):
                                nv_tilde_batch = np.mean((y_batch_outer-y_batch_mean)*(np.matmul(x_main_batch, A_outer)**2), axis=1, keepdims=True)
                                y_tilde_batch  = (y_batch_outer-y_batch_mean)*np.matmul(x_main_batch, A_outer)                                 
                                x_est_batch = x_main_batch -  (1.0/nv_tilde_batch)/hparams.num_outer_measurements *(np.matmul(( nv_tilde_batch*np.matmul(x_main_batch, A_outer)-y_tilde_batch), A_outer.T))          #hparams.outer_learning_rate/hparams.num_outer_measurements       
            
                                print('current:',method,'iter:',k+20) 
                                x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                                z_opt_batch = np.random.randn(hparams.batch_size, 100) 
                                x_hat_batch = estimator(x_est_batch,z_opt_batch, hparams) # Projectin on the GAN
                                x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True)
                                x_main_batch = x_hat_batch
    
                        elif method == 'APPGD':
                            x_main_batch = x_hat_ini1_batch
                            hparams.model_type='dcgan'
                            hparams.model_types=['dcgan']
                            tf.compat.v1.reset_default_graph()
                            estimators = utils.get_estimators(hparams)
                            measurement_losses, l2_losses = utils.load_checkpoints(hparams)
                            estimator = estimators['dcgan']                        
                            for k in range(30):
                                print('current:',method,'iter:',k+20)   
                                x_est_batch = x_main_batch + hparams.outer_learning_rate/hparams.num_outer_measurements * (np.matmul((y_batch_outer*np.sign(np.matmul(x_main_batch, A_outer)) - np.matmul(x_main_batch, A_outer)), A_outer.T))                 
                                x_est_batch=x_est_batch/LA.norm(x_est_batch, axis=(1),keepdims=True)*LA.norm(x_batch, axis=(1),keepdims=True)
                                z_opt_batch = np.random.randn(hparams.batch_size, 100) 
                                x_hat_batch = estimator(x_est_batch,z_opt_batch, hparams) # Projectin on the GAN
                                x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True)
                                x_main_batch = x_hat_batch
                            x_hat_batch=x_hat_batch/LA.norm(x_hat_batch, axis=(1),keepdims=True)                                            
 
                        else:
                            print(method+' is not in [PPower,APPGD,MPRG]!')
                            
                        x_hat_batch = x_hat_batch*LA.norm(x_batch, axis=(1),keepdims=True)#* 2.0 - 1.0
                        x_main_batch = x_hat_batch    
                        print('Saving results of '+ method+'\n')       
                        for i, key_ in enumerate(x_batch_dict.keys()):                                      
                        # for i, key in enumerate(x_batch_dict.keys()):
                            x = xs_dict[key]
                            y = y_batch_outer[i]
                            x_hat = x_hat_batch[i]
                
                            
                            x_hats_dict[method][key] = x_hat
                            y_dict[key] = y
                            measurement_losses[hparams.model_type][key] = 1.0
                            l2_losses[hparams.model_type][key] = utils.get_l2_loss(x_hat, x)
    
                            
                        print('Processed up to image {0} / {1}'.format(key + 1, len(xs_dict)))
            
                        # Checkpointing
                        if (hparams.save_images) and ((key + 1) % hparams.checkpoint_iter == 0):
                            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
                            #x_hats_dict = {'dcgan' : {}}
                            print('\nProcessed and saved first ', key + 1, 'images\n')
                
                        # x_batch_dict = {}
                    
                        # Final checkpoint
                        if hparams.save_images:
                            utils.checkpoint(x_hats_dict, measurement_losses, l2_losses, save_image, hparams)
                            print('\nProcessed and saved all {0} image(s)\n'.format(len(xs_dict)))
                    
                        if hparams.print_stats:
                            for model_type in hparams.model_types:
                                print(model_type)
                                mean_m_loss = np.mean(measurement_losses[model_type].values())
                                mean_l2_loss = np.mean(l2_losses[model_type].values())
                                print('mean measurement loss = {0}'.format(mean_m_loss))
                                print('mean l2 loss = {0}'.format(mean_l2_loss))
                
                if hparams.image_matrix > 0:
                    print('images plot')
                    outputdir = 'res/MPR_%s_%s/'%(hparams.dataset, hparams.nonlinear_model)
                    hparams.savepath=outputdir+'MPR_%s_%s_m_%d_eta_%0.3f_r_%d.png'%(hparams.dataset, hparams.nonlinear_model, hparams.num_outer_measurements,hparams.noise_std,ri)
                    if not os.path.exists('res/'):
                        os.mkdir('res/')
                    if not os.path.exists(outputdir):
                        os.mkdir(outputdir)
                    
                    
                    utils.image_matrix_mls(xs_dict, x_hats_dict, view_image, hparams)
                    img_rec_ls=[]
                    for mi in hparams.method_ls:
                        img_rec_ls = img_rec_ls+[np.stack([vi for vi in x_hats_dict[mi].values()] , axis=0)]    
                    np.savez(hparams.savepath[:-4]+'.npz',img_gd=np.stack([vi for vi in xs_dict.values()] , axis=0),img_rec=np.stack(img_rec_ls, axis=0))

                    
      
    np.savez(outputdir+'%s_%s_time_elapsed.npz'%(hparams.dataset, hparams.nonlinear_model),time_elapsed=time_elapsed)






 







if __name__ == '__main__':

    PARSER = ArgumentParser()

    # Pretrained model
    PARSER.add_argument('--pretrained-model-dir', type=str, default='./models/celebA_64_64/', help='Directory containing pretrained model')

    # Input
    PARSER.add_argument('--dataset', type=str, default='celebA', help='Dataset to use')
    PARSER.add_argument('--input-type', type=str, default='full-input', help='Where to take input from')
    PARSER.add_argument('--input-path-pattern', type=str, default='./data/celebAtest/*.jpg', help='Pattern to match to get images')
    PARSER.add_argument('--num-input-images', type=int, default=8, help='number of input images')
    PARSER.add_argument('--batch-size', type=int, default=1, help='How many examples are processed together')

    # Problem definition
    PARSER.add_argument('--measurement-type', type=str, default='gaussian', help='measurement type: as of now supports only gaussian')
    PARSER.add_argument('--noise-std', type=float, default=0.1, help='std dev of noise')
    PARSER.add_argument('--noise-std-ls', metavar='N', type=float, default=[0.1,0.05,0.01], nargs='+', help='std dev of noise')
    PARSER.add_argument('--nonlinear-model', type=str, default='abs(Ax)+eta', help='non linear model')
    # Measurement type specific hparams

    PARSER.add_argument('--num-outer-measurement-ls', metavar='N', type=int, default=[300], nargs='+',
                    help='number of measurements') #type=int, default=500, help='number of gaussian measurements(outer)')
    PARSER.add_argument('--beta-ls', metavar='N', type=float, default=[300], nargs='+',
                    help='list of beta') #type=int, default=500, help='number of gaussian measurements(outer)')
    PARSER.add_argument('--method-ls', metavar='N', type=str, default=['MPRS','APPGD','MPRG'], nargs='+', help='MPRS, APPGD, MPRG')
    
    # Model
    PARSER.add_argument('--num-experiments', type=int, default=10, help='number of experiments')    
    PARSER.add_argument('--model-types', type=str, nargs='+', default=['dcgan'], help='model(s) used for estimation')
    PARSER.add_argument('--mloss1_weight', type=float, default=0.0, help='L1 measurement loss weight')
    PARSER.add_argument('--mloss2_weight', type=float, default=1.0, help='L2 measurement loss weight')
    PARSER.add_argument('--zprior_weight', type=float, default=0.001, help='weight on z prior')
    PARSER.add_argument('--dloss1_weight', type=float, default=0.0, help='-log(D(G(z))')
    PARSER.add_argument('--dloss2_weight', type=float, default=0.0, help='log(1-D(G(z))')

    # NN specfic hparams
    PARSER.add_argument('--optimizer-type', type=str, default='adam', help='Optimizer type')
    PARSER.add_argument('--learning-rate', type=float, default=0.1, help='learning rate')
    PARSER.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    PARSER.add_argument('--max-update-iter', type=int, default=100, help='maximum updates to z')
    PARSER.add_argument('--num-random-restarts', type=int, default=4, help='number of random restarts')
    PARSER.add_argument('--decay-lr', action='store_true', help='whether to decay learning rate')
    PARSER.add_argument('--outer-learning-rate', type=float, default=0.9, help='learning rate of outer loop GD')
    PARSER.add_argument('--max-outer-iter', type=int, default=50, help='maximum no. of iterations for outer loop GD')
    # LASSO specific hparams
    PARSER.add_argument('--lmbd', type=float, default=0.02, help='lambda : regularization parameter for LASSO')
    PARSER.add_argument('--lasso-solver', type=str, default='sklearn', help='Solver for LASSO')
    # Output 
    PARSER.add_argument('--lazy', action='store_true', help='whether the evaluation is lazy')
    PARSER.add_argument('--save-images', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--save-stats', action='store_true', help='whether to save estimated images')
    PARSER.add_argument('--print-stats', action='store_true', help='whether to print statistics')
    PARSER.add_argument('--checkpoint-iter', type=int, default=50, help='checkpoint every x batches')
    PARSER.add_argument('--image-matrix', type=int, default=2,
                        help='''
                                0 = 00 =      no       image matrix,
                                1 = 01 =          show image matrix
                                2 = 10 = save          image matrix
                                3 = 11 = save and show image matrix
                             '''
                        )
    PARSER.add_argument('--gif', action='store_true', help='whether to create a gif')
    PARSER.add_argument('--gif-iter', type=int, default=1, help='save gif frame every x iter')
    PARSER.add_argument('--gif-dir', type=str, default='', help='where to store gif frames')
    HPARAMS = PARSER.parse_args()



    if HPARAMS.dataset == 'mnist':
        HPARAMS.image_shape = (28, 28, 1)
        from mnist_input import model_input
        from mnist_utils import view_image, save_image
    elif HPARAMS.dataset == 'celebA':
        HPARAMS.image_shape = (64, 64, 3)
        from celebA_input import model_input
        from celebA_utils import view_image, save_image
    else:
        raise NotImplementedError
    main(HPARAMS)
