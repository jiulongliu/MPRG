"""Estimators for compressed sensing"""
# pylint: disable = C0301, C0103, C0111, R0914

from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import tensorflow as tf
import mnist_model_def_relu as mnist_model_def
from mnist_utils_relu import save_image
import utils_relu as utils
from scipy.io import savemat

def lasso_estimator(hparams):  # pylint: disable = W0613
    """LASSO estimator"""
    def estimator(x_hat_batch,y_batch_val, A, hparams):
        for i in range(hparams.batch_size):
            colls = x_hat_batch[i]
            temp=colls.argsort()[-100:]
            for j in range(len(colls)):
                if( j not in temp):
                    x_hat_batch[i][j]=0
        return x_hat_batch
    return estimator


def omp_estimator(hparams):
    """OMP estimator"""
    omp_est = OrthogonalMatchingPursuit(n_nonzero_coefs=hparams.omp_k)
    def estimator(A_val, y_batch_val, hparams):
        x_hat_batch = []
        for i in range(hparams.batch_size):
            y_val = y_batch_val[i]
            omp_est.fit(A_val.T, y_val.reshape(hparams.num_measurements))
            x_hat = omp_est.coef_
            x_hat = np.reshape(x_hat, [-1])
            x_hat = np.maximum(np.minimum(x_hat, 1), 0)
            x_hat_batch.append(x_hat)
        x_hat_batch = np.asarray(x_hat_batch)
        return x_hat_batch
    return estimator


def vae_estimator(hparams):

    # Get a session
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) 
    # sess = tf.Session()

    # Set up palceholders
    #A = tf.placeholder(tf.float32, shape=(hparams.batch_size, 100), name='A')
    y_batch = tf.compat.v1.placeholder(tf.float32, shape=(hparams.batch_size, hparams.n_input), name='y_batch')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch,x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)


    if 0:
        init_op = tf.compat.v1.global_variables_initializer()
        sess.run(init_op)       
        saver = tf.compat.v1.train.Saver(var_list=restore_dict)
        # restorer.restore(sess, restore_path)
        
        saver.save(sess, './model_relu.ckpt', global_step=1)
         
        z_batch_val = np.random.randn(hparams.batch_size, 20)
        assign_z_opt_op = z_batch.assign(z_batch_val)
        
        
        sess.run(assign_z_opt_op)
        x_hat_batch_val=sess.run(x_hat_batch) 
        savemat("xs_generatedfromNN.mat", {'xs': x_hat_batch_val})
                    
        
    # measure the estimate

    y_hat_batch = tf.identity(x_hat_batch,name='y2_batch')

    # define all losses
    m_loss1_batch = tf.reduce_mean(input_tensor=tf.abs(y_batch - y_hat_batch), axis=1)
    m_loss2_batch = tf.reduce_mean(input_tensor=(y_batch - y_hat_batch)**2, axis=1)
    zp_loss_batch = tf.reduce_sum(input_tensor=z_batch**2, axis=1)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(input_tensor=total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(input_tensor=m_loss1_batch)
    m_loss2 = tf.reduce_mean(input_tensor=m_loss2_batch)
    zp_loss = tf.reduce_mean(input_tensor=zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.compat.v1.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(y_batch_val,z_batch_val,hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeper(hparams)
        assign_z_opt_op = z_batch.assign(z_batch_val)

        feed_dict = {y_batch: y_batch_val}

        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            sess.run(assign_z_opt_op)
            for j in range(hparams.max_update_iter):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                # logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                # print logging_format.format(i, j, lr_val, total_loss_val,
                #                             m_loss1_val,
                #                             m_loss2_val,
                #                             zp_loss_val)

            x_hat_batch_val,z_batch_val, total_loss_batch_val = sess.run([x_hat_batch,z_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val,z_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator


def learned_estimator(hparams):

    sess = tf.compat.v1.Session()
    y_batch, x_hat_batch, restore_dict = mnist_model_def.end_to_end(hparams)
    restore_path = utils.get_A_restore_path(hparams)

    # Intialize and restore model parameters
    restorer = tf.compat.v1.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):  # pylint: disable = W0613
        """Function that returns the estimated image"""
        x_hat_batch_val = sess.run(x_hat_batch, feed_dict={y_batch: y_batch_val})
        return x_hat_batch_val

    return estimator


def csgm_estimator(hparams):

    # Get a session
    sess = tf.compat.v1.Session()

    # Set up palceholders
    A = tf.compat.v1.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.compat.v1.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.matmul(x_hat_batch, A, name='y_hat_batch')

    # define all losses
    m_loss1_batch = tf.reduce_mean(input_tensor=tf.abs(y_batch - y_hat_batch), axis=1)
    m_loss2_batch = tf.reduce_mean(input_tensor=(y_batch - y_hat_batch)**2, axis=1)
    zp_loss_batch = tf.reduce_sum(input_tensor=z_batch**2, axis=1)

    # define total loss
    total_loss_batch = hparams.mloss1_weight * m_loss1_batch \
                     + hparams.mloss2_weight * m_loss2_batch \
                     + hparams.zprior_weight * zp_loss_batch
    total_loss = tf.reduce_mean(input_tensor=total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(input_tensor=m_loss1_batch)
    m_loss2 = tf.reduce_mean(input_tensor=m_loss2_batch)
    zp_loss = tf.reduce_mean(input_tensor=zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.compat.v1.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeperCSGM(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}
        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print(logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val))

                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator

def prGAN_estimator(hparams):

    # Get a session
    sess = tf.compat.v1.Session()

    # Set up palceholders
    A = tf.compat.v1.placeholder(tf.float32, shape=(hparams.n_input, hparams.num_measurements), name='A')
    y_batch = tf.compat.v1.placeholder(tf.float32, shape=(hparams.batch_size, hparams.num_measurements), name='y_batch')

    # Create the generator
    # TODO: Move z_batch definition here
    z_batch, x_hat_batch, restore_path, restore_dict = mnist_model_def.vae_gen(hparams)

    # measure the estimate
    if hparams.measurement_type == 'project':
        y_hat_batch = tf.identity(x_hat_batch, name='y_hat_batch')
    else:
        y_hat_batch = tf.abs(tf.matmul(x_hat_batch, A, name='y_hat_batch'))

    # define all losses
    m_loss1_batch = tf.reduce_mean(input_tensor=tf.abs(y_batch - y_hat_batch), axis=1)
    m_loss2_batch = tf.reduce_mean(input_tensor=tf.square(y_batch - y_hat_batch), axis=0)
    zp_loss_batch = tf.reduce_sum(input_tensor=z_batch**2, axis=1)

    # define total loss
    total_loss_batch =  hparams.mloss2_weight * m_loss2_batch \

    total_loss = tf.reduce_mean(input_tensor=total_loss_batch)

    # Compute means for logging
    m_loss1 = tf.reduce_mean(input_tensor=m_loss1_batch)
    m_loss2 = tf.reduce_mean(input_tensor=m_loss2_batch)
    zp_loss = tf.reduce_mean(input_tensor=zp_loss_batch)

    # Set up gradient descent
    var_list = [z_batch]
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = utils.get_learning_rate(global_step, hparams)
    opt = utils.get_optimizer(learning_rate, hparams)
    update_op = opt.minimize(total_loss, var_list=var_list, global_step=global_step, name='update_op')
    opt_reinit_op = utils.get_opt_reinit_op(opt, var_list, global_step)

    # Intialize and restore model parameters
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    restorer = tf.compat.v1.train.Saver(var_list=restore_dict)
    restorer.restore(sess, restore_path)

    def estimator(A_val, y_batch_val, hparams):
        """Function that returns the estimated image"""
        best_keeper = utils.BestKeeperCSGM(hparams)
        if hparams.measurement_type == 'project':
            feed_dict = {y_batch: y_batch_val}
        else:
            feed_dict = {A: A_val, y_batch: y_batch_val}
        for i in range(hparams.num_random_restarts):
            sess.run(opt_reinit_op)
            for j in range(hparams.max_update_iter):
                _, lr_val, total_loss_val, \
                m_loss1_val, \
                m_loss2_val, \
                zp_loss_val = sess.run([update_op, learning_rate, total_loss,
                                        m_loss1,
                                        m_loss2,
                                        zp_loss], feed_dict=feed_dict)
                logging_format = 'rr {} iter {} lr {} total_loss {} m_loss1 {} m_loss2 {} zp_loss {}'
                print(logging_format.format(i, j, lr_val, total_loss_val,
                                            m_loss1_val,
                                            m_loss2_val,
                                            zp_loss_val))

                if hparams.gif and ((j % hparams.gif_iter) == 0):
                    images = sess.run(x_hat_batch, feed_dict=feed_dict)
                    for im_num, image in enumerate(images):
                        save_dir = '{0}/{1}/'.format(hparams.gif_dir, im_num)
                        utils.set_up_dir(save_dir)
                        save_path = save_dir + '{0}.png'.format(j)
                        image = image.reshape(hparams.image_shape)
                        save_image(image, save_path)

            x_hat_batch_val, total_loss_batch_val = sess.run([x_hat_batch, total_loss_batch], feed_dict=feed_dict)
            best_keeper.report(x_hat_batch_val, total_loss_batch_val)
        return best_keeper.get_best()

    return estimator