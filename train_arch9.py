from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
from functools import partial
import json
import traceback

import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data_noise_in as data
import models_712 as models
import os
def matching_loss(x1, x2, y1, y2, margin):
    return tf.reduce_mean(tf.nn.relu(models.inner_product(x1, x2) - models.inner_product(y1, y2) + margin))

parser = argparse.ArgumentParser()
# settings
dataroot_default = './data/CelebA'
parser.add_argument('--dataroot', type=str, default=dataroot_default)
parser.add_argument('--dataset', type=str, default='celeba')
parser.add_argument('--gpu', type=str, default='0,1',
                    help='Specify which gpu to use by `CUDA_VISIBLE_DEVICES=num python train.py **kwargs`\
                          or `python train.py --gpu num` if you\'re running on a multi-gpu enviroment.\
                          You need to do nothing if your\'re running on a single-gpu environment or\
                          the gpu is assigned by a resource manager program.')
parser.add_argument('--threads', type=int, default=-1,
                    help='Control parallel computation threads,\
                          please leave it as is if no heavy cpu burden is observed.')

# model
# att_default = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
#                'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
att_default = ['Young', 'Mouth_Slightly_Open', 'Smiling', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
               'Receding_Hairline', 'Bangs', 'Male', 'No_Beard', 'Mustache', 'Goatee','Sideburns']
parser.add_argument('--atts', default=att_default, choices=data.Celeba.att_dict.keys(), nargs='+',
                    help='Attributes to modify by the model')
parser.add_argument('--img_size', type=int, default=128, help='input image size')
# generator
parser.add_argument('--n_downblks_gen', type=int, default=3)
parser.add_argument('--n_intermedblks_gen', type=int, default=2)
parser.add_argument('--n_upblks_gen', type=int, default=3)
parser.add_argument('--ch_gen', type=int, default=32)
# mappingnet
parser.add_argument('--n_layers_map', type=int, default=4)
parser.add_argument('--n_mlp_map', type=int, default=3)
parser.add_argument('--fc_dim_map', type=int, default=64)
parser.add_argument('--dim_noise', type=int, default=16)
# encoder
parser.add_argument('--n_downblks_enc', type=int, default=1)
parser.add_argument('--n_intermedblks_enc', type=int, default=3)
parser.add_argument('--n_mlp_enc', type=int, default=3)
parser.add_argument('--ch_enc', type=int, default=16)
# discriminator
parser.add_argument('--n_resblks_dis', type=int, default=4)
parser.add_argument('--ch_dis', type=int, default=16)

parser.add_argument('--rec_loss_weight', type=float, default=100.0)

# training
parser.add_argument('--mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
parser.add_argument('--epoch', type=int, default=100, help='# of epochs') #200
parser.add_argument('--init_epoch', type=int, default=100, help='# of epochs with init lr.') # 100
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--n_d', type=int, default=5, help='# of d updates per g update')
parser.add_argument('--n_sample', type=int, default=8, help='# of sample images')
parser.add_argument('--save_freq', type=int, default=0,
                    help='save model evary save_freq iters, 0 means to save evary epoch.')
parser.add_argument('--sample_freq', type=int, default=0,
                    help='eval on validation set every sample_freq iters, 0 means to save every epoch.')

# others
parser.add_argument('--use_cropped_img', action='store_true')
parser.add_argument('--experiment_name', default=datetime.datetime.now().strftime("%Y.%m.%d-%H%M%S"))
parser.add_argument('--num_ckpt', type=int, default=10)
parser.add_argument('--clear', default=False, action='store_true')

args = parser.parse_args()
num_gpu = 1
n_att = len(args.atts)
pylib.mkdir('./output/%s' % args.experiment_name)
with open('./output/%s/setting.txt' % args.experiment_name, 'w') as f:
    f.write(json.dumps(vars(args), indent=4, separators=(',', ':')))

# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
if args.threads >= 0:
    cpu_config = tf.ConfigProto(intra_op_parallelism_threads=args.threads // 2,
                                inter_op_parallelism_threads=args.threads // 2,
                                device_count={'CPU': args.threads})
    sess = tf.Session(config=cpu_config)
else:
    sess = tl.session()

crop_ = not args.use_cropped_img
if args.dataset == 'celeba':
    tr_data = data.Celeba(args.dataroot, att_default, args.img_size, args.batch_size, part='train', sess=sess, crop=crop_, is_tfrecord = True)
    val_data = data.Celeba(args.dataroot, args.atts, args.img_size, args.n_sample, part='val', shuffle=False, sess=sess, crop=crop_)
else:
    tr_data = data.x2y(args.dataset, args.dataroot, args.img_size, args.batch_size, part='train', sess=sess, is_tfrecord = True)
    val_data = data.x2y(args.dataset, args.dataroot, args.img_size, args.n_sample, part='val', shuffle=False, sess=sess, is_tfrecord = False)

# models
Generator = partial(models.Generator2_wRelu, n_downblks = args.n_downblks_gen, n_intermedblks = args.n_intermedblks_gen, n_upblks = args.n_upblks_gen, ch = args.ch_gen)
MappingNet = partial(models.MappingNet_multiStream_deconv_wIN_wRelu_concat, n_mlp = args.n_mlp_map, n_layers = args.n_layers_map, fc_dim = args.fc_dim_map, ch = args.ch_enc*2)
Encoder = partial(models.Encoder4_wIN_wRelu_concat,n_downblks = args.n_downblks_enc, n_intermedblks = args.n_intermedblks_enc, n_mlp = args.n_mlp_enc, ch = args.ch_enc)
# Discriminator = partial(models.PatchDiscriminator, n_att=n_att, n_resblks = args.n_resblks_dis, ch = args.ch_dis)
# Discriminator = partial(models.Discriminator_multiTask, n_att=n_att, n_resblks = args.n_resblks_dis, ch = args.ch_dis)
Discriminator = partial(models.Discriminator, n_att=n_att, n_resblks = args.n_resblks_dis, ch = args.ch_dis)


# inputs
r_M = tf.get_variable(name='G_R', shape=[182000, args.dim_noise], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
lr = tf.placeholder(dtype=tf.float32, shape=[])
xs_s = tr_data.batch_op[0]
ls_s = tf.to_float(tr_data.batch_op[1])
rs_s = tr_data.batch_op[2]
if args.dataset == 'celeba':
    # b_s = tf.random_shuffle(a_s)
    permuted_index = tf.random_shuffle(tf.range(args.batch_size))
    xt_s = tf.gather(xs_s, permuted_index)
    lt_s = tf.gather(ls_s, permuted_index)
    rt_i = tf.gather(rs_s, permuted_index)
    rt_s = tf.reshape(tf.gather(r_M, rt_i), [rt_i.shape[0], args.dim_noise])
else:
    pass



d_opt = tf.train.AdamOptimizer(lr, beta1=0.5)
g_opt = tf.train.AdamOptimizer(lr, beta1=0.5)

tower_d_grads = []
tower_g_grads = []

tower_d_loss_gan = []
tower_gp = []
tower_d_loss_cls = []

tower_g_loss_sim = []
tower_g_loss_r = []
tower_g_loss_r0 = []
tower_g_loss_gan = []
tower_g_loss_cls = []
tower_g_loss_cyc = []
tower_g_loss_reg = []
tower_g_loss_rec = []
tower_g_loss_interp = []

xs_sample = tf.placeholder(tf.float32, shape=[None, args.img_size, args.img_size, 3])
lt_sample = tf.placeholder(tf.float32, shape=[None, n_att])
ls_sample = tf.placeholder(tf.float32, shape=[None, n_att])

with tf.variable_scope(tf.get_variable_scope()):
    for i in range(1):
            with tf.name_scope("tower_%d" % i):
                xs = xs_s[i * args.batch_size // num_gpu: (i + 1) * args.batch_size // num_gpu]
                xt = xt_s[i * args.batch_size // num_gpu: (i + 1) * args.batch_size // num_gpu]
                rt = rt_s[i * args.batch_size // num_gpu: (i + 1) * args.batch_size // num_gpu]
                ls = ls_s[i * args.batch_size // num_gpu: (i + 1) * args.batch_size // num_gpu]
                lt = lt_s[i * args.batch_size // num_gpu: (i + 1) * args.batch_size // num_gpu]
                rs = rs_s[i * args.batch_size // num_gpu: (i + 1) * args.batch_size // num_gpu]

                # generate
                r_1 = tf.random_normal((args.batch_size // num_gpu, args.dim_noise))
                s_rand_t1 = MappingNet(r_1, lt - ls)
                r_0 = rt
                s_rand_t = MappingNet(r_0, lt - ls)
                s_rand_s, _ = Encoder(xs, ls - lt)
                s_rand1 = s_rand_s + s_rand_t1
                xg_rand1 = Generator(xs, s_targ=s_rand1)

                s_targ_t, _ = Encoder(xt, lt - ls)

                loss_latent = tf.losses.mean_squared_error(s_targ_t, s_rand_t)
                r_grads = tf.gradients(loss_latent, r_0)[0]
                grads_l2 = tf.reshape(tf.sqrt(tf.reduce_mean(tf.square(r_grads), 1)), [args.batch_size, 1])
                normalized_grads = (0.9 / grads_l2 + 5.0) * r_grads
                r = tf.stop_gradient(tf.clip_by_value((r_0 + normalized_grads), -1.0, 1.0))  #
                tower_assign = []
                for index_i in range(args.batch_size):
                    tower_assign.append(tf.assign(r_M[tf.reshape(rt_i[index_i], [])], r[index_i]))

                s_rand_t = MappingNet(r, lt - ls)
                s_rand = s_rand_s + s_rand_t
                xg_rand = Generator(xs, s_targ=s_rand)
                s_targ = s_targ_t + s_rand_s
                xg_targ = Generator(xs, s_targ=s_targ)

                alpha = tf.random_uniform((args.batch_size // num_gpu, 1, 1, 1), maxval=1)
                s_interp = alpha * s_rand + (1 - alpha) * s_targ
                xg_interp = Generator(xs, s_targ=s_interp)


                s_targ_t_rand, _ = Encoder(tf.stop_gradient(xg_rand), lt - ls)
                s_targ_rand = s_targ_t_rand + s_rand_s

                s_targ_t_targ, _ = Encoder(tf.stop_gradient(xg_targ), lt - ls)
                s_targ_targ = s_targ_t_targ + s_rand_s

                s_rand_rec_t = MappingNet(r, ls - ls)
                s_rand_rec_s,_ = Encoder(xs, ls - ls)
                s_rand_rec = s_rand_rec_t + s_rand_rec_s
                xg_rand_rec = Generator(xs, s_targ=s_rand_rec)

                s_targ_t_rec, _ = Encoder(xt, ls - ls)
                s_targ_rec = s_targ_t_rec + s_rand_rec_s
                xg_targ_rec = Generator(xs, s_targ=s_targ_rec)

                
                l_1 = tf.ones_like(lt)
                s_targ_t_targ_oth, _ = Encoder(xg_rand, (l_1-tf.abs(lt - ls))*(l_1-2.0*ls))
                s_rand_t_oth, _ = Encoder(xs, (l_1-tf.abs(lt - ls))*(l_1-2.0*ls))

                # discriminate
                xs_logit_gan, xs_logit_att,_= Discriminator(xs)
                xgr_logit_gan, xgr_logit_att,xgr_logit_att_feature = Discriminator(xg_rand)
                xgt_logit_gan, xgt_logit_att,xgt_logit_att_feature= Discriminator(xg_targ)
                xgi_logit_gan, xgi_logit_att, xgi_logit_att_feature = Discriminator(xg_interp)

                # discriminator losses
                if args.mode == 'wgan':  # wgan-gp
                    wd = tf.reduce_mean(xs_logit_gan) - (tf.reduce_mean(xgr_logit_gan) + tf.reduce_mean(xgt_logit_gan) + tf.reduce_mean(xgi_logit_gan)) / 3.0
                    d_loss_gan = -wd
                    gp = (models.gradient_penalty(Discriminator, xs, xg_rand) + models.gradient_penalty(Discriminator,xs, xg_targ) + \
                          models.gradient_penalty(Discriminator, xs, xg_interp)) / 3.0

                d_loss_cls = tf.losses.sigmoid_cross_entropy(ls, xs_logit_att)

                tower_d_loss_gan.append(d_loss_gan)
                tower_gp.append(gp)
                tower_d_loss_cls.append(d_loss_cls)

                d_loss = d_loss_gan + gp * 10.0 + d_loss_cls

                # generator losses
                if args.mode == 'wgan':
                    g_loss_gan = -(tf.reduce_mean(xgr_logit_gan) + tf.reduce_mean(xgt_logit_gan) + tf.reduce_mean(xgi_logit_gan)) / 3.0

                g_loss_cls = (tf.losses.sigmoid_cross_entropy(lt, xgr_logit_att) +
                              tf.losses.sigmoid_cross_entropy(lt, xgt_logit_att) +
                              tf.losses.sigmoid_cross_entropy(lt, xgi_logit_att)) / 3.0

                g_loss_rec = (tf.losses.absolute_difference(xs, xg_rand_rec) + tf.losses.absolute_difference(xs, xg_targ_rec)) / 2.0

                g_loss_cyc =  (tf.losses.absolute_difference(s_rand, s_targ_rand) + tf.losses.absolute_difference(s_targ, s_targ_targ) + tf.losses.absolute_difference(s_targ_t_targ_oth, s_rand_t_oth)) / 3.0

                g_loss_sim = tf.losses.absolute_difference(s_rand, s_targ)

                g_loss_ModeSeeking = tf.losses.absolute_difference(r_1, r)/(tf.losses.absolute_difference(tf.stop_gradient(xg_rand1), xg_rand) + 0.00001)

                tower_g_loss_gan.append(g_loss_gan)
                tower_g_loss_cls.append(g_loss_ModeSeeking)
                tower_g_loss_sim.append(g_loss_sim)
                tower_g_loss_r.append(tf.reduce_mean(r))
                tower_g_loss_r0.append(tf.reduce_mean(r_0))
                tower_g_loss_rec.append(g_loss_rec)
                tower_g_loss_cyc.append(tf.reduce_mean(g_loss_cyc))

                g_loss = g_loss_gan + g_loss_cls * 10.0 + g_loss_rec * args.rec_loss_weight + g_loss_cyc * args.rec_loss_weight /2.0 + g_loss_sim * args.rec_loss_weight /2.0 + g_loss_ModeSeeking * 0.005#+ g_loss_interp * 10

                # optimize
                tf.get_variable_scope().reuse_variables()
                d_var = tl.trainable_variables('D')
                d_grads = d_opt.compute_gradients(d_loss, var_list=d_var)
                tower_d_grads.append(d_grads)

                g_var = tl.trainable_variables('G')
                g_grads = g_opt.compute_gradients(g_loss, var_list=g_var)
                tower_g_grads.append(g_grads)



# optimize
d_grads = models.average_gradients(tower_d_grads)
d_step = d_opt.apply_gradients(d_grads)

g_grads = models.average_gradients(tower_g_grads)
g_step = g_opt.apply_gradients(g_grads)

# summary
d_summary = tl.summary({
    tf.reduce_mean(tf.stack(tower_d_loss_gan)): 'd_loss_gan',
    tf.reduce_mean(tf.stack(tower_gp)): 'gp',
    tf.reduce_mean(tf.stack(tower_d_loss_cls)): 'd_loss_cls',
}, scope='D')

g_summary = tl.summary({
    tf.reduce_mean(tf.stack(tower_g_loss_gan)): 'g_loss_gan',
    tf.reduce_mean(tf.stack(tower_g_loss_cls)): 'g_loss_cls',
    tf.reduce_mean(tf.stack(tower_g_loss_rec)): 'g_loss_rec',
    tf.reduce_mean(tf.stack(tower_g_loss_cyc)): 'g_loss_cyc',
    tf.reduce_mean(tf.stack(tower_g_loss_sim)): 'g_loss_sim',
    tf.reduce_mean(tf.stack(tower_g_loss_r)): 'g_loss_r',
    tf.reduce_mean(tf.stack(tower_g_loss_r0)): 'g_loss_r0'
}, scope='G')

lr_summary = tl.summary({lr: 'lr'}, scope='Learning_Rate')

d_summary = tf.summary.merge([d_summary, lr_summary])

# validation

test_label = lt_sample - ls_sample
r_sample = tf.random_normal((args.n_sample, args.dim_noise))
s_rand_sample_t = MappingNet(r_sample, test_label)
s_rand_sample_s, _ = Encoder(xs_sample, -test_label)
s_rand_sample = s_rand_sample_s + s_rand_sample_t
xg_rand_sample = Generator(xs_sample, s_targ = s_rand_sample)

# ==============================================================================
# =                                    train                                   =
# ==============================================================================

# iteration counter
it_cnt, update_cnt = tl.counter()

# saver
saver = tf.train.Saver(max_to_keep=args.num_ckpt)

# summary writer
summary_writer = tf.summary.FileWriter('./output/%s/summaries' % args.experiment_name, sess.graph)

# initialization
ckpt_dir = './output/%s/checkpoints' % args.experiment_name
pylib.mkdir(ckpt_dir)

try:
    assert args.clear == False
    tl.load_checkpoint(ckpt_dir, sess)
except:
    print('NOTE: Initializing all parameters...')
    sess.run(tf.global_variables_initializer())

# train
try:
    xa_sample_ipt, a_sample_ipt = val_data.get_next()
    b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
    for i in range(len(args.atts)):
        tmp = np.array(a_sample_ipt, copy=True)
        tmp[:, i] = 1 - tmp[:, i]  # inverse attribute
        if args.dataset == 'celeba':
            tmp = data.Celeba.check_attribute_conflict(tmp, args.atts[i], args.atts)
        b_sample_ipt_list.append(tmp)

    it_per_epoch = len(tr_data) // (args.batch_size * (args.n_d + 1))
    max_it = args.epoch * it_per_epoch
    for it in range(sess.run(it_cnt), max_it):
        with pylib.Timer(is_output=False) as t:
            sess.run(update_cnt)

            # which epoch
            epoch = it // it_per_epoch
            it_in_epoch = it % it_per_epoch + 1

            # learning rate
            lr_ipt = args.lr / (10 ** (epoch // args.init_epoch))

            # train D
            for i in range(args.n_d):
                d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(d_summary_opt, it)

            # train G
            g_summary_opt, _, _ = sess.run([g_summary, g_step, tower_assign], feed_dict={lr: lr_ipt})
            summary_writer.add_summary(g_summary_opt, it)

            # display
            if (it + 1) % 1 == 0:
                print("Epoch: (%3d) (%5d/%5d) Time: %s!" % (epoch, it_in_epoch, it_per_epoch, t))

            # save
            if (it + 1) % (args.save_freq if args.save_freq else it_per_epoch*5) == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
                print('Model is saved at %s!' % save_path)

            # sample
            if (it + 1) % (args.save_freq if args.save_freq else it_per_epoch) == 0:
                x_sample_opt_list = [xa_sample_ipt, np.full((args.n_sample, args.img_size, args.img_size // 10, 3), -1.0)]
                _a_sample_ipt = (a_sample_ipt * 2 - 1) * 0.5
                for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                    _b_sample_ipt = (b_sample_ipt * 2 - 1) * 0.5
                    x_sample_opt_list.append(sess.run(xg_rand_sample, feed_dict={xs_sample: xa_sample_ipt,
                                                                            lt_sample: _b_sample_ipt,
                                                                            ls_sample: _a_sample_ipt}))
                    last_images = x_sample_opt_list[-1]
                    if i > 0:  # add a mark (+/-) in the upper-left corner to identify add/remove an attribute
                        for nnn in range(last_images.shape[0]):
                            last_images[nnn, 2:5, 0:7, :] = 1.
                            if _b_sample_ipt[nnn, i - 1] > 0:
                                last_images[nnn, 0:7, 2:5, :] = 1.
                                last_images[nnn, 1:6, 3:4, :] = -1.
                            last_images[nnn, 3:4, 1:6, :] = -1.

                sample = np.concatenate(x_sample_opt_list, 2)
                save_dir = './output/%s/sample_training' % args.experiment_name
                pylib.mkdir(save_dir)
                im.imwrite(im.immerge(sample, args.n_sample, 1), '%s/Epoch_(%d)_(%dof%d).jpg' % \
                           (save_dir, epoch, it_in_epoch, it_per_epoch))

                # xa_ipt, xc_ipt, xac_ipt, xca_ipt = sess.run([xa, xc, xac, xca])
                # xa_ipt, xc_ipt, xac_ipt, xca_ipt = xa_ipt[:args.n_sample], xc_ipt[:args.n_sample], xac_ipt[:args.n_sample], xca_ipt[:args.n_sample]
                # im.imwrite(im.immerge(np.concatenate([xa_ipt, xc_ipt, xac_ipt, xca_ipt], 2), args.n_sample, 1),
                #            '%s/Epoch_(%d)_(%dof%d)_2img.jpg' % (save_dir, epoch, it_in_epoch, it_per_epoch))

except:
    traceback.print_exc()
finally:
    save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_in_epoch, it_per_epoch))
    print('Model is saved at %s!' % save_path)
    sess.close()

