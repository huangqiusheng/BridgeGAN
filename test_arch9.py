from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from functools import partial
import json
import traceback


import imlib as im
import numpy as np
import pylib
import tensorflow as tf
import tflib as tl

import data
import models_624 as models
# import models_1104 as models
import os

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', default='check', help='experiment_name')
parser.add_argument('--gpu', type=str, default='1', help='gpu')
parser.add_argument('--dataroot', type=str, default='./data/CelebA')
# if assigned, only given images will be tested.
parser.add_argument('--img', type=int, nargs='+', default=None, help='e.g., --img 182638 202599')
# for multiple attributes
parser.add_argument('--test_atts', nargs='+', default=None)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--val_or_test', type=str, default='test')
parser.add_argument('--interp', type=bool, default=False)
args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)
# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
n_downblks_gen = args['n_downblks_gen']
n_intermedblks_gen = args['n_intermedblks_gen']
ch_gen = args['ch_gen']
n_upblks_gen = args['n_upblks_gen']
n_layers_map = args['n_layers_map']
fc_dim_map = args['fc_dim_map']
n_downblks_enc = args['n_downblks_enc']
n_intermedblks_enc = args['n_intermedblks_enc']
ch_enc = args['ch_enc']
n_mlp_enc = args['n_mlp_enc']
n_resblks_dis = args['n_resblks_dis']
ch_dis = args['ch_dis']
dim_noise = args['dim_noise']
use_cropped_img = args['use_cropped_img']

print('Using selected images:', args_.img)
os.environ['CUDA_VISIBLE_DEVICES'] = args_.gpu

# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================
# data
sess = tl.session()
if args_.val_or_test == 'test':
    if args_.test_atts:
        te_data = data.Celeba(args_.dataroot, atts, img_size, args_.batch_size, part='test', sess=sess, crop=not use_cropped_img, im_no=args_.img, filter_att = [atts.index(a) for a in args_.test_atts], filter_pos=False, repeat=1)
        tar_datas = data.Celeba(args_.dataroot, atts, img_size, 5, part='test', sess=sess, crop=not use_cropped_img, filter_att = [atts.index(a) for a in args_.test_atts], filter_pos=True)
        # map(lambda x:bool(int(x)),list(bin(i).replace('0b','').zfill(3))) interp need 1
    else:
        te_data = data.Celeba(args_.dataroot, atts, img_size, args_.batch_size, part='test', sess=sess, crop=not use_cropped_img, im_no=args_.img)
        tar_datas = [data.Celeba(args_.dataroot, atts, img_size, args_.batch_size, part='test', sess=sess, crop=not use_cropped_img, filter_att = i, filter_pos=pos) for i in range(n_att) for pos in [False, True]]
elif args_.val_or_test == 'val':
    if args_.test_atts:
        te_data = data.Celeba(args_.dataroot, atts, img_size, args_.batch_size, part='val', sess=sess, crop=not use_cropped_img, im_no=args_.img, filter_att = [atts.index(a) for a in args_.test_atts], filter_pos=False,
                              drop_remainder = False, shuffle = False, repeat = 1)
        tar_datas = data.Celeba(args_.dataroot, atts, img_size, 5, part='val', sess=sess, crop=not use_cropped_img, filter_att = [atts.index(a) for a in args_.test_atts], filter_pos=True,
                                drop_remainder=False, shuffle=False, repeat=1)
        # map(lambda x:bool(int(x)),list(bin(i).replace('0b','').zfill(3)))
    else:
        te_data = data.Celeba(args_.dataroot, atts, img_size, args_.batch_size, part='val', sess=sess, crop=not use_cropped_img, im_no=args_.img,
                              drop_remainder=False, shuffle=False, repeat=1)
        tar_datas = [data.Celeba(args_.dataroot, atts, img_size, args_.batch_size, part='val', sess=sess, crop=not use_cropped_img, filter_att = i, filter_pos=pos, drop_remainder = False, shuffle = False, repeat = 1) for i in range(n_att) for pos in [False, True]]

# models
Generator = partial(models.Generator2_wRelu, n_downblks = n_downblks_gen, n_intermedblks = n_intermedblks_gen, n_upblks = n_upblks_gen, ch = ch_gen)
# Generator = partial(models.Generator2_wRelu_imation, n_downblks = n_downblks_gen, n_intermedblks = n_intermedblks_gen, n_upblks = n_upblks_gen, ch = ch_gen)
MappingNet = partial(models.MappingNet_multiStream_deconv_wIN_wRelu_concat, n_layers = n_layers_map, fc_dim = fc_dim_map)
EncoderR = partial(models.EncoderR_NoVAE, output_dim = dim_noise, fc_dim=64, n_mlp = n_mlp_enc, n_resblks = n_downblks_enc + n_intermedblks_enc, ch = ch_enc)
Encoder = partial(models.Encoder4_wIN_wRelu_concat,n_downblks = n_downblks_enc, n_intermedblks = n_intermedblks_enc, n_mlp = n_mlp_enc, ch = ch_enc)
# Encoder_REM = partial(models.Encoder4_wIN_wRelu_concat_REM,n_downblks = n_downblks_enc, n_intermedblks = n_intermedblks_enc, n_mlp = n_mlp_enc, ch = ch_enc)
# inputs
xs_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
xt_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
lt_sample = tf.placeholder(tf.float32, shape=[None, n_att])
ls_sample = tf.placeholder(tf.float32, shape=[None, n_att])
# sample
test_label = lt_sample - ls_sample
test_label = test_label * 1.5
r_sample = tf.random_normal((args_.batch_size, dim_noise), mean=0.0, stddev=1.0)
s_rand_t = MappingNet(r_sample, test_label)
s_targ_t_sample, _ = Encoder(xt_sample, test_label)

s_rand_s, _ = Encoder(xs_sample, -test_label)
s_rand = s_rand_t + s_rand_s
xg_rand_sample = Generator(xs_sample, s_targ = s_rand)
# s_rand_sample = MappingNet(r_sample, test_label)
# xg_rand_sample = Generator(xs_sample, s_targ = s_rand_sample)

# s_targ_s_sample, _ = Encoder(xs_sample, lt_sample - ls_sample)

s_targ_sample = s_targ_t_sample + s_rand_s
# s_targ_sample = SharedLayer(tf.concat([s_rand_s, s_targ_t_sample], axis = -1))
xg_targ_sample = Generator(xs_sample, s_targ = s_targ_sample)

xg_interp_sample = Generator(xs_sample, s_targ = [s_rand_t, s_targ_sample], alpha = 0.5)
# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
if args_.ckpt:
    ckpt_dir = './output/%s/checkpoints' % args_.experiment_name + '/' + args_.ckpt
else:
    ckpt_dir = './output/%s/checkpoints' % args_.experiment_name
tl.load_checkpoint(ckpt_dir , sess)
count_s = 0
# test
try:
    multi_atts = args_.test_atts is not None
    for idx, batch in enumerate(te_data):
        assert count_s < 19962
        count_s = count_s + 1
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        b_sample_ipt_list = [a_sample_ipt.copy()]
        if multi_atts:  # multi_atts
            for a in args_.test_atts:
                i = atts.index(a)
                b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]
                b_sample_ipt_list[-1] = data.Celeba.check_attribute_conflict(b_sample_ipt_list[-1], atts[i], atts)
        else:  # test_single_attributes
            for i in range(len(atts)):
                tmp = np.array(a_sample_ipt, copy=True)
                tmp[:, i] = 1 - tmp[:, i]  # inverse attribute
                tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
                b_sample_ipt_list.append(tmp)

        if multi_atts and not args_.interp:
            # for i in range(100):
            save_folder = 'sample_testing_' + '_'.join(args_.test_atts)
            save_dir = './output/%s/%s' % (args_.experiment_name, save_folder)
            pylib.mkdir(save_dir)
            target_samples, target_labels = tar_datas.get_next()
            tsample = np.concatenate([np.full((img_size, img_size,3), -1),np.concatenate(target_samples, 1)], 1)
            x_sample_opt_list = [xa_sample_ipt]
            for target_sample in target_samples:
                x_sample_opt_list.append(sess.run(xg_targ_sample, feed_dict={xs_sample: xa_sample_ipt,
                                                                         xt_sample: np.tile(np.expand_dims(target_sample, axis=0),
                                                                                    (args_.batch_size, 1, 1, 1)),
                                                                         lt_sample: b_sample_ipt_list[-1],
                                                                         ls_sample: a_sample_ipt}))

            sample = np.concatenate([tsample, np.concatenate(np.concatenate(x_sample_opt_list, 2), 0)], 0)
            print('%06d.png done!' % (idx + 182638 if args_.img is None else args_.img[idx]+i))
            im.imwrite(sample, '%s/%06d%s.png' % (save_dir, idx if args_.img is None else args_.img[idx]+i,
                                                     '_%s' % (str(args_.test_atts)) if multi_atts else ''))
                # sample = np.concatenate(target_samples, 0)
                # im.imwrite(sample, '%s/%06dreference.png' % (save_dir, idx if args_.img is None else args_.img[idx]))
        elif args_.interp and multi_atts:
            save_folder = 'sample_testing_interp_' + '_'.join(args_.test_atts)
            save_dir = './output/%s/%s' % (args_.experiment_name, save_folder)
            pylib.mkdir(save_dir)
            target_samples, target_labels = tar_datas.get_next()
            tsample = np.concatenate([np.full((img_size, img_size, 3), -1), np.full((img_size, img_size, 3), -1), np.full((img_size, img_size, 3), -1), np.concatenate(target_samples, 1)], 1)
            x_sample_opt_list = [xa_sample_ipt]
            for target_sample in target_samples:
                x_sample_opt_list.append(sess.run(xg_rand_sample, feed_dict={xs_sample: xa_sample_ipt,
                                                                             lt_sample: b_sample_ipt_list[-1],
                                                                             ls_sample: a_sample_ipt}))
                x_sample_opt_list.append(sess.run(xg_interp_sample, feed_dict={xs_sample: xa_sample_ipt,
                                                                               xt_sample: np.tile(
                                                                                   np.expand_dims(target_sample, axis=0),
                                                                                   (args_.batch_size, 1, 1, 1)),
                                                                             lt_sample: b_sample_ipt_list[-1],
                                                                             ls_sample: a_sample_ipt}))

                x_sample_opt_list.append(sess.run(xg_targ_sample, feed_dict={xs_sample: xa_sample_ipt,
                                                                             xt_sample: np.tile(
                                                                                 np.expand_dims(target_sample, axis=0),
                                                                                 (args_.batch_size, 1, 1, 1)),
                                                                             lt_sample: b_sample_ipt_list[-1],
                                                                             ls_sample: a_sample_ipt}))

            sample = np.concatenate([tsample, np.concatenate(np.concatenate(x_sample_opt_list, 2), 0)], 0)
            print('%06d.png done!' % (idx + 182638 if args_.img is None else args_.img[idx]))
            im.imwrite(sample, '%s/%06d%s.png' % (save_dir, idx if args_.img is None else args_.img[idx],
                                                  '_%s' % (str(args_.test_atts)) if multi_atts else ''))

        else:
            if args_.ckpt:
                ckpt_epoch = args_.ckpt.split('(')[1].split(')')[0]
                save_folder = 'sample_testing_' + args_.val_or_test + '_' + ckpt_epoch
            else:
                save_folder = 'sample_testing_' + args_.val_or_test
            save_dir = './output/%s/%s' % (args_.experiment_name, save_folder)
            pylib.mkdir(save_dir)
            print('Folder: %s' % save_dir)
            exist_count = 0
            for id_in_batch in range(args_.batch_size):
                if os.path.exists('%s/%06d%s_1.png' % (save_dir, idx * args_.batch_size + id_in_batch + 182638 if args_.img is None else args_.img[idx*args_.batch_size + id_in_batch],
                '_%s' % (str(args_.test_atts)) if multi_atts else '')):
                    exist_count += 1
            if exist_count == args_.batch_size:
                continue
            x_sample_opt_list = [xa_sample_ipt, np.full((args_.batch_size, img_size, img_size // 10, 3), -1.0)]
            for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                x_sample_opt_list.append(sess.run(xg_rand_sample, feed_dict={xs_sample: xa_sample_ipt,
                                                                        lt_sample: b_sample_ipt,
                                                                        ls_sample: a_sample_ipt}))
            sample = np.concatenate(x_sample_opt_list, 2)

            for id_in_batch in range(np.size(sample, 0)):
                if args_.val_or_test == 'test':
                    im.imwrite(sample[id_in_batch], '%s/%06d%s_1.png' % (save_dir,
                                                                       idx * args_.batch_size + id_in_batch + 182638 if args_.img is None else
                                                                       args_.img[idx*args_.batch_size + id_in_batch],
                                                                       '_%s' % (str(args_.test_atts)) if multi_atts else ''))
                    print('%06d.png done!' % (idx * args_.batch_size + id_in_batch + 182638 if args_.img is None else args_.img[idx*args_.batch_size + id_in_batch]))
                elif args_.val_or_test == 'val':
                    im.imwrite(sample[id_in_batch], '%s/%06d%s.png' % (save_dir,
                                                                       idx * args_.batch_size + id_in_batch + 182001 if args_.img is None else
                                                                       args_.img[idx],
                                                                       '_%s' % (str(args_.test_atts)) if multi_atts else ''))
                    print('%06d.png done!' % (idx * args_.batch_size + id_in_batch + 182001 if args_.img is None else args_.img[idx]))

            # im.imwrite(im.immerge(sample, args_.batch_size,1), '%s/%06d%s.png' % (save_dir,
            #                                                  idx + 182638 if args_.img is None else args_.img[idx],
            #                                                  '_%s' % (str(args_.test_atts)) if multi_atts else ''))
            #
            # print('%06d.png done!' % (idx + 182638 if args_.img is None else args_.img[idx]))

except:
    traceback.print_exc()
finally:
    sess.close()
