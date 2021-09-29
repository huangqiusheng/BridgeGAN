from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import random
from PIL import Image
import time
from functools import partial
import glob
import tflib as tl
def age_group_div_(age):

    #output = np.zeros([1,5],dtype=float)
    lens = age.shape[0]
    output = np.zeros([lens], dtype=float)
    for i in range(0,lens):
        if age[i] <= 50:
            output[i] = 0
        elif age[i] > 50:
            output[i] = 1
    return output
# def age_group_div_(age):
#
#     #output = np.zeros([1,5],dtype=float)
#     lens = age.shape[0]
#     output = np.zeros([lens], dtype=float)
#
#     for i in range(0,lens):
#         if age[i] <= 30:
#             output[i] = 0
#         elif age[i]>30 and age[i]<=40:
#             output[i] = 1
#         elif age[i]>40 and age[i]<=50:
#             output[i] = 2
#         elif age[i]>50:
#             output[i] = 3
#     #output = tf.convert_to_tensor(output, tf.float32)
#     return output
def session(graph=None, allow_soft_placement=True,
            log_device_placement=False, allow_growth=True):
    """Return a Session with simple config."""
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                            log_device_placement=log_device_placement)
    config.gpu_options.allow_growth = allow_growth
    return tf.Session(graph=graph, config=config)

def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, filter_fn = None):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if filter_fn:
        dataset = dataset.filter(filter_fn())

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset
def batch_dataset_MT(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, filter_fn = None):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if filter_fn:
        dataset = dataset.filter(filter_fn())

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset

def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, filter_fn = None):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3) #png
        return (img,) + label

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat, filter_fn)

    return dataset
def disk_image_batch_dataset_MT(img_paths, img_segs_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, filter_fn = None):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    if labels is None:
        dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    elif isinstance(labels, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, img_segs_paths, labels))

    def parse_func(path, path_segs, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3) #png
        img_sges = tf.read_file(path_segs)
        img_sges = tf.image.decode_png(img_sges, 1)  # png
        return (img,) + (img_sges,) + label

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset_MT(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat, filter_fn)

    return dataset

def tfrecord_batch_dataset(tfrecord_path, batch_size, label_len, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, reshape_size = 170, filter_fn = None):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # def parse_func(path, *label):
    #     img = tf.read_file(path)
    #     img = tf.image.decode_png(img, 3)
    #     return (img,) + label

    def parse_func(serialized_example):
        random_flip = True
        features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([label_len], tf.int64),
                                                                        'noise': tf.FixedLenFeature([64], tf.float32),
                                                                        'img_raw' : tf.FixedLenFeature([], tf.string),
                                                                        'img_name': tf.FixedLenFeature([], tf.string)})
        label = tf.cast(features['label'], tf.int32)
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        noise = tf.cast(features['noise'], tf.float32)
        #img_name = tf.decode_raw(features['img_name'], tf.uint8)
        img = tf.reshape(img, [reshape_size, reshape_size, 3])
        if random_flip:
            img = tf.image.random_flip_left_right(img)
        return img, label, noise#, img_name

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat, filter_fn)

    return dataset
def tfrecord_batch_dataset_MT(tfrecord_path, batch_size, label_len, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1, reshape_size = 170, filter_fn = None):
    """Disk image batch dataset.

    This function is suitable for jpg and png files

    img_paths: string list or 1-D tensor, each of which is an iamge path
    labels: label list/tuple_of_list or tensor/tuple_of_tensor, each of which is a corresponding label
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    # def parse_func(path, *label):
    #     img = tf.read_file(path)
    #     img = tf.image.decode_png(img, 3)
    #     return (img,) + label

    def parse_func(serialized_example):
        random_flip = False
        features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([label_len], tf.int64),
                                                                        'img_raw' : tf.FixedLenFeature([], tf.string),
                                                                        'img_name': tf.FixedLenFeature([], tf.string),
                                                                        'img_segs' : tf.FixedLenFeature([], tf.string)})
        label = tf.cast(features['label'], tf.int32)
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        segs = tf.decode_raw(features['img_segs'], tf.uint8)
        #img_name = tf.decode_raw(features['img_name'], tf.uint8)
        img = tf.reshape(img, [reshape_size, reshape_size, 3])
        segs = tf.reshape(segs, [321, 321, 1])
        if random_flip:
            img = tf.image.random_flip_left_right(img)
            segs = tf.image.random_flip_left_right(segs)
        return img, segs, label#, img_name

    if map_func:
        def map_func_(*args):
            return map_func(*parse_func(*args))
    else:
        map_func_ = parse_func

    # dataset = dataset.map(parse_func, num_parallel_calls=num_threads) is slower

    dataset = batch_dataset_MT(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat, filter_fn)

    return dataset


class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        self._iterator = dataset.make_initializable_iterator()
        self._batch_op = self._iterator.get_next()
        if sess:
            self._sess = sess
        else:
            self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op


class Celeba(Dataset):
    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    def __init__(self, data_dir, atts, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=2048, repeat=-1, sess=None, part='train', crop=True,
                 im_no=None, is_tfrecord = False, filter_att = None, filter_pos=None):
        super(Celeba, self).__init__()

        def _map_func(img, label, noise):
            if crop and not is_tfrecord:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            if is_tfrecord:
                label = tf.gather(label, att_id)  ##### 2020.02.07
                #noise = tf.gather(noise, att_id)
            label = (label + 1) // 2
            return img, label, noise
        def map_func(img, label):
            if crop and not is_tfrecord:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            if is_tfrecord:
               label = tf.gather(label, att_id)  ##### 2020.02.07
                #noise = tf.gather(noise, att_id)
            label = (label + 1) // 2
            return img, label

        if filter_att is not None and filter_pos is not None:
            def filter_fn(feat_id, pos, image, feature_val):
                if pos:
                    return tf.reduce_all(tf.equal(tf.gather(feature_val,feat_id), tf.ones_like(tf.gather(feature_val,feat_id))))
                else:
                    return tf.reduce_all(tf.equal(tf.gather(feature_val,feat_id), tf.zeros_like(tf.gather(feature_val,feat_id))))

            def get_filter_fn():
                return partial(filter_fn, filter_att, filter_pos)
        else:
            get_filter_fn = None

        list_file = os.path.join(data_dir, 'list_attr_celeba.txt')
        if is_tfrecord:
            img_size = 170
            att_id = [Celeba.att_dict[att] for att in atts]#atts[filter_att]
            if filter_att is not None and filter_pos is not None and os.path.exists(
                    os.path.join(data_dir, 'celeba_tfrecords', part +"_"+ atts[filter_att] +"_"+ str(filter_pos) + "_crop.tfrecords")):
                tfrecord_path = os.path.join(data_dir, 'celeba_tfrecords', part +"_"+ atts[filter_att] +"_"+ str(filter_pos) + "_crop.tfrecords")
                dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                                 batch_size=batch_size,
                                                 label_len = 40,
                                                 prefetch_batch=prefetch_batch,
                                                 drop_remainder=drop_remainder,
                                                 map_func=_map_func,
                                                 num_threads=num_threads,
                                                 shuffle=shuffle,
                                                 buffer_size=buffer_size,
                                                 repeat=repeat,
                                                 reshape_size=img_size)
            else:
                tfrecord_path = os.path.join(data_dir, 'celeba_tfrecords', part + "_crop.tfrecords")

                dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                                 batch_size=batch_size,
                                                 label_len = 40,
                                                 prefetch_batch=prefetch_batch,
                                                 drop_remainder=drop_remainder,
                                                 map_func=_map_func,
                                                 num_threads=num_threads,
                                                 shuffle=shuffle,
                                                 buffer_size=buffer_size,
                                                 repeat=repeat,
                                                 reshape_size=img_size,
                                                 filter_fn = get_filter_fn)
            if part == 'train':
                self._img_num = 182000
        else:
            if crop:
                img_dir_jpg = os.path.join(data_dir, 'img_align_celeba')
                img_dir_png = os.path.join(data_dir, 'img_align_celeba_png')
            else:
                img_dir_jpg = os.path.join(data_dir, 'img_crop_celeba')
                img_dir_png = os.path.join(data_dir, 'img_crop_celeba_png')

            names = np.loadtxt(list_file, skiprows=1, usecols=[0], dtype=np.str)
            if os.path.exists(img_dir_png):
                img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
            elif os.path.exists(img_dir_jpg):
                img_paths = [os.path.join(img_dir_jpg, name) for name in names]

            att_id = [Celeba.att_dict[att] + 1 for att in atts]
            labels = np.loadtxt(list_file, skiprows=1, usecols=att_id, dtype=np.int64)
            if len(labels.shape) == 1:
                labels = labels[:, np.newaxis]
            # if img_resize == 64:
            #     # crop as how VAE/GAN do
            #     offset_h = 40
            #     offset_w = 15
            #     img_size = 148
            # else:
            #     offset_h = 26
            #     offset_w = 3
            #     img_size = 170

            offset_h = 26
            offset_w = 3
            img_size = 170

            if im_no is not None:
                drop_remainder = False
                shuffle = False
                #repeat = 1
                img_paths = [img_paths[i - 1] for i in im_no]
                labels = labels[[i - 1 for i in im_no]]
            elif part == 'test':
                drop_remainder = False
                shuffle = False
                #repeat = 1
                img_paths = img_paths[182637:]
                labels = labels[182637:]
                # img_paths = img_paths[182637:182637+5000]
                # labels = labels[182637:182637+5000]
                # img_paths = img_paths[0:64]   ###temp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # labels = labels[0:64]
            elif part == 'val':
                img_paths = img_paths[182000:182637]
                labels = labels[182000:182637]
            else:
                img_paths = img_paths[:182000]
                labels = labels[:182000]

            dataset = disk_image_batch_dataset(img_paths=img_paths,
                                               labels=labels,
                                               batch_size=batch_size,
                                               prefetch_batch=prefetch_batch,
                                               drop_remainder=drop_remainder,
                                               map_func=map_func,
                                               num_threads=num_threads,
                                               shuffle=shuffle,
                                               buffer_size=buffer_size,
                                               repeat=repeat,
                                               filter_fn = get_filter_fn)
            self._img_num = len(img_paths)
            self.data_dir = data_dir
            self.atts = atts
            self.part = part
            self.img_paths = img_paths
            self.labels = labels
            self.offset_h = offset_h
            self.offset_w = offset_w
            self.img_size = img_size
        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num

    @staticmethod
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)

        for att in att_batch:
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
                _set(att, 0, 'Bangs')
            elif att_name == 'Bangs' and att[att_id] == 1:
                _set(att, 0, 'Bald')
                _set(att, 0, 'Receding_Hairline')
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
                # _set(att, 0, 'bald')
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
        # Removed since `Mustache` and `No_Beard` are not conflict.
        # But the two attributes are not well labeled in the dataset.
        #            elif att_name in ['Mustache', 'No_Beard'] and att[att_id] == 1:
        #                for n in ['Mustache', 'No_Beard']:
        #                    if n != att_name:
        #                        _set(att, 0, n)

        return att_batch

    @staticmethod
    def check_random_attribute_conflict(att_batch, att_names, hair_color=None):
        """ For randomly generated attributes, tested but not used in this repo. """

        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        def _idx(att_name):
            if att_name in att_names:
                return att_names.index(att_name)
            return None

        for att in att_batch:
            valid_atts = [i for i in ['Receding_Hairline', 'Bald'] if i in att_names]
            if 'Bangs' in att_names and att[_idx('Bangs')] == 1 \
                    and len(valid_atts) > 0 and sum([att[_idx(i)] for i in valid_atts]) > 0:
                _set(att, 0, 'Bangs') if random.random() < 0.5 else [_set(att, 0, i) for i in valid_atts]
            #            hair_color = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
            if hair_color is not None and sum([att[_idx(i)] for i in hair_color]) > 1:
                one = random.randint(0, len(hair_color))
                for i in range(len(hair_color)):
                    _set(att, 1 if i == one else 0, hair_color[i])
            if 'Straight_Hair' in att_names and 'Wavy_Hair' in att_names and att[_idx('Straight_Hair')] == 1 and att[
                _idx('Wavy_Hair')] == 1:
                _set(att, 0, 'Straight_Hair') if random.random() < 0.5 else _set(att, 0, 'Wavy_Hair')
        #            if 'Mustache' in att_names and 'No_Beard' in att_names and att[_idx('Mustache')] == 1 and att[_idx('No_Beard')] == 1:
        #                _set(att, 0, 'Mustache') if random.random() < 0.5 else _set(att, 0, 'No_Beard')
        return att_batch

    def creat_tfrecord(self):
        if not os.path.exists(os.path.join(self.data_dir, "celeba_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "celeba_tfrecords"))
        count = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "celeba_tfrecords", self.part + "_crop.tfrecords"))
        for i, img_path in enumerate(self.img_paths):
            img = Image.open(os.path.join(img_path))
            img = img.crop((self.offset_w, self.offset_h, self.offset_w + self.img_size, self.offset_h + self.img_size))
            img_raw = img.tobytes()
            label = self.labels[i, :]
            # sess = tl.session()
            # noise = sess.run(tf.random_normal((1, 64))).astype(np.float)
            # sess.close()
            noise = np.random.normal(0, 1, 64)
            img_name = os.path.split(img_path)[1]
            img_name_byte = img_name.encode(encoding='utf-8')
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                         'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name_byte])),
                         'noise': tf.train.Feature(float_list=tf.train.FloatList(value=noise))#
                         }))
            writer.write(example.SerializeToString())
            count = count + 1
            if count % 500 == 0:
                print('Time:{0},{1} images are processed.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count))
        print("%d images are processed." % count)
        print(self.part + ' done!')
        writer.close()

    def creat_tfrecord_split_opposite_label(self):
        if not os.path.exists(os.path.join(self.data_dir, "celeba_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "celeba_tfrecords"))
        for attribute_id in range(len(atts)):
            att = self.atts[attribute_id]
            count_true = 0
            count_false = 0
            if not os.path.exists(os.path.join(self.data_dir, "celeba_tfrecords", self.part + "_"+ str(attribute_id) +
                                                                   "_True_crop.tfrecords")):
                true_writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "celeba_tfrecords", self.part + "_"+ str(attribute_id) +
                                                                       "_True_crop.tfrecords"))
                false_writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "celeba_tfrecords", self.part + "_" + str(attribute_id) +
                                 "_False_crop.tfrecords"))
                for i, img_path in enumerate(self.img_paths):
                    img = Image.open(os.path.join(img_path))
                    img = img.crop((self.offset_w, self.offset_h, self.offset_w + self.img_size, self.offset_h + self.img_size))
                    img_raw = img.tobytes()
                    label = self.labels[i, :]
                    img_name = os.path.split(img_path)[1]
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                                 'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name]))}))
                    if label[attribute_id] == 1:
                        true_writer.write(example.SerializeToString())
                        count_true = count_true + 1
                        if count_true % 500 == 0:
                            print('Time:{0},{1} positive images are processed.'.format(
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count_true))
                    elif label[attribute_id] == -1:
                        false_writer.write(example.SerializeToString())
                        count_false = count_false + 1
                        if count_false % 500 == 0:
                            print('Time:{0},{1} negative images are processed.'.format(
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count_false))
                print("%d positive images are processed." % count_true)
                print("%d negative images are processed." % count_false)
                print(att + ' done!')
                true_writer.close()
                false_writer.close()

    def creat_tfrecord_split_true_label(self, selected_atts):
        if not os.path.exists(os.path.join(self.data_dir, "celeba_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "celeba_tfrecords"))
        for attribute_id in range(len(selected_atts)):
            att = selected_atts[attribute_id]
            att_id = self.atts.index(att)
            count_true = 0
            if not os.path.exists(os.path.join(self.data_dir, "celeba_tfrecords", self.part + "_"+ att +
                                                                   "_True_crop.tfrecords")):
                true_writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "celeba_tfrecords", self.part + "_"+ att +
                                                                       "_True_crop.tfrecords"))
                for i, img_path in enumerate(self.img_paths):
                    img = Image.open(os.path.join(img_path))
                    img = img.crop((self.offset_w, self.offset_h, self.offset_w + self.img_size, self.offset_h + self.img_size))
                    img_raw = img.tobytes()
                    label = self.labels[i, :]
                    img_name = os.path.split(img_path)[1]
                    img_name_byte = img_name.encode(encoding='utf-8')
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                                 'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name_byte]))}))
                    if label[att_id] == 1:
                        true_writer.write(example.SerializeToString())
                        count_true = count_true + 1
                        if count_true % 500 == 0:
                            print('Time:{0},{1} positive images are processed.'.format(
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count_true))
                print("%d positive images are processed." % count_true)
                print(att + ' done!')
                true_writer.close()

class x2y(Dataset):
    def __init__(self, dataset_name, data_dir, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=2048, repeat=-1, sess=None, part='train',
                 im_no=None, is_tfrecord = False):
        super(x2y, self).__init__()

        def _map_func(img, label):
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        if is_tfrecord:
            img_size = 256
            tfrecord_path = os.path.join(data_dir, dataset_name + '_tfrecords', part + ".tfrecords")

            dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                             batch_size=batch_size,
                                             label_len = 1,
                                             prefetch_batch=prefetch_batch,
                                             drop_remainder=drop_remainder,
                                             map_func=_map_func,
                                             num_threads=num_threads,
                                             shuffle=shuffle,
                                             buffer_size=buffer_size,
                                             repeat=repeat,
                                             reshape_size=img_size)
            if part == 'train':
                if dataset_name == 'apple2orange':
                    self._img_num = 2014 - 100
                elif dataset_name == 'summer2winter':
                    self._img_num = 2193 -100
                elif dataset_name == 'horse2zebra':
                    self._img_num = 2401 - 100
        else:
            img_dir_jpg = os.path.join(data_dir, dataset_name)

            img_paths = []
            labels = []
            if part == 'test':
                drop_remainder = False
                shuffle = False
                repeat = 1
                for i, partAB in enumerate(['testA', 'testB']):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.jpg'))
                    img_paths.extend(imgAB_paths)
                    labels.extend([ [2 * (1-i) -1] for _ in range(len(imgAB_paths))])
            else:
                for i, partAB in enumerate(['trainA', 'trainB']):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.jpg'))
                    if part == 'val':
                        img_paths.extend(imgAB_paths[:50])
                        labels.extend([[2 * (1-i) -1] for _ in range(len(imgAB_paths[:50]))])
                    elif part == 'train':
                        img_paths.extend(imgAB_paths[50:])
                        labels.extend([[2 * (1-i) -1] for _ in range(len(imgAB_paths[50:]))])
            labels = np.array(labels)

            dataset = disk_image_batch_dataset(img_paths=img_paths,
                                               labels=labels,
                                               batch_size=batch_size,
                                               prefetch_batch=prefetch_batch,
                                               drop_remainder=drop_remainder,
                                               map_func=_map_func,
                                               num_threads=num_threads,
                                               shuffle=shuffle,
                                               buffer_size=buffer_size,
                                               repeat=repeat)
            self._img_num = len(img_paths)
            self.data_dir = data_dir
            self.part = part
            self.img_paths = img_paths
            self.labels = labels
            self.dataset_name = dataset_name
        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num

    def creat_tfrecord(self):
        if not os.path.exists(os.path.join(self.data_dir, self.dataset_name + "_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, self.dataset_name + "_tfrecords"))
        count = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, self.dataset_name + "_tfrecords", self.part + ".tfrecords"))
        for i, img_path in enumerate(self.img_paths):
            img = Image.open(os.path.join(img_path))
            img = img.convert("RGB")
            img_raw = img.tobytes()
            label = self.labels[i, :]
            img_name = os.path.split(img_path)[1]
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                         'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name]))}))
            writer.write(example.SerializeToString())
            count = count + 1
            if count % 500 == 0:
                print('Time:{0},{1} images are processed.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count))
        print("%d images are processed." % count)
        print(self.part + ' done!')
        writer.close()
class MakeupTransfer(Dataset):
    def __init__(self, data_dir, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=2048, repeat=-1, sess=None, part='train',
                 im_no=None, is_tfrecord = False, filter_att = None, filter_pos=None, is_augment=False):
        super(MakeupTransfer, self).__init__()

        def _map_func(img, segs, label):
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            if is_augment:
                # 将图片随机进行垂直翻转
                img = tf.image.random_flip_left_right(img)
                segs = tf.image.random_flip_left_right(segs)
                # # 随机设置图片的亮度
                # img = tf.image.random_brightness(img, max_delta=0.6)
                # # 随机设置图片的对比度
                # img = tf.image.random_contrast(img, lower=0.1, upper=0.8)
                # # 随机设置图片的色度
                # img = tf.image.random_hue(img, max_delta=0.3)
                # # 随机设置图片的饱和度
                # img = tf.image.random_saturation(img, lower=0.2, upper=1.8)
            segs = tf.cast(segs, tf.float32)
            # segs = tf.reshape(segs, [batch_size, 361, 361, 1])
            #segs = tf.concat([segs, segs, segs], 2)
            segs = tf.image.resize_images(segs, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            #segs = tf.split(segs, 3, 3)
            # segs = tf.clip_by_value(segs, 0, 255) / 127.5 - 1
            segs = tf.clip_by_value(segs, 0, 255)
            segs = segs/(0.5*tf.reduce_max(segs)) - 1
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            # label = tf.to_int32(label, name='ToInt32')
            # label = tf.one_hot(label, 2, 1, 0)
            # label = tf.to_float(label, name='ToFloat')
            # label = tf.squeeze(label)
            label = (label + 1) // 2
            label = tf.to_float(label, name='ToFloat')
            return img, segs, label
        if filter_att is not None and filter_pos is not None:
            def filter_fn(feat_id, pos, image, _, feature_val):
                if pos:
                    return tf.reduce_all(tf.equal(tf.gather(feature_val,feat_id), tf.ones_like(tf.gather(feature_val,feat_id))))
                else:
                    return tf.reduce_all(tf.equal(tf.gather(feature_val,feat_id), tf.zeros_like(tf.gather(feature_val,feat_id))))

            def get_filter_fn():
                return partial(filter_fn, filter_att, filter_pos)
        else:
            get_filter_fn = None

        if is_tfrecord:
            img_size = 361
            tfrecord_path = os.path.join(data_dir, 'MakeupTransfer' + '_tfrecords', part + ".tfrecords")

            dataset = tfrecord_batch_dataset_MT(tfrecord_path=tfrecord_path,
                                             batch_size=batch_size,
                                             label_len = 1,
                                             prefetch_batch=prefetch_batch,
                                             drop_remainder=drop_remainder,
                                             map_func=_map_func,
                                             num_threads=num_threads,
                                             shuffle=shuffle,
                                             buffer_size=buffer_size,
                                             repeat=repeat,
                                             reshape_size=img_size,
                                             filter_fn = get_filter_fn)#

        else:
            img_dir_jpg = os.path.join(data_dir, 'MakeupTransfer')

            img_paths = []
            labels = []
            img_seg_paths = []
            img_segs_paths = []
            if part == 'test':
                drop_remainder = False
                shuffle = False
                repeat = 1
                for i, partAB in enumerate(['testA', 'testB']):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.png'))
                    img_paths.extend(imgAB_paths)
                    labels.extend([ [(1-i)] for _ in range(len(imgAB_paths))])
            else:
                for i, partAB in enumerate(['trainA', 'trainB']):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.png'))
                    if partAB == 'trainA':
                        for j1 in range(len(imgAB_paths)):
                            img_seg_paths.append(imgAB_paths[j1][:22]+'segs/'+imgAB_paths[j1][22:])
                    else:
                        img_seg_paths = []
                        for j2 in range(len(imgAB_paths)):
                            img_seg_paths.append(imgAB_paths[j2][:22]+'segs/'+imgAB_paths[j2][22:])
                    if part == 'val':
                        img_paths.extend(imgAB_paths[:50])
                        img_segs_paths.extend(img_seg_paths[:50])
                        labels.extend([[(1-i)] for _ in range(len(imgAB_paths[:50]))])
                    elif part == 'train':
                        img_paths.extend(imgAB_paths[50:])
                        img_segs_paths.extend(img_seg_paths[50:])
                        labels.extend([[(1-i)] for _ in range(len(imgAB_paths[50:]))])
            labels = np.array(labels)

            dataset = disk_image_batch_dataset_MT(img_paths=img_paths,
                                                  img_segs_paths=img_segs_paths,
                                               labels=labels,
                                               batch_size=batch_size,
                                               prefetch_batch=prefetch_batch,
                                               drop_remainder=drop_remainder,
                                               map_func=_map_func,
                                               num_threads=num_threads,
                                               shuffle=shuffle,
                                               buffer_size=buffer_size,
                                               repeat=repeat,
                                               filter_fn = get_filter_fn)
            self._img_num = len(img_paths)
            self.data_dir = data_dir
            self.part = part
            self.img_paths = img_paths
            self.img_segs_paths = img_segs_paths
            self.labels = labels
            self.dataset_name = 'MakeupTransfer'
        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num

    def creat_tfrecord(self):
        if not os.path.exists(os.path.join(self.data_dir, "MakeupTransfer_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "MakeupTransfer_tfrecords"))
        count = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "MakeupTransfer_tfrecords", self.part + ".tfrecords"))
        for i, img_path in enumerate(self.img_paths):
            img = Image.open(os.path.join(img_path))
            img = img.convert("RGB")
            img_raw = img.tobytes()
            img_s = Image.open(os.path.join(img_path[:22]+'segs/'+img_path[22:]))
            img_segs = img_s.tobytes()
            label = self.labels[i, :]
            img_name = os.path.split(img_path)[1]
            img_name_byte = img_name.encode(encoding='utf-8')
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                         'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name_byte])),
                         'img_segs': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_segs]))}))
            writer.write(example.SerializeToString())
            count = count + 1
            if count % 500 == 0:
                print('Time:{0},{1} images are processed.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count))
        print("%d images are processed." % count)
        print(self.part + ' done!')
        writer.close()
class AFHQ(Dataset):
    def __init__(self, data_dir, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=2048, repeat=-1, sess=None, part='train',
                 im_no=None, is_tfrecord = False, filter_att = None, filter_pos=None, is_augment=False):
        super(AFHQ, self).__init__()

        def _map_func(img, label):
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            if is_augment:
                # 将图片随机进行垂直翻转
                img = tf.image.random_flip_left_right(img)
                # # 随机设置图片的亮度
                # img = tf.image.random_brightness(img, max_delta=0.6)
                # # 随机设置图片的对比度
                # img = tf.image.random_contrast(img, lower=0.1, upper=0.8)
                # # 随机设置图片的色度
                # img = tf.image.random_hue(img, max_delta=0.3)
                # # 随机设置图片的饱和度
                # img = tf.image.random_saturation(img, lower=0.2, upper=1.8)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = tf.to_int32(label, name='ToInt32')
            label = tf.one_hot(label, 3, 1, 0)
            label = tf.to_float(label, name='ToFloat')
            label = tf.squeeze(label)
            # label = (label + 1) // 2
            return img, label
        if filter_att is not None and filter_pos is not None:  # the function of filter
            def filter_fn(feat_id, pos, image, feature_val):
                if pos:
                    return tf.equal(feature_val[feat_id], tf.ones_like(feature_val[feat_id]))
                else:
                    a = tf.equal(feature_val[feat_id], tf.zeros_like(feature_val[feat_id]))
                    return tf.equal(feature_val[feat_id], tf.zeros_like(feature_val[feat_id]))

            def get_filter_fn():
                return partial(filter_fn, filter_att, filter_pos)
        else:
            get_filter_fn = None


        if is_tfrecord:
            img_size = 512
            tfrecord_path = os.path.join(data_dir, 'AFHQ' + '_tfrecords', part + ".tfrecords")

            dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                             batch_size=batch_size,
                                             label_len = 1,
                                             prefetch_batch=prefetch_batch,
                                             drop_remainder=drop_remainder,
                                             map_func=_map_func,
                                             num_threads=num_threads,
                                             shuffle=shuffle,
                                             buffer_size=buffer_size,
                                             repeat=repeat,
                                             reshape_size=img_size,
                                             filter_fn = get_filter_fn)#

        else:
            img_dir_jpg = os.path.join(data_dir)# , 'AFHQ'

            img_paths = []
            labels = []
            if part == 'test':
                drop_remainder = False
                shuffle = False
                repeat = 1
                for i, partAB in enumerate(['testA', 'testB']):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.jpg'))
                    img_paths.extend(imgAB_paths)
                    labels.extend([ [(1-i)] for _ in range(len(imgAB_paths))])
            else:
                if part == 'val':
                    for i, partAB in enumerate(['val_cat', 'val_dog', 'val_wild']): # , 'val_wild'
                        imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.jpg'))
                        img_paths.extend(imgAB_paths)
                        labels.extend([[i] for _ in range(len(imgAB_paths))])
                elif part == 'train_cat':
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, part, '*.jpg'))
                    img_paths.extend(imgAB_paths)
                    labels.extend([[0] for _ in range(len(imgAB_paths))])
                elif part == 'train_dog':
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, part, '*.jpg'))
                    img_paths.extend(imgAB_paths)
                    labels.extend([[1] for _ in range(len(imgAB_paths))])
                elif part == 'train_wild':
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, part, '*.jpg'))
                    img_paths.extend(imgAB_paths)
                    labels.extend([[2] for _ in range(len(imgAB_paths))])

            labels = np.array(labels)

            dataset = disk_image_batch_dataset(img_paths=img_paths,
                                               labels=labels,
                                               batch_size=batch_size,
                                               prefetch_batch=prefetch_batch,
                                               drop_remainder=drop_remainder,
                                               map_func=_map_func,
                                               num_threads=num_threads,
                                               shuffle=shuffle,
                                               buffer_size=buffer_size,
                                               repeat=repeat,
                                               filter_fn = get_filter_fn)
            self._img_num = len(img_paths)
            self.data_dir = data_dir
            self.part = part
            self.img_paths = img_paths
            self.labels = labels
            self.dataset_name = 'AFHQ'
        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num

    def creat_tfrecord(self):
        if not os.path.exists(os.path.join(self.data_dir, "AFHQ_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "AFHQ_tfrecords"))
        count = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "AFHQ_tfrecords", self.part + ".tfrecords"))
        for i, img_path in enumerate(self.img_paths):
            img = Image.open(os.path.join(img_path))
            img = img.convert("RGB")
            img_raw = img.tobytes()
            label = self.labels[i, :]
            img_name = os.path.split(img_path)[1]
            img_name_byte = img_name.encode(encoding='utf-8')
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                         'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name_byte]))}))
            writer.write(example.SerializeToString())
            count = count + 1
            if count % 500 == 0:
                print('Time:{0},{1} images are processed.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count))
        print("%d images are processed." % count)
        print(self.part + ' done!')
        writer.close()

# ==========================================================
#                       Morph Dataset
# ==========================================================
class Morph(Dataset):
    att_dict = {'age': 0, 'gender': 1}

    def __init__(self, data_dir, atts, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=2048, repeat=-1, sess=None, part='train', crop=True,
                 im_no=None, is_tfrecord=False, filter_att=None, filter_pos=None, is_augment=False):
        super(Morph, self).__init__()

        def _map_func(img, label):   #one image
            n_cls = 2
            if crop and not is_tfrecord:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.cast(img, tf.float32)
            if is_augment:
                # 将图片随机进行垂直翻转
                img = tf.image.random_flip_left_right(img)
                # 随机设置图片的亮度
                img = tf.image.random_brightness(img, max_delta=0.6)
                # 随机设置图片的对比度
                img = tf.image.random_contrast(img, lower=0.1, upper=0.8)
                # 随机设置图片的色度
                img = tf.image.random_hue(img, max_delta=0.3)
                # 随机设置图片的饱和度
                img = tf.image.random_saturation(img, lower=0.2, upper=1.8)


            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1                                            #all of them are 255?
            #                                #change!
            # label = tf.to_int32(label, name='ToInt32') #
            # label = tf.one_hot(label, n_cls, 1, 0)
            # label = tf.to_float(label, name='ToFloat')
            # label = tf.squeeze(label)
            # label = tf.to_float(label, name='ToFloat')

            label = tf.to_int32(label, name='ToInt32')
            age = label[0]
            gender = label[1]
            # age = tf.one_hot(age, n_cls, 1, 0)

            age = tf.to_float(age, name='ToFloat')
            age = tf.squeeze(age)
            gender = tf.one_hot(gender, 2, 1, 0)
            gender = tf.to_float(gender, name='ToFloat')
            gender = tf.squeeze(gender)


            return img, gender, age

        if filter_att is not None and filter_pos is not None:  # the function of filter
            def filter_fn(feat_id, pos, image, feature_val, _):
                if pos:
                    return tf.equal(feature_val[feat_id], tf.ones_like(feature_val[feat_id]))
                else:
                    a = tf.equal(feature_val[feat_id], tf.zeros_like(feature_val[feat_id]))
                    return tf.equal(feature_val[feat_id], tf.zeros_like(feature_val[feat_id]))

            def get_filter_fn():
                return partial(filter_fn, filter_att, filter_pos)
        else:
            get_filter_fn = None

        list_file = os.path.join(data_dir, 'Morph_data_shuffle_10.txt')                                        #txt for what
        #read data(images and labels) by tfrecord (or not)
        if is_tfrecord:
            img_size = 300         #170
            if filter_att is not None and filter_pos is not None and os.path.exists(
                    os.path.join(data_dir, 'morph_tfrecords', part +"_"+ str(filter_att) +"_"+ str(filter_pos) + "_crop.tfrecords")):
                tfrecord_path = os.path.join(data_dir, 'morph_tfrecords', part +"_"+ str(filter_att) +"_"+ str(filter_pos) + "_crop.tfrecords")
                dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                                 batch_size=batch_size,
                                                 label_len=len(atts),
                                                 prefetch_batch=prefetch_batch,
                                                 drop_remainder=drop_remainder,
                                                 map_func=_map_func,
                                                 num_threads=num_threads,
                                                 shuffle=shuffle,
                                                 buffer_size=buffer_size,
                                                 repeat=repeat,
                                                 reshape_size=img_size)
            else:
                tfrecord_path = os.path.join(data_dir, 'morph_tfrecords', part + "_crop.tfrecords")

                dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                                 batch_size=batch_size,
                                                 label_len = len(atts),
                                                 prefetch_batch=prefetch_batch,
                                                 drop_remainder=drop_remainder,
                                                 map_func=_map_func,
                                                 num_threads=num_threads,
                                                 shuffle=shuffle,
                                                 buffer_size=buffer_size,
                                                 repeat=repeat,
                                                 reshape_size=img_size,
                                                 filter_fn=get_filter_fn)
            #the numbers of data for train
            if part == 'train':
                self._img_num = 46300
        else:

            img_dir_jpg = os.path.join(data_dir, 'Morph')                        #  the name is named by me ?
            img_dir_png = os.path.join(data_dir, 'Morph_png')


            names = np.loadtxt(list_file, skiprows=1, usecols=[0], dtype=np.str)
            if os.path.exists(img_dir_png):
                img_paths = [os.path.join(img_dir_png, name.replace('jpg', 'png')) for name in names]
            elif os.path.exists(img_dir_jpg):
                img_paths = [os.path.join(img_dir_jpg, name) for name in names]

            att_id = [Morph.att_dict[att] + 1 for att in atts]
            labels = np.loadtxt(list_file, skiprows=1, usecols=att_id, dtype=np.int64)#[:, np.newaxis]
            # if img_resize == 64:
            #     # crop as how VAE/GAN do
            #     offset_h = 40
            #     offset_w = 15
            #     img_size = 148
            # else:
            #     offset_h = 26
            #     offset_w = 3
            #     img_size = 170

            img_size = 300  # 170
            offset_h = 0  #26 #168(400)
            offset_w = 0   #3


            if im_no is not None:
                drop_remainder = False
                shuffle = False
                repeat = 1
                img_paths = [img_paths[i - 1] for i in im_no]
                labels = labels[[i - 1 for i in im_no]]
            elif part == 'test':
                drop_remainder = False
                shuffle = False
                repeat = 1
                img_paths = img_paths[46500:]#147500
                labels = labels[46500:]
                # img_paths = img_paths[182637:182637+5000]
                # labels = labels[182637:182637+5000]
                # img_paths = img_paths[0:64]                                  ###temp!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # labels = labels[0:64]
            elif part == 'val':
                img_paths = img_paths[46308:46500] #46397
                labels = labels[46308:46500]
            else:
                img_paths = img_paths[:46300] # 147101
                labels = labels[:46300]  #46300

            labels[:,0] = age_group_div_(labels[:,0])  #VGG16 train need labels = age_group_div_(labels)
            dataset = disk_image_batch_dataset(img_paths=img_paths,
                                               labels=labels,
                                               batch_size=batch_size,
                                               prefetch_batch=prefetch_batch,
                                               drop_remainder=drop_remainder,
                                               map_func=_map_func,
                                               num_threads=num_threads,
                                               shuffle=shuffle,
                                               buffer_size=buffer_size,
                                               repeat=repeat,
                                               filter_fn = get_filter_fn)
            self._img_num = len(img_paths)
            self.data_dir = data_dir
            self.atts = atts
            self.part = part
            self.img_paths = img_paths
            self.labels = labels
            self.offset_h = offset_h
            self.offset_w = offset_w
            self.img_size = img_size
        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num

    @staticmethod
    #check for celebA , no need for Age
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)

        for att in att_batch:
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
                _set(att, 0, 'Bangs')
            elif att_name == 'Bangs' and att[att_id] == 1:
                _set(att, 0, 'Bald')
                _set(att, 0, 'Receding_Hairline')
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
                # _set(att, 0, 'bald')
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
        # Removed since `Mustache` and `No_Beard` are not conflict.
        # But the two attributes are not well labeled in the dataset.
        #            elif att_name in ['Mustache', 'No_Beard'] and att[att_id] == 1:
        #                for n in ['Mustache', 'No_Beard']:
        #                    if n != att_name:
        #                        _set(att, 0, n)

        return att_batch

    @staticmethod
    def check_random_attribute_conflict(att_batch, att_names, hair_color=None):
        """ For randomly generated attributes, tested but not used in this repo. """

        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        def _idx(att_name):
            if att_name in att_names:
                return att_names.index(att_name)
            return None

        for att in att_batch:
            valid_atts = [i for i in ['Receding_Hairline', 'Bald'] if i in att_names]
            if 'Bangs' in att_names and att[_idx('Bangs')] == 1 \
                    and len(valid_atts) > 0 and sum([att[_idx(i)] for i in valid_atts]) > 0:
                _set(att, 0, 'Bangs') if random.random() < 0.5 else [_set(att, 0, i) for i in valid_atts]
            #            hair_color = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
            if hair_color is not None and sum([att[_idx(i)] for i in hair_color]) > 1:
                one = random.randint(0, len(hair_color))
                for i in range(len(hair_color)):
                    _set(att, 1 if i == one else 0, hair_color[i])
            if 'Straight_Hair' in att_names and 'Wavy_Hair' in att_names and att[_idx('Straight_Hair')] == 1 and att[
                _idx('Wavy_Hair')] == 1:
                _set(att, 0, 'Straight_Hair') if random.random() < 0.5 else _set(att, 0, 'Wavy_Hair')
        #            if 'Mustache' in att_names and 'No_Beard' in att_names and att[_idx('Mustache')] == 1 and att[_idx('No_Beard')] == 1:
        #                _set(att, 0, 'Mustache') if random.random() < 0.5 else _set(att, 0, 'No_Beard')
        return att_batch
    #creat example file by tfrecord
    def creat_tfrecord(self):
        if not os.path.exists(os.path.join(self.data_dir, "morph_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "morph_tfrecords"))
        count = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "morph_tfrecords", self.part + "_crop.tfrecords"))
        for i, img_path in enumerate(self.img_paths):
            if not os.path.exists(img_path):
                img_path = '.'.join([os.path.splitext(img_path)[0], 'jpg'])
            img = Image.open(os.path.join(img_path))
           #     for real_name in
           #        img = Image.open(re.match(os.path.join(img_path),real_name))
           #  img = img.crop((self.offset_w, self.offset_h, self.offset_w + self.img_size, self.offset_h + self.img_size))
            img_raw = img.tobytes()
            label = self.labels[i, :]    #one tensor change [i,:]
            #label[0] = age_group_div(label[0])

            img_name = os.path.split(img_path)[1]
            # img_name_byte = img_name.getBytes()
            # str.encode(img_name)
            #bytes(img_name , encoding="utf8")
            img_name_byte = img_name.encode(encoding='utf-8')
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                         'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name_byte]))}))     #gchange!
            writer.write(example.SerializeToString())
            count = count + 1
            if count % 500 == 0:
                print('Time:{0},{1} images are processed.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count))
        print("%d images are processed." % count)
        print(self.part + ' done!')
        writer.close()
    # creat labels by tfrecord
    def creat_tfrecord_split_opposite_label(self):
        if not os.path.exists(os.path.join(self.data_dir, "morph_tfrecords")):
            os.makedirs(os.path.join(self.data_dir, "morph_tfrecords"))
        for attribute_id in range(len(atts)):
            att = self.atts[attribute_id]
            count_true = 0
            count_false = 0
            if not os.path.exists(os.path.join(self.data_dir, "morph_tfrecords", self.part + "_"+ str(attribute_id) +
                                                                   "_True_crop.tfrecords")):
                true_writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "morph_tfrecords", self.part + "_"+ str(attribute_id) +
                                                                       "_True_crop.tfrecords"))
                false_writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, "morph_tfrecords", self.part + "_" + str(attribute_id) +
                                 "_False_crop.tfrecords"))
                for i, img_path in enumerate(self.img_paths):
                    img = Image.open(os.path.join(img_path))
                    img = img.crop((self.offset_w, self.offset_h, self.offset_w + self.img_size, self.offset_h + self.img_size))
                    img_raw = img.tobytes()
                    label = self.labels                                                                                #one tensor change [i,:]
                    img_name = os.path.split(img_path)[1]
                    img_name_byte = img_name.getBytes()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                                 'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name_byte]))}))
                    if label[attribute_id] == 1:
                        true_writer.write(example.SerializeToString())
                        count_true = count_true + 1
                        if count_true % 500 == 0:
                            print('Time:{0},{1} positive images are processed.'.format(
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count_true))
                    elif label[attribute_id] == -1:
                        false_writer.write(example.SerializeToString())
                        count_false = count_false + 1
                        if count_false % 500 == 0:
                            print('Time:{0},{1} negative images are processed.'.format(
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count_false))
                print("%d positive images are processed." % count_true)
                print("%d negative images are processed." % count_false)
                print(att + ' done!')
                true_writer.close()
                false_writer.close()

class Photo2Artworks(Dataset):
    def __init__(self, data_dir, img_resize, batch_size, prefetch_batch=2, drop_remainder=True,
                 num_threads=16, shuffle=True, buffer_size=2048, repeat=-1, sess=None, part='train',
                 im_no=None, is_tfrecord = False):
        super(Photo2Artworks, self).__init__()

        def one_hot(idx, length):
            output = [0 for _ in range(length)]
            output[idx] = 1
            return output

        def _map_func(img, label):
            # img = tf.image.resize_images(img, [img_resize, img_resize]) / 127.5 - 1
            # or
            img = tf.cast(img, tf.float32)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        if is_tfrecord:
            img_size = 256
            tfrecord_path = os.path.join(data_dir, 'Photo2Artworks' + '_tfrecords', part + ".tfrecords")

            dataset = tfrecord_batch_dataset(tfrecord_path=tfrecord_path,
                                             batch_size=batch_size,
                                             label_len = 5,
                                             prefetch_batch=prefetch_batch,
                                             drop_remainder=drop_remainder,
                                             map_func=_map_func,
                                             num_threads=num_threads,
                                             shuffle=shuffle,
                                             buffer_size=buffer_size,
                                             repeat=repeat,
                                             reshape_size=img_size)
            if part == 'train':
                self._img_num = 525+1072+6287+562+400

        else:
            img_dir_jpg = data_dir

            img_paths = []
            labels = []
            if part == 'test':
                drop_remainder = False
                shuffle = False
                repeat = 1
                domains = ['test_photo', 'test_cezanne','test_monet','test_ukiyoe','test_vangogh']
                for i, partAB in enumerate(domains):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.jpg'))
                    onehot_label = one_hot(i, len(domains))
                    img_paths.extend(imgAB_paths)
                    labels.extend([onehot_label for _ in range(len(imgAB_paths))])
            else:
                domains = ['train_photo', 'train_cezanne', 'train_monet', 'train_ukiyoe', 'train_vangogh']
                for i, partAB in enumerate(domains):
                    imgAB_paths = glob.glob(os.path.join(img_dir_jpg, partAB, '*.jpg'))
                    onehot_label = one_hot(i, len(domains))
                    if part == 'train':
                        img_paths.extend(imgAB_paths)
                        labels.extend([onehot_label for _ in range(len(imgAB_paths))])
            labels = np.array(labels)

            dataset = disk_image_batch_dataset(img_paths=img_paths,
                                               labels=labels,
                                               batch_size=batch_size,
                                               prefetch_batch=prefetch_batch,
                                               drop_remainder=drop_remainder,
                                               map_func=_map_func,
                                               num_threads=num_threads,
                                               shuffle=shuffle,
                                               buffer_size=buffer_size,
                                               repeat=repeat)
            self._img_num = len(img_paths)
            self.data_dir = data_dir
            self.part = part
            self.img_paths = img_paths
            self.labels = labels
        self._bulid(dataset, sess)

    def __len__(self):
        return self._img_num

    def creat_tfrecord(self):

        def shuffle_data():
            indices = np.random.permutation(self._img_num)
            self.img_paths = np.array(self.img_paths)[indices]
            # self.img_paths = self.img_paths.tolist()
            self.labels = self.labels[indices]
        shuffle_data()

        if not os.path.exists(os.path.join(self.data_dir, 'Photo2Artworks' + "_tfrecords")):
            os.makedirs(os.path.join(self.data_dir,'Photo2Artworks'+ "_tfrecords"))
        count = 0
        writer = tf.python_io.TFRecordWriter(os.path.join(self.data_dir, 'Photo2Artworks' + "_tfrecords", self.part + ".tfrecords"))
        for i, img_path in enumerate(self.img_paths):
            img = Image.open(os.path.join(img_path))
            img = img.convert("RGB")
            img_raw = img.tobytes()
            label = self.labels[i, :]
            img_name = os.path.split(img_path)[1]
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                         'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                         'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name]))}))
            writer.write(example.SerializeToString())
            count = count + 1
            if count % 500 == 0:
                print('Time:{0},{1} images are processed.'.format(
                    time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), count))
        print("%d images are processed." % count)
        print(self.part + ' done!')
        writer.close()
if __name__ == '__main__':
    import imlib as im

    # atts = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    #             'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
    #             'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
    #             'Brown_Hair', 'Bushy_Eyebrows', 'Chubby',
    #             'Double_Chin', 'Eyeglasses', 'Goatee',
    #             'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    #             'Male', 'Mouth_Slightly_Open', 'Mustache',
    #             'Narrow_Eyes', 'No_Beard', 'Oval_Face',
    #             'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    #             'Rosy_Cheeks', 'Sideburns', 'Smiling',
    #             'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    #             'Wearing_Hat', 'Wearing_Lipstick',
    #             'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    # atts = ['cat', 'dog', 'wild']
    # atts = ['Male','Young']
    atts = ['Young', 'Mouth_Slightly_Open', 'Smiling', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
                   'Receding_Hairline', 'Bangs', 'Male', 'No_Beard', 'Mustache', 'Goatee','Sideburns']
    data = Celeba('./data/CelebA', atts, 128, 32, part='train', is_tfrecord=True)
    # data.creat_tfrecord_split_true_label(['Young', 'Mouth_Slightly_Open', 'Smiling', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
    #            'Receding_Hairline', 'Bangs', 'Male', 'No_Beard', 'Mustache', 'Goatee','Sideburns'])
    batch = data.get_next()
    # print(len(data))
    # print(batch[1][1], batch[1].dtype)
    # print(batch[0].min(), batch[1].max(), batch[0].dtype)
    # im.imshow(batch[0][1])
    # im.show()
    # atts = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows', 'Eyeglasses',
    #                'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young']
    # data = Celeba('./data/CelebA', atts, 128, 32, part='train')
    # data = MakeupTransfer('./data', 128, 96, part = 'train', is_tfrecord=True, shuffle=True, filter_att = 0, filter_pos=0)#  , filter_att = 0, filter_pos=1
    # data = Morph(data_dir='./data/Morph', atts=atts, img_resize=128, batch_size=48, part='train', im_no=None,
    #              is_tfrecord=True, crop=False, filter_att=1, filter_pos=3)  # , filter_att=3, filter_pos=1
    # data = x2y('horse2zebra', './data/horse2zebra', 256, 16, part='train', is_tfrecord=False) , filter_att = 0, filter_pos=-1
    # data = x2y('summer2winter', './data/summer2winter', 256, 16, part='train', is_tfrecord=False)
    # data = Photo2Artworks('./data/Photo2Artworks', 256, 16, part = 'train', is_tfrecord=True)
    #data = AFHQ('./data/afhq', 128, 96, part='val', is_tfrecord=False, shuffle=True, filter_att = 1, filter_pos=1)
    #data.creat_tfrecord()
    # data.creat_tfrecord_split_opposite_label()
    # data = Celeba('./data/CelebA', atts, 128, 32, part='val', is_tfrecord=False)

    batch = data.get_next()
    # print(len(data))
    print(batch[1][1], batch[1].dtype)
    # print(batch[2][1], batch[1].dtype)
    print(batch[0].min(), batch[1].max(), batch[0].dtype)
    a = batch[0][2]
    im.imshow(batch[0][0])
    # im.imshow(batch[1][0])
    im.show()
