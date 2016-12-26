'''
prepare files for action recognition
_features.h5 _filename.txt _frame_num.txt _labels.txt
author: wyd
'''

import os
import h5py
import collections
import array
import numpy as np
import random
from sklearn.cross_validation import train_test_split


def read_binary_blob(filename):
    # Read binary blob file from C3D
    # precision is set to 'single', used by C3D
    # :param filename: input filename.
    # :return:
    #         s: a 1x5 matrix indicates the size of the blob, which is [num channel length height width].
    #         blob: a 5-D tensor size num x channel x length x height x width. containing the blob data.
    #         read_status : a scalar value = 1 if sucessfully read, 0 otherwise.
    read_status = 1
    blob = collections.namedtuple('Blob', ['size', 'data'])

    f = open(filename, 'rb')
    s = array.array("i")  # int32
    s.fromfile(f, 5)

    if len(s) == 5:
        m = s[0]*s[1]*s[2]*s[3]*s[4]
        data_aux = array.array("f")
        data_aux.fromfile(f, m)
        data = np.array(data_aux.tolist())

        if len(data) != m:
            read_status = 0
    else:
        read_status = 0
    # If failed to read, set empty output and return
    if not read_status:
        s = []
        blob_data = []
        b = blob(s, blob_data)
        return s, b, read_status
    # reshape the data buffer to blob
    # note that MATLAB use column order, while C3D uses row-order
    # blob = zeros(s(1), s(2), s(3), s(4), s(5), Float);
    blob_data = np.zeros((s[0], s[1], s[2], s[3], s[4]), np.float32)
    off = 0
    image_size = s[3]*s[4]
    for n in range(0, s[0]):
        for c in range(0, s[1]):
            for l in range(0, s[2]):
                # print n, c, l, off, off+image_size
                tmp = data[np.array(range(off, off+image_size))]
                blob_data[n][c][l][:][:] = tmp.reshape(s[3], -1)
                off = off+image_size
    b = blob(s, blob_data)
    f.close()
    return s, b, read_status


# make prefix-appended name
def _pf(pp, ap_name):
    return '%s_%s' % (pp, ap_name)

dataset = 'hmdb51'

if dataset is 'ucf11':
    actions = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling'
               'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']
    output_dir = '/home/wyd/C3D/examples/c3d_feature_extraction/output/UCF11/'
    bg_action = len(output_dir)

    # train test split
    with open('./lists/video_lists.txt') as vf:
        file_list = vf.read().splitlines()

    random.shuffle(file_list)

    X_train, X_test = train_test_split(file_list, test_size=0.3)

    for pref in ['train', 'test']:
        feat_h5f = h5py.File('../data/' + _pf(pref, 'features.h5'), 'w')
        feat_framenum = open('../data/' + _pf(pref, 'framenum.txt'), 'w')
        feat_filename = open('../data/' + _pf(pref, 'filename.txt'), 'w')
        feat_labels = open('../data/' + _pf(pref, 'labels.txt'), 'w')
        feat_h5dset = feat_h5f.create_dataset('features', (5000, 8192), compression="gzip")

        index = 0

        for video_line in eval(_pf('X', pref)):
            print video_line
            video_word = video_line .split()
            feat_dir = video_word[0].replace('input', 'output').strip('.avi')
            feat_filename.write(feat_dir + '\n')

            feat_num = int(video_word[2])
            feat_framenum.write(str(feat_num) + '\n')

            for action in actions:
                if feat_dir.find(action, bg_action) != -1:
                    feat_labels.write(str(actions.index(action)) + '\n')
            _f_num = 0
            for root, subdirs, files in os.walk(feat_dir):
                for name in files:
                    if name.find('.pool5') != -1:
                            _f_num += 1
            assert _f_num == feat_num

            for feat_ind in range(feat_num):
                _feat_name = '%06d.pool5' % (feat_ind*8)
                feat_name = os.path.join(feat_dir, _feat_name)
                # read feature file to array
                ss, bb, r_status = read_binary_blob(feat_name)
                if r_status:
                    feat_dim = ss[0]*ss[1]*ss[2]*ss[3]*ss[4]
                    feat_h5dset[index:index+1, :] = bb.data.reshape(1, feat_dim)
                    print index
                    index += 1
                else:
                    print 'read %s failed' % feat_name
        feat_h5dset.resize(index, axis=0)
        feat_h5f.close()
        feat_framenum.close()
        feat_filename.close()
        feat_labels.close()

elif dataset is 'hmdb51':

    output_dir = '/home/wyd/C3D/examples/c3d_feature_extraction/output/HMDB51/'
    actions = ['brush_hair', 'cartwheel', 'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive',
               'draw_sword', 'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac', 'golf',
               'handstand', 'hit', 'hug', 'jump', 'kick_ball', 'kick', 'kiss', 'laugh', 'pick', 'pour',
               'pullup', 'punch', 'push', 'pushup', 'ride_bike', 'ride_horse', 'run', 'shake_hands',
               'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 'smoke', 'somersault',
               'stand', 'swing_baseball', 'sword_exercise', 'sword', 'talk', 'throw', 'turn', 'walk', 'wave']

    split_dir = '/home/wyd/C3D/examples/c3d_feature_extraction/input/HMDB51/testTrainMulti_7030_splits/'
    split_id = 1

    train_feat_h5f = h5py.File('../data/train_features.h5', 'w')
    train_feat_framenum = open('../data/train_framenum.txt', 'w')
    train_feat_filename = open('../data/train_filename.txt', 'w')
    train_feat_labels = open('../data/train_labels.txt', 'w')
    train_feat_h5dset = train_feat_h5f.create_dataset('features', (600000, 8192), compression="gzip")

    test_feat_h5f = h5py.File('../data/test_features.h5', 'w')
    test_feat_framenum = open('../data/test_framenum.txt', 'w')
    test_feat_filename = open('../data/test_filename.txt', 'w')
    test_feat_labels = open('../data/test_labels.txt', 'w')
    test_feat_h5dset = test_feat_h5f.create_dataset('features', (200000, 8192), compression="gzip")

    train_index = 0
    test_index = 0

    for label, action in enumerate(actions):
        split_name = '%s_test_split%d.txt' % (action, split_id)
        for line in open(split_dir + split_name):
            words = line.split()
            if words[1] is '1' or words[1] is '2':
                feat_dir = os.path.join(output_dir, action, words[0].rstrip('.avi'))
                print 'feat dir: ', feat_dir
                
                for rt, subs, fs in os.walk(feat_dir):
                    feat_num = 0
                    for f_name in fs:
                        if f_name.find('.pool5') != -1:
                            feat_num += 1
                print 'chunk num: ', feat_num

                for feat_ind in range(feat_num):
                    _feat_name = '%06d.pool5' % (feat_ind*8)
                    feat_name = os.path.join(feat_dir, _feat_name) 
                    #print 'feature name: ', feat_name
                    ss, bb, r_status = read_binary_blob(feat_name)
                    if r_status:
                        feat_dim = ss[0] * ss[1] * ss[2] * ss[3] * ss[4]
                        if words[1] is '1':
                            print 'train: ', train_index
                            train_feat_h5dset[train_index:train_index + 1, :] = bb.data.reshape(1, feat_dim)
                            train_index += 1
                        if words[1] is '2':
                            print 'test: ', test_index
                            test_feat_h5dset[test_index:test_index + 1, :] = bb.data.reshape(1, feat_dim)
                            test_index += 1
                    else:
                        print 'read %s failed' % feat_name
                        exit()
                
                if words[1] is '1':
                    train_feat_filename.write(feat_dir + '\n')
                    train_feat_labels.write(str(actions.index(action)) + '\n')
                    train_feat_framenum.write(str(feat_num) + '\n')
                if words[1] is '2':
                    test_feat_filename.write(feat_dir + '\n')
                    test_feat_labels.write(str(actions.index(action)) + '\n')
                    test_feat_framenum.write(str(feat_num) + '\n')

    train_feat_h5dset.resize(train_index, axis=0)
    train_feat_h5f.close()
    train_feat_framenum.close()
    train_feat_filename.close()
    train_feat_labels.close()

    test_feat_h5dset.resize(test_index, axis=0)
    test_feat_h5f.close()
    test_feat_framenum.close()
    test_feat_filename.close()
    test_feat_labels.close()

