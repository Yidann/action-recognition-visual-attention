import numpy as np
caffe_path = '/home/wyd/caffe/python'
import os
import sys
sys.path.append(caffe_path)
import caffe
import cv2
import time
import random
from sklearn.cross_validation import train_test_split
import h5py

# load net
caffe.set_mode_gpu()
net = caffe.Net('/home/wyd/caffe/VGG_ILSVRC_16_layers_deploy.prototxt',
                '/home/wyd/caffe/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)


# make prefix-appended name
def _pf(pp, ap_name):
    return '%s_%s' % (pp, ap_name)

dataset = 'ucf11'

data_dir = '/home/wyd/C3D/examples/c3d_feature_extraction/input/UCF11/'
len_dir = len(data_dir)
# traversing the video_dir to get videolist
video_list_f = open('./lists/ucf11/video_lists.txt', 'w')

for root, subdirs, files in os.walk(data_dir):
    for name in files:
        file_name = os.path.join(root, name)
        if '.avi' in file_name or '.mpg' in file_name:
            video_list_f.write(file_name + '\n')
video_list_f.close()

resize_w = 112
resize_h = 112

mean = [104, 116, 123]
if dataset is 'ucf11':
    actions = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
               'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking'] 
    # train test split
    with open('./lists/ucf11/video_lists.txt') as vf:
        file_list = vf.read().splitlines()

    random.shuffle(file_list)

    X_train, X_test = train_test_split(file_list, test_size=0.3)

    for pref in ['train', 'test']:
        feat_h5f = h5py.File('../data/vgg/ucf11/' + _pf(pref, 'features.h5'), 'w', libver='latest')
        feat_framenum = open('../data/vgg/ucf11/' + _pf(pref, 'framenum.txt'), 'w')
        feat_filename = open('../data/vgg/ucf11/' + _pf(pref, 'filename.txt'), 'w')
        feat_labels = open('../data/vgg/ucf11/' + _pf(pref, 'labels.txt'), 'w')
        feat_h5dset = feat_h5f.create_dataset('features', (250000, 25088), compression="gzip")
        feat_array = np.zeros((600, 25088))
        index = 0

        for video_line in eval(_pf('X', pref)):
            video_name = video_line.strip()
            _start = time.time()
            print 'procsessing video: ', video_name
            cap = cv2.VideoCapture(video_name)
            if not cap.isOpened():
                print "could not open :", video_name
                continue
            num_frames = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                num_frames += 1  
                frame = cv2.resize(frame, (resize_w, resize_h))  # h w k
                in_ = np.array(frame, dtype=np.float32)
                in_ = in_[:,:,::-1]  # bgr2rgb
                in_ -= mean
                in_ = in_.transpose((2,0,1))
                # shape for input (data blob is N x C x H x W), set data
                net.blobs['data'].reshape(1, *in_.shape)
                net.blobs['data'].data[...] = in_
                net.forward()
                feature = net.blobs['conv5_3'].data[0,:,:,:]  # N X C X H X W
                feat_array[num_frames-1:num_frames, :] = feature.reshape(1, 25088)
                index += 1
                #print 'index: ', index
            print 'num of frames: ', num_frames
            feat_h5dset[index-num_frames:index, :] = feat_array[0:num_frames, :]
            _end = time.time()
            print 'time to process a video : ', (_end - _start)
            feat_filename.write(video_name + '\n')
            feat_framenum.write(str(num_frames) + '\n')
            label_name = video_name[len_dir:-1]
            label_name = label_name[:label_name.find('/')]
            print 'label: ', label_name
            assert label_name in actions, 'label is not in list'
            feat_labels.write(str(actions.index(label_name)) + '\n')
        feat_h5dset.resize(index, axis=0)
        feat_h5f.close()
        feat_framenum.close()
        feat_filename.close()
        feat_labels.close()
