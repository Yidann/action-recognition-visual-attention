'''
extract c3d features files,
produce video_lists as format: video_name(full_path) frame_num feature_num
'''
import os
import stat
import subprocess
from get_frame_num import get_frame_num


# video_dir = "/home/wyd/C3D/examples/c3d_feature_extraction/input/HMDB51"
# if not os.path.exists(video_dir):
#     print "cannot find video directory: %s" % video_dir
#
# # create input list
# # <string path> <starting frame> <label>(0)
# step_size = 8
input_list = "./lists/c3d/input_lists.txt"
# input_f = open(input_list, 'w')
# # create outputlist
# # <output_prefix>
output_list = "./lists/c3d/output_lists.txt"
# output_f = open(output_list, 'w')
#
# video_list_f = open('./lists/video_lists.txt', 'w')
#
# # traversing the video_dir
# for root, subdirs, files in os.walk(video_dir):
#     # for sub in subdirs:
#     #     print os.path.join(root, sub)
#     for name in files:
#         file_name = os.path.join(root, name)
#         if '.avi' in file_name:
#             video_path = file_name
#             output_folder = video_path.replace('input', 'output').strip('.avi')
#             if not os.path.exists(output_folder):
#                 os.makedirs(output_folder)
#             # get the frame num
#             print 'video: ', video_path
#             f_num = get_frame_num(video_path)
#             print 'frame: ', f_num
#             video_list_f.write(video_path + ' ' + str(f_num))  # write video name in lists
#             chunk_num = 0
#             for start in range(0, f_num-step_size+1, step_size):
#                 input_f.write(video_path + ' ')
#                 input_f.write(str(start))
#                 input_f.write(' 0\n')
#                 # create an output folder for each video:
#                 # output_folder/%06d % start
#                 output_f.write(output_folder)
#                 output_prefix = '/%06d' % start
#                 output_f.write(output_prefix)
#                 output_f.write('\n')
#                 chunk_num += 1
#             video_list_f.write(' ' + str(chunk_num) + '\n')
#
# video_list_f.close()
# input_f.close()
# output_f.close()
# run
data_dir = 'HMDB51'
cpp_file = './cpp.sh'
with open(cpp_file, 'w') as cf:
    cf.write('./build/C3D ')
    cf.write(data_dir)
os.chmod(cpp_file, stat.S_IRWXU)
subprocess.call(cpp_file, shell=True)
# change the prototxt file
mean_file = "/home/wyd/C3D/examples/c3d_feature_extraction/sport1m_train16_128_mean.binaryproto"
org_proto_file = "/home/wyd/C3D/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt"
proto_file = "./lists/c3d/c3d_sport1m_feature_extractor_video.prototxt"
proto_f = open(proto_file, 'w')
for line in open(org_proto_file):
    words = line.split()
    if 'source:' in words:
        proto_f.write('    source: ')
        proto_f.write('\"{}\"'.format(input_list))
        proto_f.write('\n')
    elif 'mean_file:' in words:
        proto_f.write('    mean_file: ')
        proto_f.write('\"{}\"'.format(mean_file))
        proto_f.write('\n')
    else:
        proto_f.write(line)
proto_f.close()

pretrained_model = "/home/wyd/C3D/examples/c3d_feature_extraction/conv3d_deepnetA_sport1m_iter_1900000"
gpu_id = '0'
batch_size = '50'
batch_num = '136'
feature_names = "pool5"

job_file = "./job.sh"
# create job file
with open(job_file, 'w') as f:
    f.write('GLOG_logtosterr=1 ')
    f.write('/home/wyd/C3D/build/tools/extract_image_features.bin ')
    f.write(proto_file + ' ')
    f.write(pretrained_model + ' ')
    f.write('{} {} {} '.format(gpu_id, batch_size, batch_num))
    f.write(output_list + ' ')
    f.write(feature_names)
# run the job
# os.chmod(job_file, stat.S_IRWXU)
# subprocess.call(job_file, shell=True)

# after this you will get the features in output_dir saved as
# $output_dir/$action/$video_name/$start_frame.feature
