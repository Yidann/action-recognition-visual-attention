# a function to get video frame num
import os
import skvideo.io


def get_frame_num(video_path):
    # video_path (string): path where the video is stored
    if not os.path.exists(video_path):
        print "no such file called :", video_path
        return -1

    print video_path
    cap = skvideo.io.VideoCapture(video_path)

    if not cap.isOpened():
        print "could not open :", video_path
        return -1
    num_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        num_frames += 1
    # print "the num_frames of %s is %d" % (video_path, num_frames)
    return num_frames

if __name__ == '__main__':
    frames = get_frame_num("/home/wyd/C3D/examples/c3d_feature_extraction/input/HMDB51/sword/Takeda_Ryu_Iaido_sword_f_nm_np2_le_med_1.avi")
    print frames
