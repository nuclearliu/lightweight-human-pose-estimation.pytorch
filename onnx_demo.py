import argparse

import cv2
import numpy as np
import torch
import onnxruntime

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name

        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(self.file_name)

    def __iter__(self):

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


class Exercise(object):
    def __init__(self):
        self.ExerciseList = ["Weight lifting", "Power twister"]
        self.CurrentExercise = 0
        self.count = 0
        self.arm = 0
        self.updated = False
        self.isReady = False
        self.id = None

    def calc_angle(self, a, b, c, d):
        upper_arm = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        lower_arm = np.array([c[0] - d[0], c[1] - d[1]], dtype=float)
        angle = np.dot(upper_arm, lower_arm) / (np.linalg.norm(upper_arm) * np.linalg.norm(lower_arm))
        return angle

    def calc_dist(self, a, b):
        return np.sqrt(np.sum(np.array([a[0] - b[0], a[1] - b[1]], dtype=float) * np.array([a[0] - b[0], a[1] - b[1]], dtype=float)))

    def getReady(self, pose):
        if self.CurrentExercise == 0:
            if pose.keypoints[2][0] != -1 and pose.keypoints[3][0] != -1 and pose.keypoints[4][0] != -1:
                ang = self.calc_angle(pose.keypoints[2], pose.keypoints[3], pose.keypoints[3], pose.keypoints[4])
                if ang > 0.8:
                    self.arm = 0
                    self.isReady = True
        elif self.CurrentExercise == 1:
            if -1 not in [pose.keypoints[i][0] for i in range(2, 8)]:
                ang = self.calc_angle(pose.keypoints[2], pose.keypoints[5], pose.keypoints[4], pose.keypoints[7])
                if ang > 0.95 and pose.keypoints[3][1] > pose.keypoints[4][1] and pose.keypoints[6][1] > pose.keypoints[7][1]:
                    self.arm = 0
                    self.isReady = True

    def update(self, pose):
        if self.id is None:
            self.id = pose.id
        if self.id != pose.id:
            return
        if self.isReady:
            if self.CurrentExercise == 0:
                self.WeightLifting(pose)
            elif self.CurrentExercise == 1:
                self.PowerTwister(pose)
            if self.updated:
                print(self.count)
                self.updated = False
        else:
            self.getReady(pose)

    def WeightLifting(self, pose):
        if pose.keypoints[2][0] != -1 and pose.keypoints[3][0] != -1 and pose.keypoints[4][0] != -1:
            ang = self.calc_angle(pose.keypoints[2], pose.keypoints[3], pose.keypoints[3], pose.keypoints[4])
            if ang > 0.8:
                # print("down")
                if self.arm == 0:
                    pass
                elif self.arm == 1:
                    self.arm = 0
                    self.count += 1
                    self.updated = True
            elif ang < -0.6:
                # print("up")
                if self.arm == 1:
                    pass
                elif self.arm == 0:
                    self.arm = 1
            # print(ang)

    def PowerTwister(self, pose):
        if -1 not in [pose.keypoints[i][0] for i in range(2, 8)]:
            shoulder_dist = self.calc_dist(pose.keypoints[2], pose.keypoints[5])
            wrist_dist = self.calc_dist(pose.keypoints[4], pose.keypoints[7])
            if wrist_dist < shoulder_dist:
                if self.arm == 0:
                    pass
                elif self.arm == 1:
                    self.arm = 0
                    self.count += 1
                    self.updated = True
            else:
                if self.arm == 1:
                    pass
                elif self.arm == 0:
                    self.arm = 1



def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = np.expand_dims(padded_img.transpose(2, 0, 1), 0)

    # stages_output = net(tensor_img)
    ort_inputs = {net.get_inputs()[0].name: tensor_img}
    stages_output = net.run(None, ort_inputs)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().data, (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().data, (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    # print(heatmaps.shape, pafs.shape, scale, pad)
    return heatmaps, pafs, scale, pad


def run_demo(net, image_provider, height_size, cpu, track, smooth, exercise):
    # net = net.eval()
    # if not cpu:
    #     net = net.cuda()
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        # out_img = np.zeros(orig_img.shape[:2], dtype=np.uint8) + 255
        for pose in current_poses:
            pose.draw(img)
            exercise.update(pose)

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        # img = out_img
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        if not exercise.isReady:
            message_bgcolor = [90, 179, 81] # BGR
            font_color = [255, 255, 255]
            message = "Please get ready"
            message_region = np.array([[message_bgcolor]], dtype=np.uint8) * np.ones((200, 1920, 3), dtype=np.uint8)
            text_position = (550, 130)
        else:
            message_bgcolor = [245, 204, 138]
            font_color = [255, 255, 255]
            message = exercise.ExerciseList[exercise.CurrentExercise] + ": " + str(exercise.count)
            message_region = np.array([[message_bgcolor]], dtype=np.uint8) * np.ones((200, 1920, 3), dtype=np.uint8)
            text_position = (550, 130)
        cv2.putText(message_region, message, text_position, cv2.FONT_HERSHEY_SIMPLEX, 3, font_color, 5)

        img = np.vstack((message_region, img))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    # parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='rtsp://192.168.31.25', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default=[''], help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', default='True', help='run network inference on cpu')
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    args = parser.parse_args()
    exercise = Exercise()
    exercise.CurrentExercise = 1

    if args.video == '' and args.images == '':
        raise ValueError('Either --video or --image has to be provided')

    # net = PoseEstimationWithMobileNet()
    net = onnxruntime.InferenceSession("scripts/human-pose-estimation.onnx")
    # checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    # load_state(net, checkpoint)

    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
    else:
        args.track = 0

    run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth, exercise)
