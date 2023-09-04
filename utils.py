
import torch

import pdb

import cv2
cv2.setNumThreads(0)
import numpy as np

import math

from config import *



def bbox_overlaps(boxes,query_boxes):
    """
    inputs:
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float

    outputs:
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K))

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def nms(boxes, scores, overlap = 0.5, top_k = 200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    count = 0
    if boxes.numel() == 0:
        return keep, count
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out = xx1)
        torch.index_select(y1, 0, idx, out = yy1)
        torch.index_select(x2, 0, idx, out = xx2)
        torch.index_select(y2, 0, idx, out = yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min = x1[i])
        yy1 = torch.clamp(yy1, min = y1[i])
        xx2 = torch.clamp(xx2, max = x2[i])
        yy2 = torch.clamp(yy2, max = y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min = 0.0)
        h = torch.clamp(h, min = 0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]

    return keep, count


def colorize(value, factor = 1, vmin = None, vmax = None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - factor: resize factor, scalar
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    value = np.sum(value, axis = -1)
    vmin = np.min(value) if vmin is None else vmin
    vmax = np.max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    value = (value * 255).astype(np.uint8)
    value = cv2.applyColorMap(value, cv2.COLORMAP_JET)
    value = cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    x, y, _ = value.shape
    value = cv2.resize(value, (y * factor, x * factor))

    return value


#Util function to load calib matrices
CAM = 2
def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [line.split()[1:] for line in lines][:-1]

    P = np.array(lines[CAM]).reshape(3, 4)
    P = np.concatenate((P, np.array([[0, 0, 0, 0]])), 0)

    Tr_velo_to_cam = np.array(lines[5]).reshape(3, 4)
    Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3, :3] = np.array(lines[4][:9]).reshape(3, 3)

    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')

    return P, Tr_velo_to_cam, R_cam_to_rect


def lidar_to_bird_view(x, y, factor = 1):
    # using the cfg.INPUT_XXX
    a = (x - config.X_MIN) / config.VOXEL_X_SIZE * factor
    b = (y - config.Y_MIN) / config.VOXEL_Y_SIZE * factor
    a = np.clip(a, a_max = (config.X_MAX - config.X_MIN) / config.VOXEL_X_SIZE * factor, a_min = 0)
    b = np.clip(b, a_max = (config.Y_MAX - config.Y_MIN) / config.VOXEL_Y_SIZE * factor, a_min = 0)

    return a, b

def batch_lidar_to_bird_view(points, factor = 1):
    # Input:
    #   points (N, 2)
    # Outputs:
    #   points (N, 2)
    # using the cfg.INPUT_XXX
    a = (points[:, 0] - config.X_MIN) / config.VOXEL_X_SIZE * factor
    b = (points[:, 1] - config.Y_MIN) / config.VOXEL_Y_SIZE * factor
    a = np.clip(a, a_max = (config.X_MAX - config.X_MIN) / config.VOXEL_X_SIZE * factor, a_min = 0)
    b = np.clip(b, a_max = (config.Y_MAX - config.Y_MIN) / config.VOXEL_Y_SIZE * factor, a_min = 0)

    return np.concatenate([a[:, np.newaxis], b[:, np.newaxis]], axis = -1)


def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2

    return angle


def camera_to_lidar(x, y, z, T_VELO_2_CAM = None, R_RECT_0 = None):
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(config.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(config.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(R_RECT_0), p)
    p = np.matmul(np.linalg.inv(T_VELO_2_CAM), p)
    p = p[0 : 3]

    return tuple(p)


def lidar_to_camera(x, y, z, T_VELO_2_CAM = None, R_RECT_0 = None):
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(config.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(config.MATRIX_R_RECT_0)

    p = np.array([x, y, z, 1])
    p = np.matmul(T_VELO_2_CAM, p)
    p = np.matmul(R_RECT_0, p)
    p = p[0:3]

    return tuple(p)


def camera_to_lidar_point(points, T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)

    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(config.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(config.MATRIX_R_RECT_0)

    points = np.matmul(np.linalg.inv(R_RECT_0), points)
    points = np.matmul(np.linalg.inv(T_VELO_2_CAM), points).T  # (4, N) -> (N, 4)
    points = points[:, 0:3]

    return points.reshape(-1, 3)


def lidar_to_camera_point(points, T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T
    
    
    if type(T_VELO_2_CAM) == type(None):
        T_VELO_2_CAM = np.array(config.MATRIX_T_VELO_2_CAM)
    
    if type(R_RECT_0) == type(None):
        R_RECT_0 = np.array(config.MATRIX_R_RECT_0)

    points = np.matmul(T_VELO_2_CAM, points)
    points = np.matmul(R_RECT_0, points).T
    points = points[:, 0:3]

    return points.reshape(-1, 3)


def camera_to_lidar_box(boxes, T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])

    return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes, T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z, T_VELO_2_CAM, R_RECT_0), h, w, l, -rz - np.pi / 2
        ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])

    return np.array(ret).reshape(-1, 7)


def center_to_corner_box2d(boxes_center, coordinate = 'lidar', T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = center_to_corner_box3d(
        boxes3d_center, coordinate = coordinate, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)

    return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center, coordinate = 'lidar', T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype = np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center, T_VELO_2_CAM, R_RECT_0)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx], T_VELO_2_CAM, R_RECT_0)

    return ret


def corner_to_center_box2d(boxes_corner, coordinate='lidar', T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 4, 2) -> (N, 5); x,y,w,l,r
    N = boxes_corner.shape[0]
    boxes3d_corner = np.zeros((N, 8, 3))
    boxes3d_corner[:, 0:4, 0:2] = boxes_corner
    boxes3d_corner[:, 4:8, 0:2] = boxes_corner
    boxes3d_center = corner_to_center_box3d(
        boxes3d_corner, coordinate = coordinate, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)

    return boxes3d_center[:, [0, 1, 4, 5, 6]]


def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4); x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis = 1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis = 1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis = 1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis = 1)

    return standup_boxes2d


# TODO: 0/90 may be not correct
def anchor_to_standup_box2d(anchors):
    # (N, 4) -> (N, 4); x,y,w,l -> x1,y1,x2,y2
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    return anchor_standup


def corner_to_center_box3d(boxes_corner, coordinate='camera', T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 8, 3) -> (N, 7); x,y,z,h,w,l,ry/z
    if coordinate == 'lidar':
        for idx in range(len(boxes_corner)):
            boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx], T_VELO_2_CAM, R_RECT_0)
    ret = []
    for roi in boxes_corner:
        if config.CORNER2CENTER_AVG:  # average version
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x = np.sum(roi[:, 0], axis = 0)/ 8
            y = np.sum(roi[0:4, 1], axis = 0)/ 4
            z = np.sum(roi[:, 2], axis = 0)/ 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x = np.sum(roi[:, 0], axis = 0)/ 8
            y = np.sum(roi[0:4, 1], axis = 0)/ 4
            z = np.sum(roi[:, 2], axis = 0)/ 8
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        ret.append([x, y, z, h, w, l, ry])
    if coordinate == 'lidar':
        ret = camera_to_lidar_box(np.array(ret), T_VELO_2_CAM, R_RECT_0)

    return np.array(ret)


# This just for visulize and testing
def lidar_box3d_to_camera_box(boxes3d, cal_projection = False, P2 = None, T_VELO_2_CAM = None, R_RECT_0 = None):
    # (N, 7) -> (N, 4)/(N, 8, 2); x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype = np.int32)
    projections = np.zeros((num, 8, 2), dtype = np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d, coordinate = 'lidar', T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    if type(P2) == type(None):
        P2 = np.array(config.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d, T_VELO_2_CAM, R_RECT_0)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T

        points = np.nan_to_num(points)

        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = 0 if np.isnan(np.min(points[:, 0])) else int(np.min(points[:, 0]))
        maxx = 0 if np.isnan(np.max(points[:, 0])) else int(np.max(points[:, 0]))
        miny = 0 if np.isnan(np.min(points[:, 1])) else int(np.min(points[:, 1]))
        maxy = 0 if np.isnan(np.max(points[:, 1])) else int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d


def lidar_to_bird_view_img(lidar, factor = 1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (config.INPUT_HEIGHT * factor, config.INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if config.X_MIN < x < config.X_MAX and config.Y_MIN < y < config.Y_MAX:
            x, y = int((x - config.X_MIN) / config.VOXEL_X_SIZE *
                       factor), int((y - config.Y_MIN) / config.VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min = 0, a_max = 255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview


def draw_lidar_box3d_on_image(img, boxes3d, scores, gt_boxes3d = np.array([]), color = (0, 255, 255),
                              gt_color = (255, 0, 255), thickness = 1, P2 = None, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection = True, P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection = True, P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)

    # Draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)
    # Draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    


def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d = np.array([]),
                                 color = (0, 255, 255), gt_color = (255, 0, 255), thickness = 1, factor = 1, P2 = None, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate = 'lidar', T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate = 'lidar', T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor = factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor = factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor = factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor = factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor = factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor = factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor = factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor = factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def label_to_gt_box3d(labels, cls = 'Car', coordinate = 'camera', T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else: # all
        acc_cls = []

    for label in labels:
        boxes3d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
                box3d = np.array([x, y, z, h, w, l, r])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label), T_VELO_2_CAM, R_RECT_0)

        boxes3d.append(np.array(boxes3d_a_label).reshape(-1, 7))

    return boxes3d


def box3d_to_label(batch_box3d, batch_cls, batch_score = [], coordinate='camera', P2 = None, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input:
    #   (N, N', 7) x y z h w l r
    #   (N, N')
    #   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate(input): 'camera' or 'lidar'
    # Output:
    #   label: (N, N') N batches and N lines
    batch_label = []
    if batch_score:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
        for boxes, scores, clses in zip(batch_box3d, batch_score, batch_cls):
            label = []
            for box, score, cls in zip(boxes, scores, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0), cal_projection = False,
                        P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection = False, P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(
                    cls, 0, 0, 0, *box2d, *box3d, float(score)))
            batch_label.append(label)
    else:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(14)]) + '\n'
        for boxes, clses in zip(batch_box3d, batch_cls):
            label = []
            for box, cls in zip(boxes, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0),
                        cal_projection = False,  P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), T_VELO_2_CAM, R_RECT_0)[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection = False, P2 = P2, T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(cls, 0, 0, 0, *box2d, *box3d))
            batch_label.append(label)

    return np.array(batch_label)


def cal_anchors():
    # Output:
    # Anchors: (w, l, 2, 7) x y z h w l r
    x = np.linspace(config.X_MIN, config.X_MAX, config.FEATURE_WIDTH)
    y = np.linspace(config.Y_MIN, config.Y_MAX, config.FEATURE_HEIGHT)
    cx, cy = np.meshgrid(x, y)
    # All are (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * config.ANCHOR_Z
    w = np.ones_like(cx) * config.ANCHOR_W
    l = np.ones_like(cx) * config.ANCHOR_L
    h = np.ones_like(cx) * config.ANCHOR_H
    r = np.ones_like(cx)
    r[..., 0] = 0  # 0
    r[..., 1] = 90 / 180 * np.pi  # 90

    # 7 * (w, l, 2) -> (w, l, 2, 7)
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis = -1)

    return anchors


def cal_rpn_target(labels, feature_map_shape, anchors, cls = 'Car', coordinate = 'lidar'):
    # Input:
    #   labels: (N, N')
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)
    # Output:
    #   pos_equal_one (N, w, l, 2)
    #   neg_equal_one (N, w, l, 2)
    #   targets (N, w, l, 14)
    # Attention: cal IoU on birdview

    batch_size = labels.shape[0]
    batch_gt_boxes3d = label_to_gt_box3d(labels, cls = cls, coordinate = coordinate)
    # Defined in eq(1) in 2.2
    anchors_reshaped = anchors.reshape(-1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
    pos_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    neg_equal_one = np.zeros((batch_size, *feature_map_shape, 2))
    targets = np.zeros((batch_size, *feature_map_shape, 14))

    for batch_id in range(batch_size):
        # BOTTLENECK; from (x,y,w,l) to (x1,y1,x2,y2)
        anchors_standup_2d = anchor_to_standup_box2d(anchors_reshaped[:, [0, 1, 4, 5]])
        # BOTTLENECK
        gt_standup_2d = corner_to_standup_box2d(center_to_corner_box2d(
            batch_gt_boxes3d[batch_id][:, [0, 1, 4, 5, 6]], coordinate = coordinate))

        iou = bbox_overlaps(
            np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
            np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        )
        # iou = cal_box3d_iou(anchors_reshaped, batch_gt_boxes3d[batch_id])

        # Find anchor with highest iou (iou should also > 0)
        id_highest = np.argmax(iou.T, axis = 1)
        id_highest_gt = np.arange(iou.T.shape[0])
        mask = iou.T[id_highest_gt, id_highest] > 0
        id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # Find anchor iou > cfg.XXX_POS_IOU
        id_pos, id_pos_gt = np.where(iou > config.RPN_POS_IOU)

        # Find anchor iou < cfg.XXX_NEG_IOU
        id_neg = np.where(np.sum(iou < config.RPN_NEG_IOU, axis = 1) == iou.shape[1])[0]

        id_pos = np.concatenate([id_pos, id_highest])
        id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        # TODO: uniquify the array in a more scientific way
        id_pos, index = np.unique(id_pos, return_index = True)
        id_pos_gt = id_pos_gt[index]
        id_neg.sort()

        # Cal the target and set the equal one
        index_x, index_y, index_z = np.unravel_index(id_pos, (*feature_map_shape, 2))
        pos_equal_one[batch_id, index_x, index_y, index_z] = 1

        # ATTENTION: index_z should be np.array
        targets[batch_id, index_x, index_y, np.array(index_z) * 7] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 0] - anchors_reshaped[id_pos, 0]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 1] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 1] - anchors_reshaped[id_pos, 1]) / anchors_d[id_pos]
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 2] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 2] - anchors_reshaped[id_pos, 2]) / config.ANCHOR_H
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 3])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 4] / anchors_reshaped[id_pos, 4])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
            batch_gt_boxes3d[batch_id][id_pos_gt, 5] / anchors_reshaped[id_pos, 5])
        targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 6] = (
            batch_gt_boxes3d[batch_id][id_pos_gt, 6] - anchors_reshaped[id_pos, 6])

        index_x, index_y, index_z = np.unravel_index(id_neg, (*feature_map_shape, 2))
        neg_equal_one[batch_id, index_x, index_y, index_z] = 1
        # To avoid a box be pos/neg in the same time
        index_x, index_y, index_z = np.unravel_index(id_highest, (*feature_map_shape, 2))
        neg_equal_one[batch_id, index_x, index_y, index_z] = 0

    return pos_equal_one, neg_equal_one, targets


# BOTTLENECK
def delta_to_boxes3d(deltas, anchors, coordinate = 'lidar'):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    anchors_reshaped = anchors.reshape(-1, 7)
    deltas = deltas.reshape(deltas.shape[0], -1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
    boxes3d = np.zeros_like(deltas)
    boxes3d[..., [0, 1]] = deltas[..., [0, 1]] * \
        anchors_d[:, np.newaxis] + anchors_reshaped[..., [0, 1]]
    boxes3d[..., [2]] = deltas[..., [2]] * \
        config.ANCHOR_H + anchors_reshaped[..., [2]]
    boxes3d[..., [3, 4, 5]] = np.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

    return boxes3d


def point_transform(points, tx, ty, tz, rx = 0, ry = 0, rz = 0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])

    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)

    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)

    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)

    return points[:, 0:3]


def box_transform(boxes, tx, ty, tz, r = 0, coordinate = 'lidar'):
    # Input:
    #   boxes: (N, 7) x y z h w l rz/y
    # Output:
    #   boxes: (N, 7) x y z h w l rz/y
    boxes_corner = center_to_corner_box3d(
        boxes, coordinate = coordinate)  # (N, 8, 3)
    for idx in range(len(boxes_corner)):
        if coordinate == 'lidar':
            boxes_corner[idx] = point_transform(
                boxes_corner[idx], tx, ty, tz, rz = r)
        else:
            boxes_corner[idx] = point_transform(
                boxes_corner[idx], tx, ty, tz, ry = r)

    return corner_to_center_box3d(boxes_corner, coordinate = coordinate)


def cal_iou2d(box1, box2, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input: 
    #   box1/2: x, y, w, l, r
    # Output :
    #   iou
    buf1 = np.zeros((config.INPUT_HEIGHT, config.INPUT_WIDTH, 3))
    buf2 = np.zeros((config.INPUT_HEIGHT, config.INPUT_WIDTH, 3))
    tmp = center_to_corner_box2d(np.array([box1, box2]), coordinate = 'lidar', T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    box1_corner = batch_lidar_to_bird_view(tmp[0]).astype(np.int32)
    box2_corner = batch_lidar_to_bird_view(tmp[1]).astype(np.int32)
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color = (1, 1, 1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color = (1, 1, 1))[..., 0]
    indiv = np.sum(np.absolute(buf1-buf2))
    share = np.sum((buf1 + buf2) == 2)
    if indiv == 0:
        return 0.0 # when target is out of bound

    return share / (indiv + share)


def cal_z_intersect(cz1, h1, cz2, h2):
    b1z1, b1z2 = cz1 - h1 / 2, cz1 + h1 / 2
    b2z1, b2z2 = cz2 - h2 / 2, cz2 + h2 / 2
    if b1z1 > b2z2 or b2z1 > b1z2:
        return 0
    elif b2z1 <= b1z1 <= b2z2:
        if b1z2 <= b2z2:
            return h1 / h2
        else:
            return (b2z2 - b1z1) / (b1z2 - b2z1)
    elif b1z1 < b2z1 < b1z2:
        if b2z2 <= b1z2:
            return h2 / h1
        else:
            return (b1z2 - b2z1) / (b2z2 - b1z1)

def cal_iou3d(box1, box2, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Input:
    #   box1/2: x, y, z, h, w, l, r
    # Output:
    #   iou
    buf1 = np.zeros((config.INPUT_HEIGHT, config.INPUT_WIDTH, 3))
    buf2 = np.zeros((config.INPUT_HEIGHT, config.INPUT_WIDTH, 3))
    tmp = center_to_corner_box2d(np.array([box1[[0, 1, 4, 5, 6]], box2[[0, 1, 4, 5, 6]]]), coordinate = 'lidar',
                                 T_VELO_2_CAM = T_VELO_2_CAM, R_RECT_0 = R_RECT_0)
    box1_corner = batch_lidar_to_bird_view(tmp[0]).astype(np.int32)
    box2_corner = batch_lidar_to_bird_view(tmp[1]).astype(np.int32)
    buf1 = cv2.fillConvexPoly(buf1, box1_corner, color = (1, 1, 1))[..., 0]
    buf2 = cv2.fillConvexPoly(buf2, box2_corner, color = (1, 1, 1))[..., 0]
    share = np.sum((buf1 + buf2) == 2)
    area1 = np.sum(buf1)
    area2 = np.sum(buf2)
    
    z1, h1, z2, h2 = box1[2], box1[3], box2[2], box2[3]
    z_intersect = cal_z_intersect(z1, h1, z2, h2)

    return share * z_intersect / (area1 * h1 + area2 * h2 - share * z_intersect)


def cal_box3d_iou(boxes3d, gt_boxes3d, cal_3d = 0, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes3d)
    N2 = len(gt_boxes3d)
    output = np.zeros((N1, N2), dtype = np.float32)

    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = float(cal_iou3d(boxes3d[idx], gt_boxes3d[idy], T_VELO_2_CAM, R_RECT_0))
            else:
                output[idx, idy] = float(
                    cal_iou2d(boxes3d[idx, [0, 1, 4, 5, 6]], gt_boxes3d[idy, [0, 1, 4, 5, 6]], T_VELO_2_CAM, R_RECT_0))

    return output


def cal_box2d_iou(boxes2d, gt_boxes2d, T_VELO_2_CAM = None, R_RECT_0 = None):
    # Inputs:
    #   boxes2d: (N1, 5) x,y,w,l,r
    #   gt_boxes2d: (N2, 5) x,y,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes2d)
    N2 = len(gt_boxes2d)
    output = np.zeros((N1, N2), dtype = np.float32)
    for idx in range(N1):
        for idy in range(N2):
            output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy], T_VELO_2_CAM, R_RECT_0)

    return output



def down_sample_data(inputs,portion):
    # randomly select points from the point cloud and delete them
    # selection portion from all the points
    tag = inputs[0]
    label = inputs[1]
    vox_features = inputs[2]
    vox_numbers = inputs[3]
    vox_coordinates = inputs[4]
    
    reduce_number = int(vox_numbers * portion)
    vox_numbers -= reduce_number
    for _ in range(reduce_number):
        random_index = np.random.choice(range(vox_numbers))
        del vox_coordinates[random_index]
        del vox_features[random_index]
    return [tag,label,vox_features,vox_numbers,vox_coordinates]

def partial_omit_sample(inputs,portion):
    # a box no more than the size of portion is created
    # and eery point within the box will be deleted
    tag = inputs[0]
    label = inputs[1]
    vox_features = inputs[2]
    vox_numbers = inputs[3]
    vox_coordinates = inputs[4]

    reduce_box = np.random.random_sample([4,]) * portion
    counter = 0
    for i in range(vox_numbers):
        loc_coord = vox_coordinates[i]
        in_lower = loc_coord > reduce_box[:2]
        in_higher = reduce_box[2:] > loc_coord
        if in_lower[0] and in_lower[1] and in_higher[0] and in_higher[1]:
            # this is a point within the box !
            counter += 1
            del vox_features[i]
            del vox_coordinates[i] 
    vox_numbers -= counter
    return [tag,label,vox_features,vox_numbers,vox_coordinates]

from torch.utils.data import DataLoader

def eval(model,dataset):
    if not isinstance(dataset,DataLoader):
        dataloader = DataLoader(dataset,shuffle = True,batch_size = 4)
    else: dataloader = dataset

    pt_at = 0
    pt_af = 0
    pf_at = 0
    pf_af = 0

    for sample in dataloader:
        x,label = sample

        prediction,feature,_ = model(x.permute(0,2,1))
        cls_num = prediction[0].shape[0]
        for cls in range(cls_num):
            for i in range(label.shape[0]):
                predict_label = np.argmax(prediction[i].cpu().detach().numpy())
                actual_label = int(label[i])
                if predict_label == cls:
                    if actual_label == predict_label:pt_at += 1.
                    else:pt_af += 1.

                if predict_label != cls:
                    if actual_label == predict_label:pf_at += 1.
                    else:pf_af += 1.
    accuracy = (pt_at + pt_af) / (pt_at + pt_af + pf_at + pf_af)
    precision = pt_at/(pt_at + pf_at)
    recall = pt_at/(pt_at + pf_af)
    f1 = 2 * (precision * recall) / (precision + recall )
    print("Raw:{} {} {} {}".format(pt_at,pt_af,pf_at,pf_af))
    print("Actual:{} Precision:{} Recall:{} F1:{} ".format(accuracy,\
                            pt_at/(pt_at + pf_at),\
                            pt_at/(pt_at + pf_af),\
                                f1))