import os
from re import L
import h5py
import numpy as np

from torch.utils.data import DataLoader,Dataset

from config import *

import os
import glob
import numpy as np

import cv2
cv2.setNumThreads(0)

import torch
import torch.utils.data as data_utl

from utils import *

dataDir = "AAU"
hdf5Files = ["training_pointcloud_hdf5", "testing_pointcloud_hdf5"]
dataTypes = ["synthetic", "real"]
partitions = ["Training", "Validation"]

classLabels = {0:"Normal", 1:"Displacement", 2:"Brick", 3:"Rubber Ring"}


class AAUSewer(Dataset):
    def __init__(self,split = "train",type = "real",l=None, dataDir = dataDir):
        super().__init__()
        self.split = split
        self.length = l
        self.train_data = []
        self.labels = []

        path = os.path.join(dataDir, "{}_{}.h5".format("{}ing_pointcloud_hdf5".format(split), type))
        print(path)
        with h5py.File(path, 'r') as hdf:
            if split == "train":
                partitions = ["Training"]
            else:
                partitions = ["Testing"]

            for partition in partitions:
                self.train_data = np.asarray(hdf[f'{partition}/PointClouds'][:])
                self.labels = np.asarray(hdf[f'{partition}/Labels'][:])


    def __len__(self):
      if self.length is not None:return self.length
      return self.labels.shape[0]

    def __getitem__(self,index):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return torch.tensor(self.train_data[index]).to(device),self.labels[index]


data_dir = 'velodyne'

def aug_data(tag, object_dir):
    np.random.seed()
    # rgb = cv2.imread(os.path.join(object_dir,'image_2', tag + '.png'))
    rgb = cv2.resize(cv2.imread(os.path.join(object_dir, 'image_2', tag + '.png')), (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
    lidar = np.fromfile(os.path.join(object_dir, 'velodyne', tag + '.bin'), dtype = np.float32).reshape(-1, 4)
    label = np.array([line for line in open(os.path.join(object_dir, 'label_2', tag + '.txt'), 'r').readlines()])  # (N')

    cls = np.array([line.split()[0] for line in label])  # (N')
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls = '', coordinate = 'camera')[0]  # (N', 7); 7 means (x, y, z, h, w, l, r)

    choice = np.random.randint(0, 10)
    if choice >= 7:
        # Disable this augmention here. Current implementation will decrease the performances.
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_corner_gt_box3d = center_to_corner_box3d(lidar_center_gt_box3d, coordinate = 'lidar')
        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = np.random.normal()
                t_y = np.random.normal()
                t_z = np.random.normal()
                # Check collision
                tmp = box_transform(lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')
                is_collision = False
                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[0, 1, 4, 5, 6]]
                    iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype = np.float32),
                                    np.array([x2, y2, w2, l2, r2], dtype = np.float32))
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = lidar_corner_gt_box3d[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(lidar[bound_box, 0:3], t_x, t_y, t_z, rz = t_rz)
                lidar_center_gt_box3d[idx] = box_transform(lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')

        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_1_{}'.format(tag, np.random.randint(1, 1024))
    elif choice < 7 and choice >= 4:
        # Global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz = angle)
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d = box_transform(lidar_center_gt_box3d, 0, 0, 0, r = angle, coordinate = 'lidar')
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    else:
        # Global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], coordinate = 'camera')[0]  # (N')
    voxel_dict = process_pointcloud(lidar)  # Contains: feature_buffer, number_buffer, coordinate_buffer

    return newtag, rgb, lidar, voxel_dict, label

def process_pointcloud(point_cloud, cls = config.DETECT_OBJ):
    # Input:
    #   (N, 4)
    # Output:
    #   voxel_dict
    if cls == 'Car':
        scene_size = np.array([4, 80, 70.4], dtype = np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype = np.float32)
        grid_size = np.array([10, 400, 352], dtype = np.int64)
        lidar_coord = np.array([0, 40, 3], dtype = np.float32)
        max_point_number = 35
    else:
        scene_size = np.array([4, 40, 48], dtype = np.float32)
        voxel_size = np.array([0.4, 0.2, 0.2], dtype = np.float32)
        grid_size = np.array([10, 200, 240], dtype = np.int64)
        lidar_coord = np.array([0, 20, 3], dtype = np.float32)
        max_point_number = 45

        np.random.shuffle(point_cloud)

    shifted_coord = point_cloud[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    voxel_index = np.floor(
        shifted_coord[:, ::-1] / voxel_size).astype(np.int)

    bound_x = np.logical_and(
        voxel_index[:, 2] >= 0, voxel_index[:, 2] < grid_size[2])
    bound_y = np.logical_and(
        voxel_index[:, 1] >= 0, voxel_index[:, 1] < grid_size[1])
    bound_z = np.logical_and(
        voxel_index[:, 0] >= 0, voxel_index[:, 0] < grid_size[0])

    bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)

    point_cloud = point_cloud[bound_box]
    voxel_index = voxel_index[bound_box]

    # [K, 3] coordinate buffer as described in the paper
    coordinate_buffer = np.unique(voxel_index, axis = 0)

    K = len(coordinate_buffer)
    T = max_point_number

    # [K, 1] store number of points in each voxel grid
    number_buffer = np.zeros(shape = (K), dtype = np.int64)

    # [K, T, 7] feature buffer as described in the paper
    feature_buffer = np.zeros(shape = (K, T, 7), dtype = np.float32)

    # build a reverse index for coordinate buffer
    index_buffer = {}
    for i in range(K):
        index_buffer[tuple(coordinate_buffer[i])] = i

    for voxel, point in zip(voxel_index, point_cloud):
        index = index_buffer[tuple(voxel)]
        number = number_buffer[index]
        if number < T:
            feature_buffer[index, number, :4] = point
            number_buffer[index] += 1

    feature_buffer[:, :, -3:] = feature_buffer[:, :, :3] - \
        feature_buffer[:, :, :3].sum(axis = 1, keepdims = True)/number_buffer.reshape(K, 1, 1)

    voxel_dict = {'feature_buffer': feature_buffer,
                  'coordinate_buffer': coordinate_buffer,
                  'number_buffer': number_buffer}

    return voxel_dict


class Processor:
    def __init__(self, data_tag, f_rgb, f_lidar, f_label, data_dir, aug, is_testset):
        self.data_tag = data_tag
        self.f_rgb = f_rgb
        self.f_lidar = f_lidar
        self.f_label = f_label
        self.data_dir = data_dir
        self.aug = aug
        self.is_testset = is_testset


    def __call__(self, load_index):
        if self.aug:
            ret = aug_data(self.data_tag[load_index], self.data_dir)
        else:
            rgb = cv2.resize(cv2.imread(self.f_rgb[load_index]), (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            raw_lidar = np.fromfile(self.f_lidar[load_index], dtype = np.float32).reshape((-1, 4))
            if not self.is_testset:
                labels = [line for line in open(self.f_label[load_index], 'r').readlines()]
            else:
                labels = ['']
            tag = self.data_tag[load_index]
            voxel = process_pointcloud(raw_lidar)
            ret = [tag, rgb, raw_lidar, voxel, labels]

        return ret


class KITTI(data_utl.Dataset):
    def __init__(self, data_dir, shuffle = False, aug = False, is_testset = False):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.aug = aug
        self.is_testset = is_testset

        self.f_rgb = glob.glob(os.path.join(self.data_dir, 'image_2', '*.png'))
        self.f_lidar = glob.glob(os.path.join(self.data_dir, 'velodyne', '*.bin'))
        self.f_label = glob.glob(os.path.join(self.data_dir, 'label_2', '*.txt'))

        self.f_rgb.sort()
        self.f_lidar.sort()
        self.f_label.sort()

        self.data_tag = [name.split('/')[-1].split('.')[-2] for name in self.f_rgb]

        assert len(self.data_tag) != 0, 'Dataset folder is not correct!'
        assert len(self.data_tag) == len(self.f_rgb) == len(self.f_lidar), 'Dataset folder is not correct!'

        nums = len(self.f_rgb)
        self.indices = list(range(nums))
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Build a data processor
        self.proc = Processor(self.data_tag, self.f_rgb, self.f_lidar, self.f_label, self.data_dir, self.aug, self.is_testset)


    def __getitem__(self, index):
        # A list of [tag, rgb, raw_lidar, voxel, labels]
        ret = self.proc(self.indices[index])

        return ret


    def __len__(self):
        return len(self.indices)


def collate_fn(rets):
    tag = [ret[0] for ret in rets]
    rgb = [ret[1] for ret in rets]
    raw_lidar = [ret[2] for ret in rets]
    voxel = [ret[3] for ret in rets]
    labels = [ret[4] for ret in rets]

    # Only for voxel
    _, vox_feature, vox_number, vox_coordinate = build_input(voxel)

    res = (
        np.array(tag),
        np.array(labels),
        [torch.from_numpy(x) for x in vox_feature],
        np.array(vox_number),
        [torch.from_numpy(x) for x in vox_coordinate],
        np.array(rgb),
        np.array(raw_lidar)
    )

    return res


def build_input(voxel_dict_list):
    batch_size = len(voxel_dict_list)

    feature_list = []
    number_list = []
    coordinate_list = []
    for i, voxel_dict in zip(range(batch_size), voxel_dict_list):
        feature_list.append(voxel_dict['feature_buffer'])   # (K, T, 7); K is max number of non-empty voxels; T = 35
        number_list.append(voxel_dict['number_buffer'])     # (K,)
        coordinate = voxel_dict['coordinate_buffer']        # (K, 3)
        coordinate_list.append(np.pad(coordinate, ((0, 0), (1, 0)), mode = 'constant', constant_values = i))

    # feature = np.concatenate(feature_list)
    # number = np.concatenate(number_list)
    # coordinate = np.concatenate(coordinate_list)
    #
    # return batch_size, feature, number, coordinate

    return batch_size, feature_list, number_list, coordinate_list