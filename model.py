import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from utils  import *
import pdb

import os

cfg = config

class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = int(out_channels / 2)

        self.dense = nn.Sequential(nn.Linear(self.in_channels, self.units), nn.ReLU())
        self.batch_norm = nn.BatchNorm1d(self.units)


    def forward(self, inputs, mask):
        # [ΣK, T, in_ch] -> [ΣK, T, units] -> [ΣK, units, T]
        tmp = self.dense(inputs).transpose(1, 2)
        # [ΣK, units, T] -> [ΣK, T, units]
        pointwise = self.batch_norm(tmp).transpose(1, 2)

        # [ΣK, 1, units]
        aggregated, _ = torch.max(pointwise, dim = 1, keepdim = True)

        # [ΣK, T, units]
        repeated = aggregated.expand(-1, cfg.VOXEL_POINT_COUNT, -1)

        # [ΣK, T, 2 * units]
        concatenated = torch.cat([pointwise, repeated], dim = 2)

        # [ΣK, T, 1] -> [ΣK, T, 2 * units]
        mask = mask.expand(-1, -1, 2 * self.units)

        concatenated = concatenated * mask.float()

        return concatenated


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.vfe1 = VFELayer(7, 32)
        self.vfe2 = VFELayer(32, 128)


    def forward(self, feature, number, coordinate):

        batch_size = len(feature)

        feature = torch.cat(feature, dim = 0)   # [ΣK, cfg.VOXEL_POINT_COUNT, 7]; cfg.VOXEL_POINT_COUNT = 35/45
        coordinate = torch.cat(coordinate, dim = 0)     # [ΣK, 4]; each row stores (batch, d, h, w)

        vmax, _ = torch.max(feature, dim = 2, keepdim = True)
        mask = (vmax != 0)  # [ΣK, T, 1]

        x = self.vfe1(feature, mask)
        x = self.vfe2(x, mask)

        # [ΣK, 128]
        voxelwise, _ = torch.max(x, dim = 1)

        # Car: [B, 10, 400, 352, 128]; Pedestrain/Cyclist: [B, 10, 200, 240, 128]
        #print(cfg.INPUT_DEPTH, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH)
        #print(coordinate.t().shape,voxelwise.shape)
        outputs = torch.sparse.FloatTensor(coordinate.t(), voxelwise,torch.Size([batch_size, cfg.INPUT_DEPTH, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))

        #print(outputs.shape)
        outputs = outputs.to_dense()
        outputs = outputs.reshape( torch.Size([batch_size, cfg.INPUT_DEPTH, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))

        return outputs


class ConvMD(nn.Module):
    def __init__(self, M, cin, cout, k, s, p, bn = True, activation = True):
        super(ConvMD, self).__init__()

        self.M = M  # Dimension of input
        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn
        self.activation = activation

        if self.M == 2:     # 2D input
            self.conv = nn.Conv2d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm2d(self.cout)
        elif self.M == 3:   # 3D input
            self.conv = nn.Conv3d(self.cin, self.cout, self.k, self.s, self.p)
            if self.bn:
                self.batch_norm = nn.BatchNorm3d(self.cout)
        else:
            raise Exception('No such mode!')


    def forward(self, inputs):

        out = self.conv(inputs)

        if self.bn:
            out = self.batch_norm(out)

        if self.activation:
            return F.relu(out)
        else:
            return out


class Deconv2D(nn.Module):
    def __init__(self, cin, cout, k, s, p, bn = True):
        super(Deconv2D, self).__init__()

        self.cin = cin
        self.cout = cout
        self.k = k
        self.s = s
        self.p = p
        self.bn = bn

        self.deconv = nn.ConvTranspose2d(self.cin, self.cout, self.k, self.s, self.p)

        if self.bn:
            self.batch_norm = nn.BatchNorm2d(self.cout)


    def forward(self, inputs):
        out = self.deconv(inputs)

        if self.bn == True:
            out = self.batch_norm(out)

        return F.relu(out)


class MiddleAndRPN(nn.Module):
    def __init__(self, alpha = 1.5, beta = 1, sigma = 3, training = True, name = ''):
        super(MiddleAndRPN, self).__init__()

        self.middle_layer = nn.Sequential(ConvMD(3, 128, 64, 3, (2, 1, 1,), (1, 1, 1)),
                                          ConvMD(3, 64, 64, 3, (1, 1, 1), (0, 1, 1)),
                                          ConvMD(3, 64, 64, 3, (2, 1, 1), (1, 1, 1)))


        if cfg.DETECT_OBJ == 'Car':
            self.block1 = nn.Sequential(ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))
        else:   # Pedestrian/Cyclist
            self.block1 = nn.Sequential(ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                        ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))

        self.deconv1 = Deconv2D(128, 256, 3, (1, 1), (1, 1))

        self.block2 = nn.Sequential(ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 128, 128, 3, (1, 1), (1, 1)))

        self.deconv2 = Deconv2D(128, 256, 2, (2, 2), (0, 0))

        self.block3 = nn.Sequential(ConvMD(2, 128, 256, 3, (2, 2), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
                                    ConvMD(2, 256, 256, 3, (1, 1), (1, 1)))

        self.deconv3 = Deconv2D(256, 256, 4, (4, 4), (0, 0))

        self.prob_conv = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), bn = False, activation = False)

        self.reg_conv = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), bn = False, activation = False)

        self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]


    def forward(self, inputs):

        batch_size, DEPTH, HEIGHT, WIDTH, C = inputs.shape  # [batch_size, 10, 400/200, 352/240, 128]

        inputs = inputs.permute(0, 4, 1, 2, 3)  # (B, D, H, W, C) -> (B, C, D, H, W)

        temp_conv = self.middle_layer(inputs)   # [batch, 64, 2, 400, 352]
        temp_conv = temp_conv.reshape(batch_size, -1, HEIGHT, WIDTH)   # [batch, 128, 400, 352]

        temp_conv = self.block1(temp_conv)      # [batch, 128, 200, 176]
        temp_deconv1 = self.deconv1(temp_conv)

        temp_conv = self.block2(temp_conv)      # [batch, 128, 100, 88]
        temp_deconv2 = self.deconv2(temp_conv)

        temp_conv = self.block3(temp_conv)      # [batch, 256, 50, 44]
        temp_deconv3 = self.deconv3(temp_conv)

        temp_conv = torch.cat([temp_deconv3, temp_deconv2, temp_deconv1], dim = 1)

        # Probability score map, [batch, 2, 200/100, 176/120]
        p_map = self.prob_conv(temp_conv)

        # Regression map, [batch, 14, 200/100, 176/120]
        r_map = self.reg_conv(temp_conv)

        return torch.sigmoid(p_map), r_map

small_addon_for_BCE = 1e-6


class RPN3D(nn.Module):
    def __init__(self, cls = 'Car', alpha = 1.5, beta = 1, sigma = 3):
        super(RPN3D, self).__init__()

        self.cls = cls
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma

        self.feature = FeatureNet()
        self.rpn = MiddleAndRPN()

        # Generate anchors
        self.anchors = cal_anchors()    # [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2, 7]; 2 means two rotations; 7 means (cx, cy, cz, h, w, l, r)

        self.rpn_output_shape = self.rpn.output_shape


    def forward(self, data):
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        #print(tag)
        #print(label)

        #print(vox_feature[0].shape)
        #print(vox_number[0].shape)
        #print(vox_coordinate[0].shape)

        features = self.feature(vox_feature, vox_number, vox_coordinate)

        prob_output, delta_output = self.rpn(features)
        #print(prob_output.shape,delta_output.shape)
        # Calculate ground-truth
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors, cls = cfg.DETECT_OBJ, coordinate = 'lidar')
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis = -1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis = (1, 2, 3)).reshape(-1, 1, 1, 1), a_min = 1, a_max = None)
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis = (1, 2, 3)).reshape(-1, 1, 1, 1), a_min = 1, a_max = None)

        # Move to gpu
        device = features.device
        pos_equal_one = torch.from_numpy(pos_equal_one).to(device).float()
        neg_equal_one = torch.from_numpy(neg_equal_one).to(device).float()

        targets = torch.from_numpy(targets).to(device).float()

        pos_equal_one_for_reg = torch.from_numpy(pos_equal_one_for_reg).to(device).float()
        pos_equal_one_sum = torch.from_numpy(pos_equal_one_sum).to(device).float()
        neg_equal_one_sum = torch.from_numpy(neg_equal_one_sum).to(device).float()

        # [batch, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2/14] -> [batch, 2/14, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]
        pos_equal_one = pos_equal_one.permute(0, 3, 1, 2)
        neg_equal_one = neg_equal_one.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2)
        pos_equal_one_for_reg = pos_equal_one_for_reg.permute(0, 3, 1, 2)

        # Calculate loss
        cls_pos_loss = (-pos_equal_one * torch.log(prob_output + small_addon_for_BCE)) / pos_equal_one_sum
        cls_neg_loss = (-neg_equal_one * torch.log(1 - prob_output + small_addon_for_BCE)) / neg_equal_one_sum

        cls_loss = torch.sum(self.alpha * cls_pos_loss + self.beta * cls_neg_loss)
        cls_pos_loss_rec = torch.sum(cls_pos_loss)
        cls_neg_loss_rec = torch.sum(cls_neg_loss)

        reg_loss = smooth_l1(delta_output * pos_equal_one_for_reg, targets * pos_equal_one_for_reg, self.sigma) / pos_equal_one_sum
        reg_loss = torch.sum(reg_loss)

        loss = cls_loss + reg_loss

        return prob_output, delta_output, loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec


    def predict(self, data, probs, deltas, summary = False, vis = False):
        '''
        probs: (batch, 2, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH)
        deltas: (batch, 14, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH)
        '''
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        img = data[5]
        lidar = data[6]

        batch_size, _, _, _ = probs.shape
        device = probs.device

        batch_gt_boxes3d = None
        if summary or vis:
            batch_gt_boxes3d = label_to_gt_box3d(label, cls = self.cls, coordinate = 'lidar')

        # Move to cpu and convert to numpy array
        probs = probs.cpu().detach().numpy()
        deltas = deltas.cpu().detach().numpy()

        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate = 'lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0, 1, 4, 5, 6]]
        batch_probs = probs.reshape((batch_size, -1))

        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(batch_size):
            # Remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(center_to_corner_box2d(tmp_boxes2d, coordinate = 'lidar'))

            # 2D box index after nms
            ind, cnt = nms(torch.from_numpy(boxes2d).to(device), torch.from_numpy(tmp_scores).to(device),
                           cfg.RPN_NMS_THRESH, cfg.RPN_NMS_POST_TOPK)
            ind = ind[:cnt].cpu().detach().numpy()

            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis = -1))

        if summary:
            # Only summry the first one in a batch
            cur_tag = tag[0]
            P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))

            front_image = draw_lidar_box3d_on_image(img[0], ret_box3d[0], ret_score[0],
                                                    batch_gt_boxes3d[0], P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

            bird_view = lidar_to_bird_view_img(lidar[0], factor = cfg.BV_LOG_FACTOR)

            bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[0], ret_score[0], batch_gt_boxes3d[0],
                                                     factor = cfg.BV_LOG_FACTOR, P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

            heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)

            ret_summary = [['predict/front_view_rgb', front_image[np.newaxis, ...]],  # [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3]
                           # [None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3]
                           ['predict/bird_view_lidar', bird_view[np.newaxis, ...]],
                           # [None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3]
                           ['predict/bird_view_heatmap', heatmap[np.newaxis, ...]]]

            return tag, ret_box3d_score, ret_summary

        if vis:
            front_images, bird_views, heatmaps = [], [], []
            for i in range(len(img)):
                cur_tag = tag[i]
                P, Tr, R = load_calib(os.path.join(cfg.CALIB_DIR, cur_tag + '.txt'))

                front_image = draw_lidar_box3d_on_image(img[i], ret_box3d[i], ret_score[i],
                                                        batch_gt_boxes3d[i], P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

                bird_view = lidar_to_bird_view_img(lidar[i], factor = cfg.BV_LOG_FACTOR)

                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[i], ret_score[i], batch_gt_boxes3d[i],
                                                         factor = cfg.BV_LOG_FACTOR, P2 = P, T_VELO_2_CAM = Tr, R_RECT_0 = R)

                heatmap = colorize(probs[i, ...], cfg.BV_LOG_FACTOR)

                front_images.append(front_image)
                bird_views.append(bird_view)
                heatmaps.append(heatmap)

            return tag, ret_box3d_score, front_images, bird_views, heatmaps

        return tag, ret_box3d_score


def smooth_l1(deltas, targets, sigma = 3.0):
    # Reference: https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0 / sigma2).float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1