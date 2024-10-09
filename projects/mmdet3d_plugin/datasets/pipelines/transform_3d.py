import numpy as np
from numpy import random
import mmcv
from mmcv.utils import build_from_cfg
from ..registry import OBJECTSAMPLERS
from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import RandomFlip
# from mmdet3d.core import project_pts_on_img_save
import torch
from PIL import Image
import cv2
from pyquaternion import Quaternion

try:
    from nuscenes.utils.geometry_utils import transform_matrix
except:
    print("nuScenes devkit not Found!")

@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(
                img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(
                img, self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        
        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb


    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = [mmcv.imnormalize(img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype np.float32,'\
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta,
                                    self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(self.saturation_lower,
                                            self.saturation_upper)

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower,
                                        self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str



@PIPELINES.register_module()
class CustomCollect3D(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".
    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:
        - 'img_shape': shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is \
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is \
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.
    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx', 'prev_idx', 'next_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow', 'scene_token',
                            'can_bus',
                            )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.
        Args:
            results (dict): Result dict contains the data to collect.
        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
       
        data = {}
        img_metas = {}
      
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'



@PIPELINES.register_module()
class RandomScaleImageMultiViewImage(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==1

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_ind = np.random.permutation(range(len(self.scales)))[0]
        rand_scale = self.scales[rand_ind]

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['img_shape'] = ([img.shape for img in results['img']])
        results['ori_shape'] = [img.shape for img in results['img']]

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str

@PIPELINES.register_module()
class RandomScaleImageMultiViewImageCus(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales)==2

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_scale_y = self.scales[0]
        rand_scale_x = self.scales[1]

        y_size = [int(img.shape[0] * rand_scale_y) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale_x) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale_y
        scale_factor[1, 1] *= rand_scale_x
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        results['img_scale'] = scale_factor
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        H, W, C = results['img'][0].shape
        num_cam = len(results['img'])
        results['lidar2img'] = lidar2img
        results['img_shape'] = (H, W, C, num_cam)
        results['ori_shape'] = (H, W, C, num_cam)

        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str


@PIPELINES.register_module()
class MultCamImageAugmentation(object):
    """ MultiModalBEVAUgmentation (BEVDet)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
    """

    def __init__(self,ida_aug_conf):
        self.ida_aug_conf = ida_aug_conf

    def sample_augmentation(self, H, W):
        resize = 1 + np.random.uniform(*self.ida_aug_conf['resize'])
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*self.ida_aug_conf['crop_h'])) *
                        newH) - H
        crop_w = int(np.random.uniform(0, max(0, newW - W)))
        crop = (crop_w, crop_h, crop_w + W, crop_h + H)
        flip = self.ida_aug_conf['flip'] and np.random.choice([0, 1])
        rotate = np.random.uniform(*self.ida_aug_conf['rot'])

        return resize, resize_dims, crop, flip, rotate
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform_v2(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        resize_mat = np.eye(4)
        flip_mat = np.eye(4)
        rot_mat1 = np.eye(4)
        rot_mat2 = np.eye(4)
        rot_mat3 = np.eye(4)
        
        resize_mat[:2,:2] = resize * np.eye(2)
        resize_mat[:2,2] = -np.array(crop[:2])
        if flip:
            flip_mat[0,0] = -1
            flip_mat[:2,2] = np.array([crop[2] - crop[0],0])
        rot_mat1[:2,2] = -np.array((crop[2] - crop[0],crop[3]-crop[1]))/2
        rot_mat2[:2,:2] = self.get_rot(rotate / 180 * np.pi)
        rot_mat3[:2,2] = np.array((crop[2] - crop[0],crop[3]-crop[1]))/2

        post_mat = rot_mat3@rot_mat2@rot_mat1@flip_mat@resize_mat
        return img, post_mat

    def __call__(self, input_dict):
        H,W,_,_ = input_dict['img_shape'] # (H, W, 3, num_cam)
        img_list = input_dict['img']
        input_dict['img_ori'] = input_dict['img'].copy()
        input_dict['cam_intrinsic_ori'] = input_dict['cam_intrinsic'].copy()
        cam = input_dict['cam_intrinsic'].copy()
        l2c = input_dict['lidar2cam'].copy()
        post_rots = []
        img_transforms_ = []
        new_img = []

        for idx, img in enumerate(img_list):
            Img = Image.fromarray(img.astype(np.uint8))
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                H=Img.height, W=Img.width)
            img_transforms_.append([resize_dims, crop, flip, rotate])
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            img, post_rot_ = \
                self.img_transform_v2(Img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)
            new_img.append(np.array(img).astype(np.float32))
            post_rots.append(post_rot_)
            new_cam = post_rot_@cam[idx]
            input_dict['cam_intrinsic'][idx] = new_cam
            if 'img_scale' in input_dict:
                img_scale = input_dict['img_scale']
                input_dict['lidar2img'][idx] = img_scale@new_cam@l2c[idx]
            else:
                input_dict['lidar2img'][idx] = new_cam@l2c[idx]
        input_dict['img'] = new_img 
 
        return input_dict


@PIPELINES.register_module()
class MultiModalBEVAugmentation(object):
    """ MultiModalBEVAUgmentation (BEVDet)

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
    """

    def __init__(self,bda_aug_conf):
        self.bda_aug_conf = bda_aug_conf

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def __call__(self, input_dict):
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation()
        
        angle = torch.tensor(rotate_bda)
        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],[0, 0, 1]])
        scale_mat = torch.Tensor([[scale_bda, 0, 0], [0, scale_bda, 0],
                                  [0, 0, scale_bda]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            flip_dx = input_dict['pcd_vertical_flip']
            flip_dy = input_dict['pcd_horizontal_flip']
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1

        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat_1 = flip_mat @ (scale_mat @ rot_mat)
    
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        bda_mat[:3, :3] = rot_mat_1

        if flip_dx:
            bev_direction = 'vertical'
            for key in input_dict['bbox3d_fields']:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict[key].flip(
                        bev_direction, points=input_dict['points'])
                else:
                    input_dict[key].flip(bev_direction)

        if flip_dy:
            bev_direction = 'horizontal'
            for key in input_dict['bbox3d_fields']:
                if 'points' in input_dict:
                    input_dict['points'] = input_dict[key].flip(
                        bev_direction, points=input_dict['points'])
                else:
                    input_dict[key].flip(bev_direction)

        points = input_dict['points']
        points.scale(scale_bda)
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale_bda)

        if len(input_dict['bbox3d_fields']) == 0:
            input_dict['points'].rotate(rot_mat)
            input_dict['pcd_rotation'] =rot_mat
            return

        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    rot_mat, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat
        
        l2c = np.array(input_dict['lidar2cam'].copy())
        cam = input_dict['cam_intrinsic'].copy()

        l2c_ = l2c @ bda_mat.numpy()
        if 'img_scale' in input_dict:
            img_scale = input_dict['img_scale']
            input_dict['lidar2img'] = [img_scale@cam[0]@l2c_[0],img_scale@cam[1]@l2c_[1],img_scale@cam[2]@l2c_[2],img_scale@cam[3]@l2c_[3],img_scale@cam[4]@l2c_[4],img_scale@cam[5]@l2c_[5]]
        else:
            input_dict['lidar2img'] = [cam[0]@l2c_[0],cam[1]@l2c_[1],cam[2]@l2c_[2],cam[3]@l2c_[3],cam[4]@l2c_[4],cam[5]@l2c_[5]]

        return input_dict 


@PIPELINES.register_module()
class ObjectSample_V2(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
    """

    def __init__(self, db_sampler, sample_2d=False, with_info=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.with_info = with_info
        self.keep_raw = True
        self.sample_method = 'by_order'

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points


    @staticmethod
    def remove_polar_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        polar_pts = points.polar_coord
        masks = box_np_ops.pts_in_polar_coord_rbbox(polar_pts, boxes,1)
        points = points[np.logical_not(masks.any(-1))]
        return points
    
    
    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        if self.with_info:
            data_info = input_dict["data_info"]
            cam_images = input_dict['img']
            cam_orders = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',  
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        else:
            data_info, cam_images = None, None

        num_gt = len(gt_bboxes_3d)
        crop_img_list = [[np.zeros((1,1,1))] * num_gt for _ in range(len(cam_orders))]
        if self.with_info:
            sample_record = data_info.get('sample', input_dict['sample_idx'])
            pointsensor_token = sample_record['data']['LIDAR_TOP']
        if self.with_info and num_gt:
            # Transform points
            sample_coords = input_dict["gt_bboxes_3d"].corners
            #  sample_coords = box_np_ops.rbbox3d_to_corners(gt_bboxes_3d.tensor.cpu().numpy())
            #  crop_img_list = [[] for _ in range(len(sample_coords))]
            #  crop_img_list = [[np.zeros((1,1,1))] * len(sample_coords) for _ in range(len(cam_orders))]
            # Crop images from raw images
            for _idx, _key in enumerate(cam_orders):
                cam_key = _key.upper()
                camera_token = sample_record['data'][cam_key]
                cam = data_info.get('sample_data', camera_token)
                lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
                points_cam = (points_3d @ lidar2cam.T).T
                # Filter useless boxes according to depth
                cam_mask = (points_cam[2] > 0).all(axis=0)
                cam_count = cam_mask.nonzero()[0]
                if cam_mask.sum() == 0:
                    continue
                points_cam = points_cam[...,cam_mask].reshape(4, -1)
                point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                point_img = point_img.reshape(3, 8, -1)
                point_img = point_img.transpose()[...,:2]
                minxy = np.min(point_img, axis=-2)
                maxxy = np.max(point_img, axis=-2)
                bbox = np.concatenate([minxy, maxxy], axis=-1)
                #  bbox = (bbox * res['image_scale']).astype(np.int32)
                bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=input_dict['img_shape'][1]-1)
                bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=input_dict['img_shape'][0]-1)
                cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
                if cam_mask.sum() == 0:
                    continue
                cam_count = cam_count[cam_mask]
                bbox = bbox[cam_mask]

                for __idx, _box in enumerate(bbox.astype(np.int)):
                    #  crop_img_list[__idx] = cam_images[_idx][_box[1]:_box[3],_box[0]:_box[2]]
                    crop_img_list[_idx][cam_count[__idx]] = cam_images[_idx][_box[1]:_box[3],_box[0]:_box[2]]

        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            # Assume for now 3D & 2D bboxes are the same
            #  sampled_dict = self.db_sampler.sample_all(
                #  gt_bboxes_3d.tensor.numpy(),
                #  gt_labels_3d,
                #  img=img,
                #  data_info=data_info
                #  )
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                img=img,
                data_info=data_info,
                sample=gt_bboxes_3d
                )
        else:
            #  sampled_dict = self.db_sampler.sample_all(
                #  gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None)
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(), gt_labels_3d, img=None, sample=gt_bboxes_3d)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            # for changed pkl
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            # breakpoint()
            if sampled_dict['dbinfo_name'] == 'polar':
                points = self.remove_polar_points_in_boxes(points, sampled_gt_bboxes_3d)
            else:
                points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.with_info:
                # Transform points
                sample_coords = gt_bboxes_3d.corners
                #  sample_coords = box_np_ops.rbbox3d_to_corners(sampled_dict['gt_bboxes_3d'])
                raw_gt_box_num = len(sampled_dict["gt_bboxes_3d"])
                #  sample_crops = crop_img_list + sampled_dict['img_crops']
                sample_crops = []
                for cam_order in range(len(cam_orders)):
                    sample_crops.append(crop_img_list[cam_order] + sampled_dict['img_crops'])
                if not self.keep_raw:
                    points_coords = points[:,:4].copy()
                    points_coords[:,-1] = 1
                    point_keep_mask = np.ones(len(points_coords)).astype(np.bool)
                # Paste image according to sorted strategy
                for _idx, _key in enumerate(cam_orders):
                    cam_key = _key.upper()
                    camera_token = sample_record['data'][cam_key]
                    cam = data_info.get('sample_data', camera_token)
                    lidar2cam, cam_intrinsic = get_lidar2cam_matrix(data_info, pointsensor_token, cam)
                    points_3d = np.concatenate([sample_coords, np.ones((*sample_coords.shape[:2], 1))], axis=-1)
                    points_cam = (points_3d @ lidar2cam.T).T
                    depth = points_cam[2]
                    cam_mask = (depth > 0).all(axis=0)
                    cam_count = cam_mask.nonzero()[0]
                    if cam_mask.sum() == 0:
                        continue
                    depth = depth.mean(0)[cam_mask]
                    points_cam = points_cam[...,cam_mask].reshape(4, -1)
                    point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                    point_img = point_img.reshape(3, 8, -1)
                    point_img = point_img.transpose()[...,:2]
                    minxy = np.min(point_img, axis=-2)
                    maxxy = np.max(point_img, axis=-2)
                    bbox = np.concatenate([minxy, maxxy], axis=-1)
                    #  bbox = (bbox * res['image_scale']).astype(np.int32)
                    bbox[:,0::2] = np.clip(bbox[:,0::2], a_min=0, a_max=input_dict['img_shape'][1]-1)
                    bbox[:,1::2] = np.clip(bbox[:,1::2], a_min=0, a_max=input_dict['img_shape'][0]-1)
                    cam_mask = (bbox[:,2]-bbox[:,0])*(bbox[:,3]-bbox[:,1])>0
                    if cam_mask.sum() == 0:
                        continue
                    depth = depth[cam_mask]
                    if 'depth' in self.sample_method:
                        paste_order = depth.argsort()
                        paste_order = paste_order[::-1]
                    else:
                        paste_order = np.arange(len(depth), dtype=np.int64)
                    cam_count = cam_count[cam_mask][paste_order]
                    bbox = bbox[cam_mask][paste_order]

                    paste_mask = -255 * np.ones(input_dict['img_shape'][:2], dtype=np.int64)
                    fg_mask = np.zeros(input_dict['img_shape'][:2], dtype=np.int64)
                    for _count, _box in zip(cam_count, bbox.astype(np.int)):
                        img_crop = sample_crops[_idx][_count]
                        #  img_crop = sample_crops[_count]
                        if len(img_crop) == 0: continue
                        if img_crop.shape[0] == 0 or img_crop.shape[1] == 0: continue
                        if _box[2] - _box[0] < 1 or _box[3] - _box[1] < 0: continue
                        img_crop = cv2.resize(img_crop, tuple(_box[[2,3]]-_box[[0,1]]))
                        cam_images[_idx][_box[1]:_box[3],_box[0]:_box[2]] = img_crop
                        paste_mask[_box[1]:_box[3],_box[0]:_box[2]] = _count
                        # foreground area of original point cloud in image plane
                        if _count < raw_gt_box_num:
                            fg_mask[_box[1]:_box[3],_box[0]:_box[2]] = 1
                    # calculate modify mask
                    if not self.keep_raw:
                        points_cam = (points_coords @ lidar2cam.T).T
                        cam_mask = points_cam[2] > 0
                        if cam_mask.sum() == 0:
                            continue
                        point_img = view_points(points_cam[:3, :], np.array(cam_intrinsic), normalize=True)
                        point_img = point_img.transpose()[...,:2]
                        #  point_img = (point_img * res['image_scale']).astype(np.int64)
                        cam_mask = (point_img[:,0] > 0) & (point_img[:,0] < input_dict['img_shape'][1]) & \
                                   (point_img[:,1] > 0) & (point_img[:,1] < input_dict['img_shape'][0]) & cam_mask
                        point_img = point_img[cam_mask]
                        new_mask = paste_mask[point_img[:,1], point_img[:,0]]==(points_idx[cam_mask]+raw_gt_box_num)
                        raw_fg = (fg_mask == 1) & (paste_mask >= 0) & (paste_mask < raw_gt_box_num)
                        raw_bg = (fg_mask == 0) & (paste_mask < 0)
                        raw_mask = raw_fg[point_img[:,1], point_img[:,0]] | raw_bg[point_img[:,1], point_img[:,0]]
                        keep_mask = new_mask | raw_mask
                        point_keep_mask[cam_mask] = point_keep_mask[cam_mask] & keep_mask

                # Replace the original images
                input_dict['img'] = cam_images
                # remove overlaped LIDAR points
                if not self.keep_raw:
                    points = points[point_keep_mask]
            if self.sample_2d and False:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.long)
        input_dict['points'] = points

        pts = input_dict['points']

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str

def get_lidar2cam_matrix(nusc, sensor_token, cam):
    # Get nuScenes lidar2cam and cam_intrinsic matrix
    pointsensor = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    ego_from_lidar = transform_matrix(
        cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=False
    )

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    global_from_ego = transform_matrix(
        poserecord["translation"], Quaternion(poserecord["rotation"]), inverse=False
    )

    lidar2global = np.dot(global_from_ego, ego_from_lidar)

    # Transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])

    ego_from_global = transform_matrix(
        poserecord["translation"], Quaternion(poserecord["rotation"]), inverse=True
    )
    
    # Transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    cam_from_ego = transform_matrix(
        cs_record["translation"], Quaternion(cs_record["rotation"]), inverse=True
    )
    global2cam = np.dot(cam_from_ego, ego_from_global)
    lidar2cam = np.dot(global2cam, lidar2global)
    cam_intrinsic = cs_record['camera_intrinsic']

    return lidar2cam, cam_intrinsic

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    # copied from https://github.com/nutonomy/nuscenes-devkit/
    # only for debug use
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


def lidar2cam_flip_v3(l2c_r_ ,cam_f_t):
    from scipy.spatial.transform import Rotation as R
    # breakpoint()
    lidar2cam_rt_flip = np.tile(np.eye(4),(6,1,1))
    lidar2cam_r_flip  = np.tile(np.eye(3),(6,1,1))
    quat = R.from_matrix(l2c_r_).as_quat()
    quat[:,1:3] = -quat[:,1:3]
    mat = R.from_quat(quat).as_matrix()

    # breakpoint()
    for idx in range(len(mat)):
        if idx%3 ==0:
            lidar2cam_rt_flip[idx][:3, :3] = mat[idx].T
            lidar2cam_r_flip[idx] = mat[idx].T
            lidar2cam_rt_flip[idx][3, :3] = -cam_f_t[idx]@mat[idx].T
        elif idx%3==1:
            lidar2cam_rt_flip[idx][:3, :3] = mat[idx+1].T
            lidar2cam_r_flip[idx] = mat[idx+1].T
            lidar2cam_rt_flip[idx][3, :3] = -cam_f_t[idx+1]@mat[idx+1].T
        elif idx%3==2:
            lidar2cam_rt_flip[idx][:3, :3] = mat[idx-1].T
            lidar2cam_r_flip[idx] = mat[idx-1].T
            lidar2cam_rt_flip[idx][3, :3] = -cam_f_t[idx-1]@mat[idx-1].T
        lidar2cam_rt_flip[idx] = lidar2cam_rt_flip[idx].T
        lidar2cam_r_flip[idx] = lidar2cam_r_flip[idx].T
    return lidar2cam_rt_flip, lidar2cam_r_flip
