import mmcv
import numpy as np
from mmdet.datasets.builder import PIPELINES
from nuscenes.utils.data_classes import RadarPointCloud
from mmdet3d.core.points import get_points_type

@PIPELINES.register_module()
class LoadRadarPointsFromMultiSweepsV3(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int): Number of sweeps. Defaults to 10.
        load_dim (int): Dimension number of the loaded points. Defaults to 5.
        use_dim (list[int]): Which dimension to use. Defaults to [0, 1, 2, 4].
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool): Whether to remove close points.
            Defaults to False.
        test_mode (bool): If test_model=True used for testing, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """


    def __init__(self,
                 sweeps_num=10,
                 file_client_args=dict(backend='disk'),
                 invalid_states = [0],
                 dynprop_states = range(7),
                 ambig_states = [3],
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.sweeps_num = sweeps_num
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.invalid_states = invalid_states
        self.dynprop_states = dynprop_states
        self.ambig_states = ambig_states
        self.coord_type = 'LIDAR'


    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            # x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
            raw_points = RadarPointCloud.from_file(
                pts_filename,
                invalid_states=self.invalid_states,
                dynprop_states=self.dynprop_states,
                ambig_states=self.ambig_states).points.T
            xyz = raw_points[:, :3]
            rcs = raw_points[:, 5].reshape(-1, 1)
            vxy_comp = raw_points[:, 8:10]
            points = np.hstack((xyz, rcs, vxy_comp))
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points


    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud \
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data. \
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point \
                    cloud arrays.
        """
        sweep_points_list = []
        if self.pad_empty_sweeps and len(results['radar_sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            num_sweeps = len(results['radar_sweeps']) # sample과 sweep radar가 모두 들어있음
            for idx in range(num_sweeps):
                sweep_pts_file_name = results['radar_sweeps'][idx]
                points_sweep = self._load_points(sweep_pts_file_name) # x, y, z, RCS, vx_comp, vy_comp
                # print(points_sweep.shape)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                points_sweep[:, :2] += points_sweep[:, 4:6] * results['radar_sweeps_time_gap'][idx] ################ radial velocity
                
                R = results['radar_sweeps_r2l_rot'][idx]
                T = results['radar_sweeps_r2l_trans'][idx]
                ### create sensor2lidar_rot/tran end ###
                
                points_sweep[:, :3] = points_sweep[:, :3] @ R
                points_sweep[:, 4:6] = points_sweep[:, 4:6] @ R[:2, :2]
                points_sweep[:, :3] += T # lidar coordinate radar point clouds
                sweep_points_list.append(points_sweep)
        
        points = np.concatenate(sweep_points_list)
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=None)
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'