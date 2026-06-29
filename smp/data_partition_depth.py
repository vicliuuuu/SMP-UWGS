import os
import math
import copy
import pickle
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from smp.geom import CameraInfo, storePly
from smp.geom import BasicPointCloud
from smp.graham_scan import run_graham_scan
from smp.data_partition import ProgressiveDataPartitioning as BasePartitioning

@dataclass
class CameraPartitionMutable:
    partition_id: str
    cameras: list
    point_cloud: BasicPointCloud
    ori_camera_bbox: list
    extend_camera_bbox: list
    extend_rate: float
    ori_point_bbox: list
    extend_point_bbox: list
    region_weight: Optional[float] = field(default=None)

class ProgressiveDataPartitioning(BasePartitioning):

    def __init__(self, scene_info, train_cameras, model_path, m_region=2, n_region=4, extend_rate=0.2, visible_rate=0.01, tau=0.12):
        self.tau = tau
        super().__init__(scene_info, train_cameras, model_path, m_region, n_region, extend_rate, visible_rate)
        self.partition_scene = [self._convert_to_mutable(p) for p in self.partition_scene]
        self.partition_scene = self.add_region_weights(self.partition_scene)
        self.save_partition_data_dir = os.path.join(self.model_path, 'partition_data_depthaware.pkl')
        self.save_partition_data()

    def _convert_to_mutable(self, part):
        return CameraPartitionMutable(partition_id=part.partition_id, cameras=part.cameras, point_cloud=part.point_cloud, ori_camera_bbox=getattr(part, 'ori_camera_bbox', []), extend_camera_bbox=getattr(part, 'extend_camera_bbox', []), extend_rate=getattr(part, 'extend_rate', 0.0), ori_point_bbox=getattr(part, 'ori_point_bbox', []), extend_point_bbox=getattr(part, 'extend_point_bbox', []))

    def _bbox_to_8corners(self, bbox):
        if bbox is None or len(bbox) == 0:
            return None
        if len(bbox) == 6:
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
        elif len(bbox) == 4:
            x_min, x_max, z_min, z_max = bbox
            y_min, y_max = (-1.0, 1.0)
        else:
            return None
        return np.array([[x_min, y_min, z_min], [x_min, y_min, z_max], [x_min, y_max, z_min], [x_min, y_max, z_max], [x_max, y_min, z_min], [x_max, y_min, z_max], [x_max, y_max, z_min], [x_max, y_max, z_max]])

    def _partition_centroid(self, part: CameraPartitionMutable) -> np.ndarray:
        pts = getattr(part.point_cloud, 'points', None)
        if pts is not None and len(pts) > 0:
            return np.mean(pts, axis=0)
        for key in ['extend_point_bbox', 'ori_point_bbox', 'extend_camera_bbox', 'ori_camera_bbox']:
            bbox = getattr(part, key, None)
            if bbox and len(bbox) >= 4:
                if len(bbox) == 6:
                    x_min, x_max, y_min, y_max, z_min, z_max = bbox
                    return np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2])
                else:
                    x_min, x_max, z_min, z_max = bbox
                    return np.array([(x_min + x_max) / 2, 0.0, (z_min + z_max) / 2])
        cams = part.cameras
        if cams:
            poses = np.array([c.pose for c in cams])
            return np.mean(poses, axis=0)
        return np.zeros(3)

    def _world_points_to_camera_z(self, camera: CameraInfo, points_world: np.ndarray) -> np.ndarray:
        if points_world is None or len(points_world) == 0:
            return np.array([], dtype=float)
        R = camera.R
        T = camera.T
        w2c_R = np.transpose(R)
        w2c_t = T
        pts_cam = (w2c_R @ points_world.T).T + w2c_t
        return pts_cam[:, 2]

    def compute_region_weight(self, partition: CameraPartitionMutable) -> float:
        cams = partition.cameras
        if len(cams) == 0:
            return 0.0
        rates, depths = ([], [])
        corner_pts = None
        ex_bbox = getattr(partition, 'extend_camera_bbox', None)
        if ex_bbox:
            corner_pts = self._bbox_to_8corners(ex_bbox)
        if corner_pts is None:
            ext_pt_bbox = getattr(partition, 'extend_point_bbox', None)
            if ext_pt_bbox and len(ext_pt_bbox) == 6:
                corner_pts = self._bbox_to_8corners(ext_pt_bbox)
        if corner_pts is None:
            c = self._partition_centroid(partition)
            delta = 0.5
            corner_pts = np.array([[c[0] - delta, c[1] - delta, c[2] - delta], [c[0] - delta, c[1] - delta, c[2] + delta], [c[0] + delta, c[1] + delta, c[2] - delta], [c[0] + delta, c[1] + delta, c[2] + delta], [c[0] - delta, c[1] + delta, c[2] - delta], [c[0] + delta, c[1] - delta, c[2] - delta], [c[0] + delta, c[1] - delta, c[2] + delta], [c[0] + delta, c[1] + delta, c[2] + delta]])
        pts_world = getattr(partition.point_cloud, 'points', None)
        centroid_world = self._partition_centroid(partition)
        for cam_pose in cams:
            cam = cam_pose.camera
            try:
                proj_all, _, _ = self.point_in_image(cam, corner_pts)
                if len(proj_all) < 3:
                    r_i = 0.0
                else:
                    pkg = run_graham_scan(np.clip(proj_all, 0, [cam.image_width, cam.image_height]), cam.image_width, cam.image_height)
                    r_i = float(pkg.get('intersection_rate', 0.0))
            except Exception:
                r_i = 0.0
            rates.append(r_i)
            d_i = None
            if pts_world is not None and len(pts_world) > 0:
                try:
                    z_vals = self._world_points_to_camera_z(cam, pts_world)
                    z_pos = z_vals[z_vals > 1e-06]
                    if z_pos.size > 0:
                        d_i = float(np.median(z_pos))
                except Exception:
                    pass
            if d_i is None:
                try:
                    z_c = self._world_points_to_camera_z(cam, centroid_world.reshape(1, 3))
                    if z_c.size > 0 and z_c[0] > 1e-06:
                        d_i = float(z_c[0])
                except Exception:
                    pass
            if d_i is None:
                try:
                    camera_center = np.array(cam_pose.pose)
                    d_i = float(np.linalg.norm(camera_center - centroid_world))
                except Exception:
                    d_i = 0.0
            depths.append(max(0.0, d_i))
        mean_rate = float(np.mean(rates)) if rates else 0.0
        mean_depth = float(np.mean(depths)) if depths else 0.0
        try:
            all_z = np.abs(self.scene_info.point_cloud.points[:, 2])
            scale_z = np.percentile(all_z, 99)
        except Exception:
            scale_z = max(1.0, mean_depth)
        norm_depth = mean_depth / max(scale_z, 1e-06)
        log_depth = math.log(1.0 + norm_depth * 100.0)
        bias = 0.01
        depth_factor = 1 + self.tau * log_depth
        weight = (mean_rate + bias) * depth_factor
        m, n = map(int, partition.partition_id.split('_'))
        is_corner = m == 1 and n == 1 or (m == 1 and n == self.n_region) or (m == self.m_region and n == 1) or (m == self.m_region and n == self.n_region)
        is_edge = m == 1 or m == self.m_region or n == 1 or (n == self.n_region)
        if is_corner:
            weight *= 2.0
            print(f'[EdgeBoost] Corner partition {partition.partition_id}: weight ×2.0')
        elif is_edge:
            weight *= 1.5
            print(f'[EdgeBoost] Edge partition {partition.partition_id}: weight ×1.5')
        weight = max(0.0, min(1.0, float(weight)))
        print(f'[DepthAware DEBUG] {partition.partition_id}: mean_rate={mean_rate:.3f}, mean_depth={mean_depth:.3f}, log_depth={log_depth:.3f}, weight={weight:.4f}')
        return weight

    def add_region_weights(self, partition_list: List[CameraPartitionMutable]) -> List[CameraPartitionMutable]:
        result = copy.deepcopy(partition_list)
        for idx, part in enumerate(result):
            try:
                w = self.compute_region_weight(part)
                part.region_weight = float(w)
            except Exception:
                part.region_weight = 0.0
            print(f'[DepthAware] Partition {part.partition_id}: region_weight={part.region_weight:.5f}')
        return result

    def save_partition_data(self):
        with open(self.save_partition_data_dir, 'wb') as f:
            pickle.dump(self.partition_scene, f)
