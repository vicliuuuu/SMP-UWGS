import copy
import os
import numpy as np
from typing import NamedTuple
import pickle
import math
from smp.geom import CameraInfo, storePly
from smp.geom import BasicPointCloud
from smp.graham_scan import run_graham_scan
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array

class CameraPartition(NamedTuple):
    partition_id: str
    cameras: list
    point_cloud: BasicPointCloud
    ori_camera_bbox: list
    extend_camera_bbox: list
    extend_rate: float
    ori_point_bbox: list
    extend_point_bbox: list

class ProgressiveDataPartitioning:

    def __init__(self, scene_info, train_cameras, model_path, m_region=2, n_region=4, extend_rate=0.1, visible_rate=0.25, sample_ratio=0.1):
        self.partition_scene = None
        self.pcd = scene_info.point_cloud
        self.model_path = model_path
        self.partition_dir = os.path.join(model_path, 'partition_point_cloud')
        self.partition_ori_dir = os.path.join(self.partition_dir, 'ori')
        self.partition_extend_dir = os.path.join(self.partition_dir, 'extend')
        self.partition_visible_dir = os.path.join(self.partition_dir, 'visible')
        self.save_partition_data_dir = os.path.join(self.model_path, 'partition_data.pkl')
        self.m_region = m_region
        self.n_region = n_region
        self.extend_rate = extend_rate
        self.visible_rate = visible_rate
        self.sample_ratio = sample_ratio
        if not os.path.exists(self.partition_ori_dir):
            os.makedirs(self.partition_ori_dir)
        if not os.path.exists(self.partition_extend_dir):
            os.makedirs(self.partition_extend_dir)
        if not os.path.exists(self.partition_visible_dir):
            os.makedirs(self.partition_visible_dir)
        self.fig, self.ax = self.draw_pcd(self.pcd, train_cameras)
        self.run_DataPartition(train_cameras)

    def draw_pcd(self, pcd, train_cameras):
        x_coords = pcd.points[:, 0]
        z_coords = pcd.points[:, 2]
        fig, ax = plt.subplots()
        ax.scatter(x_coords, z_coords, c=pcd.colors, s=1)
        ax.title.set_text('Plot of 2D Points')
        ax.title.set_fontsize(20)
        ax.set_xlabel('X-axis', fontsize=20)
        ax.set_ylabel('Z-axis', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_path, 'pcd.png'), dpi=200)
        x_coords = np.array([cam.camera_center[0].item() for cam in train_cameras])
        z_coords = np.array([cam.camera_center[2].item() for cam in train_cameras])
        ax.scatter(x_coords, z_coords, color='red', s=1)
        fig.savefig(os.path.join(self.model_path, 'camera_on_pcd.png'), dpi=200)
        return (fig, ax)

    def draw_partition(self, partition_list):
        for partition in partition_list:
            ori_bbox = partition.ori_camera_bbox
            extend_bbox = partition.extend_camera_bbox
            x_min, x_max, z_min, z_max = ori_bbox
            ex_x_min, ex_x_max, ex_z_min, ex_z_max = extend_bbox
            rect_ori = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, linewidth=3, edgecolor='blue', facecolor='none')
            rect_ext = patches.Rectangle((ex_x_min, ex_z_min), ex_x_max - ex_x_min, ex_z_max - ex_z_min, linewidth=2, edgecolor='y', facecolor='none')
            self.ax.add_patch(rect_ori)
            self.ax.text(x=rect_ori.get_x(), y=rect_ori.get_y(), s=f'{partition.partition_id}', color='black', fontsize=12)
            self.ax.add_patch(rect_ext)
        self.fig.savefig(os.path.join(self.model_path, f'regions.png'), dpi=200)
        return

    def run_DataPartition(self, train_cameras):
        if not os.path.exists(self.save_partition_data_dir):
            partition_dict = self.Camera_position_based_region_division(train_cameras)
            partition_dict, refined_ori_bbox = self.refine_ori_bbox(partition_dict)
            partition_list = self.Position_based_data_selection(partition_dict, refined_ori_bbox)
            self.draw_partition(partition_list)
            self.partition_scene = self.Visibility_based_camera_selection(partition_list)
            self.save_partition_data()
        else:
            self.partition_scene = self.load_partition_data()

    def save_partition_data(self):
        with open(self.save_partition_data_dir, 'wb') as f:
            pickle.dump(self.partition_scene, f)

    def load_partition_data(self):
        with open(self.save_partition_data_dir, 'rb') as f:
            partition_scene = pickle.load(f)
        return partition_scene

    def refine_ori_bbox_average(self, partition_dict):
        bbox_with_id = {}
        for partition_idx, cameras in partition_dict.items():
            camera_list = cameras['camera_list']
            min_x, max_x = (min((camera.pose[0] for camera in camera_list)), max((camera.pose[0] for camera in camera_list)))
            min_z, max_z = (min((camera.pose[2] for camera in camera_list)), max((camera.pose[2] for camera in camera_list)))
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox
        for m in range(1, self.m_region + 1):
            for n in range(1, self.n_region + 1):
                if n + 1 == self.n_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n)
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n + 1)
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_z = (max_z_1 + min_z_2) / 2
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, min_z_1, mid_z]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, mid_z, max_z_2]
        for m in range(1, self.m_region + 1):
            if m + 1 == self.m_region + 1:
                break
            max_x_left = -np.inf
            min_x_right = np.inf
            for n in range(1, self.n_region + 1):
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if max_x > max_x_left:
                    max_x_left = max_x
            for n in range(1, self.n_region + 1):
                partition_idx = str(m + 1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if min_x < min_x_right:
                    min_x_right = min_x
            for n in range(1, self.n_region + 1):
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [min_x, mid_x, min_z, max_z]
            for n in range(1, self.n_region + 1):
                partition_idx = str(m + 1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [mid_x, max_x, min_z, max_z]
        new_partition_dict = {f'{partition_id}': cameras['camera_list'] for partition_id, cameras in partition_dict.items()}
        return (new_partition_dict, bbox_with_id)

    def refine_ori_bbox(self, partition_dict):
        bbox_with_id = {}
        for partition_idx, cameras in partition_dict.items():
            camera_list = cameras['camera_list']
            min_x, max_x = (min((camera.pose[0] for camera in camera_list)), max((camera.pose[0] for camera in camera_list)))
            min_z, max_z = (min((camera.pose[2] for camera in camera_list)), max((camera.pose[2] for camera in camera_list)))
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox
        for m in range(1, self.m_region + 1):
            for n in range(1, self.n_region + 1):
                if n + 1 == self.n_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n + 1)
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n)
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_x, mid_y, mid_z = partition_dict[partition_idx_2]['z_mid_camera'].pose
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, mid_z, max_z_1]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, min_z_2, mid_z]
        for n in range(1, self.n_region + 1):
            for m in range(1, self.m_region + 1):
                if m + 1 == self.m_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n)
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m + 1) + '_' + str(n)
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_x, mid_y, mid_z = partition_dict[partition_idx_1]['x_mid_camera'].pose
                bbox_with_id[partition_idx_1] = [min_x_1, mid_x, min_z_1, max_z_1]
                bbox_with_id[partition_idx_2] = [mid_x, max_x_2, min_z_2, max_z_2]
        new_partition_dict = {f'{partition_id}': cameras['camera_list'] for partition_id, cameras in partition_dict.items()}
        return (new_partition_dict, bbox_with_id)

    def Camera_position_based_region_division(self, train_cameras):
        m, n = (self.m_region, self.n_region)
        CameraPose_list = []
        camera_centers = []
        for idx, camera in enumerate(train_cameras):
            pose = np.array(camera.camera_center.cpu())
            camera_centers.append(pose)
            CameraPose_list.append(CameraPose(camera=camera, pose=pose))
        storePly(os.path.join(self.partition_dir, 'camera_centers.ply'), np.array(camera_centers), np.zeros_like(np.array(camera_centers)))
        m_partition_dict = {}
        total_camera = len(CameraPose_list)
        num_of_camera_per_m_partition = total_camera // m
        sorted_CameraPose_by_x_list = sorted(CameraPose_list, key=lambda x: x.pose[0])
        for i in range(m):
            m_partition_dict[str(i + 1)] = {'camera_list': sorted_CameraPose_by_x_list[i * num_of_camera_per_m_partition:(i + 1) * num_of_camera_per_m_partition]}
            if i != m - 1:
                m_partition_dict[str(i + 1)].update({'x_mid_camera': sorted_CameraPose_by_x_list[(i + 1) * num_of_camera_per_m_partition - 1]})
            else:
                m_partition_dict[str(i + 1)].update({'x_mid_camera': None})
        if total_camera % m != 0:
            m_partition_dict[str(m)]['camera_list'].extend(sorted_CameraPose_by_x_list[m * num_of_camera_per_m_partition:])
        partition_dict = {}
        for partition_idx, cameras in m_partition_dict.items():
            partition_total_camera = len(cameras['camera_list'])
            num_of_camera_per_n_partition = partition_total_camera // n
            sorted_CameraPose_by_z_list = sorted(cameras['camera_list'], key=lambda x: x.pose[2])
            for i in range(n):
                partition_dict[f'{partition_idx}_{i + 1}'] = {'camera_list': sorted_CameraPose_by_z_list[i * num_of_camera_per_n_partition:(i + 1) * num_of_camera_per_n_partition]}
                if i != n - 1:
                    partition_dict[f'{partition_idx}_{i + 1}'].update({'x_mid_camera': cameras['x_mid_camera']})
                    partition_dict[f'{partition_idx}_{i + 1}'].update({'z_mid_camera': sorted_CameraPose_by_z_list[(i + 1) * num_of_camera_per_n_partition - 1]})
                else:
                    partition_dict[f'{partition_idx}_{i + 1}'].update({'x_mid_camera': cameras['x_mid_camera']})
                    partition_dict[f'{partition_idx}_{i + 1}'].update({'z_mid_camera': None})
            if partition_total_camera % n != 0:
                partition_dict[f'{partition_idx}_{n}']['camera_list'].extend(sorted_CameraPose_by_z_list[n * num_of_camera_per_n_partition:])
        return partition_dict

    def extract_point_cloud(self, pcd, bbox):
        mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (pcd.points[:, 2] >= bbox[2]) & (pcd.points[:, 2] <= bbox[3])
        points = pcd.points[mask]
        colors = pcd.colors[mask]
        normals = pcd.normals[mask]
        return (points, colors, normals)

    def get_point_range(self, points):
        x_list = points[:, 0]
        y_list = points[:, 1]
        z_list = points[:, 2]
        return [min(x_list), max(x_list), min(y_list), max(y_list), min(z_list), max(z_list)]

    def identify_edge_partitions_from_dict(self, partition_dict):
        edge_ids = set()
        for partition_idx in partition_dict.keys():
            try:
                m, n = map(int, partition_idx.split('_'))
                if m == 1 and n == 1 or (m == 1 and n == self.n_region) or (m == self.m_region and n == 1) or (m == self.m_region and n == self.n_region):
                    edge_ids.add(partition_idx)
                elif m == 1 or m == self.m_region or n == 1 or (n == self.n_region):
                    edge_ids.add(partition_idx)
            except ValueError:
                continue
        print(f'[Edge Detection] Identified edge partitions: {edge_ids}')
        return edge_ids

    def Position_based_data_selection(self, partition_dict, refined_ori_bbox):
        pcd = self.pcd
        partition_list = []
        edge_partitions = self.identify_edge_partitions_from_dict(partition_dict)
        complete_points = pcd.points
        complete_colors = pcd.colors
        complete_normals = pcd.normals
        num_points = len(complete_points)
        sample_size = int(num_points * self.sample_ratio)
        indices = np.random.choice(num_points, sample_size, replace=False)
        sampled_points = complete_points[indices]
        sampled_colors = complete_colors[indices]
        sampled_normals = complete_normals[indices]
        print(f'🎯 Global Sampling: {num_points} -> {sample_size} points ({self.sample_ratio:.1%})')
        for partition_idx, camera_list in partition_dict.items():
            min_x, max_x, min_z, max_z = refined_ori_bbox[partition_idx]
            base_extend_rate = self.extend_rate
            if partition_idx in edge_partitions:
                extend_rate = base_extend_rate * 2.0
            else:
                extend_rate = base_extend_rate * 1.5
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            extend_camera_bbox = [min_x - extend_rate * (max_x - min_x), max_x + extend_rate * (max_x - min_x), min_z - extend_rate * (max_z - min_z), max_z + extend_rate * (max_z - min_z)]
            base_points = sampled_points.copy()
            base_colors = sampled_colors.copy()
            base_normals = sampled_normals.copy()
            focus_points, focus_colors, focus_normals = self.extract_point_cloud(pcd, extend_camera_bbox)
            if len(focus_points) > 0:
                enhanced_points = np.concatenate([base_points, focus_points], axis=0)
                enhanced_colors = np.concatenate([base_colors, focus_colors], axis=0)
                enhanced_normals = np.concatenate([base_normals, focus_normals], axis=0)
            else:
                enhanced_points = base_points
                enhanced_colors = base_colors
                enhanced_normals = base_normals
            if partition_idx in edge_partitions:
                neighbor_points = self.get_adjacent_partition_points(partition_idx, partition_dict, refined_ori_bbox, pcd)
                if len(neighbor_points) > 0:
                    neighbor_colors = np.ones_like(neighbor_points) * 0.7
                    enhanced_points = np.concatenate([enhanced_points, neighbor_points], axis=0)
                    enhanced_colors = np.concatenate([enhanced_colors, neighbor_colors], axis=0)
                    enhanced_normals = np.concatenate([enhanced_normals, np.zeros_like(neighbor_points)], axis=0)
            new_camera_list = []
            for id, camera_list_all in partition_dict.items():
                for camera_pose in camera_list_all:
                    if extend_camera_bbox[0] <= camera_pose.pose[0] <= extend_camera_bbox[1] and extend_camera_bbox[2] <= camera_pose.pose[2] <= extend_camera_bbox[3]:
                        new_camera_list.append(camera_pose)
            total_points = len(enhanced_points)
            focus_ratio = len(focus_points) / total_points if total_points > 0 else 0
            partition_list.append(CameraPartition(partition_id=partition_idx, cameras=new_camera_list, point_cloud=BasicPointCloud(enhanced_points, enhanced_colors, enhanced_normals), ori_camera_bbox=ori_camera_bbox, extend_camera_bbox=extend_camera_bbox, extend_rate=extend_rate, ori_point_bbox=self.get_point_range(focus_points), extend_point_bbox=self.get_point_range(enhanced_points)))
            print(f'🎯 Partition {partition_idx}: {total_points} points (focus: {len(focus_points)}, complete: {len(complete_points)}, focus_ratio: {focus_ratio:.1%})')
        return partition_list

    def get_adjacent_partition_points(self, target_partition_id, partition_dict, refined_ori_bbox, pcd):
        adjacent_points = []
        target_bbox = refined_ori_bbox[target_partition_id]
        for partition_id, bbox in refined_ori_bbox.items():
            if partition_id == target_partition_id:
                continue
            if self.are_bboxes_adjacent(target_bbox, bbox):
                points, _, _ = self.extract_point_cloud(pcd, bbox)
                if len(points) > 0:
                    edge_points = self.extract_edge_points(points, target_bbox, bbox)
                    if len(edge_points) > 0:
                        adjacent_points.append(edge_points)
                        print(f'  ↳ Added {len(edge_points)} edge points from partition {partition_id}')
        return np.concatenate(adjacent_points, axis=0) if adjacent_points else np.array([])

    def extract_edge_points(self, points, target_bbox, neighbor_bbox, edge_ratio=0.2):
        bbox_width = neighbor_bbox[1] - neighbor_bbox[0]
        bbox_height = neighbor_bbox[3] - neighbor_bbox[2]
        edge_width = bbox_width * edge_ratio
        edge_height = bbox_height * edge_ratio
        shared_boundary = self.find_shared_boundary(target_bbox, neighbor_bbox)
        if shared_boundary == 'right':
            mask = points[:, 0] <= neighbor_bbox[0] + edge_width
        elif shared_boundary == 'left':
            mask = points[:, 0] >= neighbor_bbox[1] - edge_width
        elif shared_boundary == 'top':
            mask = points[:, 1] <= neighbor_bbox[2] + edge_height
        elif shared_boundary == 'bottom':
            mask = points[:, 1] >= neighbor_bbox[3] - edge_height
        return points[mask]

    def find_shared_boundary(self, bbox1, bbox2):
        min_x1, max_x1, min_z1, max_z1 = bbox1
        min_x2, max_x2, min_z2, max_z2 = bbox2
        if abs(max_x1 - min_x2) < 1e-06:
            return 'right'
        elif abs(min_x1 - max_x2) < 1e-06:
            return 'left'
        elif abs(max_z1 - min_z2) < 1e-06:
            return 'top'
        elif abs(min_z1 - max_z2) < 1e-06:
            return 'bottom'
        return None

    def are_bboxes_adjacent(self, bbox1, bbox2, distance_threshold=1.0):
        x_min1, x_max1, z_min1, z_max1 = bbox1
        x_min2, x_max2, z_min2, z_max2 = bbox2
        center1 = np.array([(x_min1 + x_max1) / 2, (z_min1 + z_max1) / 2])
        center2 = np.array([(x_min2 + x_max2) / 2, (z_min2 + z_max2) / 2])
        distance = np.linalg.norm(center1 - center2)
        x_overlap = not (x_max1 < x_min2 or x_max2 < x_min1)
        z_overlap = not (z_max1 < z_min2 or z_max2 < z_min1)
        return x_overlap and z_overlap or distance < distance_threshold

    def get_8_corner_points(self, bbox):
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        return {'minx_miny_minz': [x_min, y_min, z_min], 'minx_miny_maxz': [x_min, y_min, z_max], 'minx_maxy_minz': [x_min, y_max, z_min], 'minx_maxy_maxz': [x_min, y_max, z_max], 'maxx_miny_minz': [x_max, y_min, z_min], 'maxx_miny_maxz': [x_max, y_min, z_max], 'maxx_maxy_minz': [x_max, y_max, z_min], 'maxx_maxy_maxz': [x_max, y_max, z_max]}

    def point_in_image(self, camera, points):
        R = camera.R
        T = camera.T
        w2c = np.eye(4)
        w2c[:3, :3] = np.transpose(R)
        w2c[:3, 3] = T
        fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))
        intrinsic_matrix = np.array([[fx, 0, camera.image_width // 2], [0, fy, camera.image_height // 2], [0, 0, 1]])
        points_camera = np.dot(w2c[:3, :3], points.T) + w2c[:3, 3:].reshape(3, 1)
        points_camera = points_camera.T
        points_camera = points_camera[np.where(points_camera[:, 2] > 0)]
        points_image = np.dot(intrinsic_matrix, points_camera.T)
        points_image = points_image[:2, :] / points_image[2, :]
        points_image = points_image.T
        mask = np.where(np.logical_and.reduce((points_image[:, 0] >= 0, points_image[:, 0] < camera.image_height, points_image[:, 1] >= 0, points_image[:, 1] < camera.image_width)))[0]
        return (points_image, points_image[mask], mask)

    def identify_edge_partitions(self, partition_list):
        edge_ids = set()
        for partition in partition_list:
            try:
                m, n = map(int, partition.partition_id.split('_'))
                if m == 1 and n == 1 or (m == 1 and n == self.n_region) or (m == self.m_region and n == 1) or (m == self.m_region and n == self.n_region):
                    edge_ids.add(partition.partition_id)
                elif m == 1 or m == self.m_region or n == 1 or (n == self.n_region):
                    edge_ids.add(partition.partition_id)
            except ValueError:
                continue
        print(f'[Edge Detection] Identified edge partitions: {edge_ids}')
        return edge_ids

    def Visibility_based_camera_selection(self, partition_list):
        add_visible_camera_partition_list = copy.deepcopy(partition_list)
        edge_partitions = self.identify_edge_partitions(partition_list)
        for idx, partition_i in enumerate(partition_list):
            new_points = []
            new_colors = []
            new_normals = []
            pcd_i = partition_i.point_cloud
            partition_id_i = partition_i.partition_id
            is_edge_partition = partition_id_i in edge_partitions
            partition_ori_point_bbox = partition_i.ori_point_bbox
            partition_extend_point_bbox = partition_i.extend_point_bbox
            ori_8_corner_points = self.get_8_corner_points(partition_ori_point_bbox)
            extent_8_corner_points = self.get_8_corner_points(partition_extend_point_bbox)
            corner_points = []
            for point in extent_8_corner_points.values():
                corner_points.append(point)
            storePly(os.path.join(self.partition_extend_dir, f'{partition_id_i}_corner_points.ply'), np.array(corner_points), np.zeros_like(np.array(corner_points)))
            total_partition_camera_count = 0
            visible_threshold = self.visible_rate
            if is_edge_partition:
                visible_threshold = self.visible_rate * 0.3
                print(f'[Edge Enhancement] Partition {partition_id_i} using relaxed threshold: {visible_threshold}')
            for partition_j in partition_list:
                partition_id_j = partition_j.partition_id
                if partition_id_i == partition_id_j:
                    continue
                print(f'Now processing partition i:{partition_id_i} and j:{partition_id_j}')
                pcd_j = partition_j.point_cloud
                append_camera_count = 0
                for cameras_pose in partition_j.cameras:
                    camera = cameras_pose.camera
                    proj_8_corner_points = {}
                    for key, point in extent_8_corner_points.items():
                        points_in_image, _, _ = self.point_in_image(camera, np.array([point]))
                        if len(points_in_image) == 0:
                            continue
                        proj_8_corner_points[key] = points_in_image[0]
                    if not len(list(proj_8_corner_points.values())) > 2:
                        continue
                    try:
                        pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height, is_edge_partition=is_edge_partition)
                    except Exception as e:
                        print(f'[GrahamScan Error] Partition {partition_id_i}: {e}')
                        continue
                    if pkg['intersection_rate'] >= visible_threshold:
                        collect_names = [camera_pose.camera.image_name for camera_pose in add_visible_camera_partition_list[idx].cameras]
                        if cameras_pose.camera.image_name in collect_names:
                            continue
                        append_camera_count += 1
                        add_visible_camera_partition_list[idx].cameras.append(cameras_pose)
                        _, _, mask = self.point_in_image(camera, pcd_j.points)
                        updated_points, updated_colors, updated_normals = (pcd_j.points[mask], pcd_j.colors[mask], pcd_j.normals[mask])
                        if is_edge_partition and len(updated_points) == 0:
                            all_points_image, all_mask = self.point_in_image_relaxed(camera, pcd_j.points)
                            if len(all_points_image) > 0:
                                updated_points = pcd_j.points[all_mask]
                                updated_colors = pcd_j.colors[all_mask]
                                updated_normals = pcd_j.normals[all_mask]
                        new_points.append(updated_points)
                        new_colors.append(updated_colors)
                        new_normals.append(updated_normals)
                        with open(os.path.join(self.model_path, 'graham_scan'), 'a') as f:
                            f.write(f'intersection_area:{pkg['intersection_area']:.2f}\timage_area:{pkg['image_area']:.2f}\tintersection_rate:{pkg['intersection_rate']:.4f}\tpartition_i:{partition_id_i}\tpartition_j:{partition_id_j}\tis_edge:{is_edge_partition}\tappend_camera_id:{camera.image_name}\tappend_camera_count:{append_camera_count}\n')
                total_partition_camera_count += append_camera_count
            with open(os.path.join(self.model_path, 'partition_cameras'), 'a') as f:
                f.write(f'partition_id:{partition_id_i}\tis_edge:{is_edge_partition}\ttotal_append_camera_count:{total_partition_camera_count}\ttotal_camera:{len(add_visible_camera_partition_list[idx].cameras)}\n')
            camera_centers = []
            for camera_pose in add_visible_camera_partition_list[idx].cameras:
                camera_centers.append(camera_pose.pose)
            storePly(os.path.join(self.partition_visible_dir, f'{partition_id_i}_camera_centers.ply'), np.array(camera_centers), np.zeros_like(np.array(camera_centers)))
            point_cloud = add_visible_camera_partition_list[idx].point_cloud
            new_points.append(point_cloud.points)
            new_colors.append(point_cloud.colors)
            new_normals.append(point_cloud.normals)
            if new_points:
                new_points = np.concatenate(new_points, axis=0)
                new_colors = np.concatenate(new_colors, axis=0)
                new_normals = np.concatenate(new_normals, axis=0)
                new_points, mask = np.unique(new_points, return_index=True, axis=0)
                new_colors = new_colors[mask]
                new_normals = new_normals[mask]
                add_visible_camera_partition_list[idx] = add_visible_camera_partition_list[idx]._replace(point_cloud=BasicPointCloud(points=new_points, colors=new_colors, normals=new_normals))
            storePly(os.path.join(self.partition_visible_dir, f'{partition_id_i}_visible.ply'), new_points, new_colors)
        return add_visible_camera_partition_list

    def point_in_image_relaxed(self, camera, points, margin_ratio=0.1):
        R = camera.R
        T = camera.T
        w2c = np.eye(4)
        w2c[:3, :3] = np.transpose(R)
        w2c[:3, 3] = T
        fx = camera.image_width / (2 * math.tan(camera.FoVx / 2))
        fy = camera.image_height / (2 * math.tan(camera.FoVy / 2))
        intrinsic_matrix = np.array([[fx, 0, camera.image_width // 2], [0, fy, camera.image_height // 2], [0, 0, 1]])
        points_camera = np.dot(w2c[:3, :3], points.T) + w2c[:3, 3:].reshape(3, 1)
        points_camera = points_camera.T
        points_camera = points_camera[np.where(points_camera[:, 2] > 0)]
        points_image = np.dot(intrinsic_matrix, points_camera.T)
        points_image = points_image[:2, :] / points_image[2, :]
        points_image = points_image.T
        margin_w = camera.image_width * margin_ratio
        margin_h = camera.image_height * margin_ratio
        mask = np.where(np.logical_and.reduce((points_image[:, 0] >= -margin_w, points_image[:, 0] < camera.image_width + margin_w, points_image[:, 1] >= -margin_h, points_image[:, 1] < camera.image_height + margin_h)))[0]
        return (points_image, mask)
