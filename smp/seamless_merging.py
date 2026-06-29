import os.path
import json
import numpy as np
from glob import glob
import pickle
import torch
from plyfile import PlyData, PlyElement
from smp.geom import storePly

def extend_inf_x_z_bbox(partition_id, m_region, n_region):
    x, z = (int(partition_id.split('_')[0]), int(partition_id.split('_')[1]))
    if x == 1 and z == 1:
        return [True, False, True, False]
    if x == m_region and z == 1:
        return [False, True, True, False]
    if x == 1 and z == n_region:
        return [True, False, False, True]
    if x == m_region and z == n_region:
        return [False, True, False, True]
    if 2 <= x <= m_region - 1 and z == 1:
        return [False, False, True, False]
    if 2 <= z <= n_region - 1 and x == 1:
        return [True, False, False, False]
    if 2 <= x <= m_region - 1 and z == n_region:
        return [False, False, False, True]
    if 2 <= z <= n_region - 1 and x == m_region:
        return [False, True, False, False]
    if 2 <= x <= m_region - 1 and 2 <= z <= n_region - 1:
        return [False, False, False, False]

def load_ply(path):
    plydata = PlyData.read(path)
    max_sh_degree = 3
    xyz = np.stack((np.asarray(plydata.elements[0]['x']), np.asarray(plydata.elements[0]['y']), np.asarray(plydata.elements[0]['z'])), axis=1)
    opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
    return (xyz, features_dc, features_extra, opacities, scales, rots)

def extract_point_cloud(points, bbox):
    mask = (points[:, 0] >= bbox[0]) & (points[:, 0] <= bbox[1]) & (points[:, 1] >= bbox[2]) & (points[:, 1] <= bbox[3]) & (points[:, 2] >= bbox[4]) & (points[:, 2] <= bbox[5])
    return mask

def seamless_merge(model_path, partition_point_cloud_dir):
    save_merge_dir = os.path.join(partition_point_cloud_dir, 'point_cloud.ply')
    with open(os.path.join(model_path, 'partition_data.pkl'), 'rb') as f:
        partition_scene = pickle.load(f)
    m_region, n_region = (0, 0)
    for partition in partition_scene:
        m, n = (int(partition.partition_id.split('_')[0]), int(partition.partition_id.split('_')[1]))
        if m > m_region:
            m_region = m
        if n > n_region:
            n_region = n
    xyz_list = []
    features_dc_list = []
    features_extra_list = []
    opacities_list = []
    scales_list = []
    rots_list = []
    for partition in partition_scene:
        point_cloud_path = os.path.join(partition_point_cloud_dir, f'{partition.partition_id}_point_cloud.ply')
        if not os.path.exists(point_cloud_path):
            continue
        xyz, features_dc, features_extra, opacities, scales, rots = load_ply(point_cloud_path)
        ori_camera_bbox = partition.ori_camera_bbox
        extend_camera_bbox = partition.extend_camera_bbox
        x_max = ori_camera_bbox[1]
        x_min = ori_camera_bbox[0]
        z_max = ori_camera_bbox[3]
        z_min = ori_camera_bbox[2]
        flag = extend_inf_x_z_bbox(partition.partition_id, m_region, n_region)
        x_max = np.inf if flag[1] else x_max
        x_min = -np.inf if flag[0] else x_min
        z_max = np.inf if flag[3] else z_max
        z_min = -np.inf if flag[2] else z_min
        print('region:', point_cloud_path)
        print('x_min:{}, x_max:{}, z_min:{}, z_max:{}'.format(x_min, x_max, z_min, z_max))
        point_select_bbox = [x_min, x_max, -np.inf, np.inf, z_min, z_max]
        mask = extract_point_cloud(xyz, point_select_bbox)
        xyz_list.append(xyz[mask])
        features_dc_list.append(features_dc[mask])
        features_extra_list.append(features_extra[mask])
        opacities_list.append(opacities[mask])
        scales_list.append(scales[mask])
        rots_list.append(rots[mask])
        print('point_cloud_path:', point_cloud_path, '\n')
        storePly(os.path.join(partition_point_cloud_dir, f'{partition.partition_id}_seamless.ply'), xyz[mask], np.zeros_like(xyz[mask]))
    points = np.concatenate(xyz_list, axis=0)
    features_dc_list = np.concatenate(features_dc_list, axis=0)
    features_extra_list = np.concatenate(features_extra_list, axis=0)
    opacities_list = np.concatenate(opacities_list, axis=0)
    scales_list = np.concatenate(scales_list, axis=0)
    rots_list = np.concatenate(rots_list, axis=0)
    points, mask = np.unique(points, axis=0, return_index=True)
    features_dc_list = features_dc_list[mask]
    features_extra_list = features_extra_list[mask]
    opacities_list = opacities_list[mask]
    scales_list = scales_list[mask]
    rots_list = rots_list[mask]
    _save_gaussian_ply(save_merge_dir, points, features_dc_list, features_extra_list, opacities_list, scales_list, rots_list)
if __name__ == '__main__':
    seamless_merge('output/train', 'output/train/point_cloud/iteration_60000')

def _save_gaussian_ply(path, xyz, features_dc, features_extra, opacities, scales, rots):
    dtype_full = [(attribute, 'f4') for attribute in ('x', 'y', 'z')] + [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    for i in range(3):
        dtype_full.append((f'f_dc_{i}', 'f4'))
    for i in range(features_extra.shape[1] * features_extra.shape[2]):
        dtype_full.append((f'f_rest_{i}', 'f4'))
    dtype_full.append(('opacity', 'f4'))
    for i in range(scales.shape[1]):
        dtype_full.append((f'scale_{i}', 'f4'))
    for i in range(rots.shape[1]):
        dtype_full.append((f'rot_{i}', 'f4'))
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attrs = [xyz, np.zeros_like(xyz)]
    f_dc = features_dc.transpose(0, 2, 1).reshape(xyz.shape[0], -1)
    f_rest = features_extra.transpose(0, 2, 1).reshape(xyz.shape[0], -1)
    attrs.extend([f_dc, f_rest, opacities, scales, rots])
    attributes = np.concatenate(attrs, axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyElement.describe(elements, 'vertex')
    ply = PlyData([PlyElement.describe(elements, 'vertex')])
    ply.write(path)
