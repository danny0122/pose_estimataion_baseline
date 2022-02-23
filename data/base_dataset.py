import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
import cv2

from torch.utils.data import Dataset

from core.config import cfg
from core.logger import logger
from img_utils import load_img, annToMask
from coord_utils import generate_joint_heatmap, sampling_non_joint, image_bound_check
from aug_utils import img_processing, coord2D_processing, coord3D_processing, smpl_param_processing, flip_joint, transform_joint_to_other_db
from human_models import smpl

from vis_utils import vis_keypoints, vis_keypoints_with_skeleton, vis_3d_pose, vis_heatmaps, save_obj


class BaseDataset(Dataset):
    def __init__(self):
        self.transform = None
        self.data_split = None
        self.has_joint_cam = False
        self.has_smpl_param = False
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        #깊은 복사 - 그냥 복사해버리면 복사한 배열에서 뭔가를 수정하면 원래 배열에 영향이 갈 수 도 있음
        data = copy.deepcopy(self.datalist[index])
        
        img_path = data['img_path']
        img = load_img(img_path)

        bbox, joint_img, joint_valid = data['bbox'], data['joint_img'], data['joint_valid']
        
        # 원본 이미지에서 형식에 맞는 image patch 추출 / augmentation 및 그 변수 얻기
        img, img2bb_trans, bb2img_trans, rot, do_flip = img_processing(img, bbox, self.data_split)
        #joint의 원본 이미지에서 좌표를 image patch에서의 좌표로 변환 / flip됬을 경우 joint번호 변경처리  
        joint_img = coord2D_processing(joint_img, img2bb_trans, do_flip, cfg.MODEL.input_img_shape, self.joint_set['flip_pairs'])
        if do_flip: joint_valid = flip_joint(joint_valid, None, self.joint_set['flip_pairs'])
        
        #augmentation 됬을 경우 그에 맞게 카메라 좌표계에서 3d좌표도 변경
        if self.has_joint_cam:
            joint_cam = coord3D_processing(data['joint_cam'], rot, do_flip, self.joint_set['flip_pairs'])
            joint_cam = joint_cam - joint_cam[self.root_joint_idx]
            has_3D = np.array([1])
        else:
            joint_cam = np.zeros((smpl.joint_num, 3))
            has_3D = np.array([0])
        
        #augmentation 됬을 경우 그에 맞게 smpl pose의 루트만 변경 / 나머지는 변하지 않음
        if self.has_smpl_param:
            smpl_pose, smpl_shape = smpl_param_processing(data['smpl_param'], data['cam_param'], do_flip, rot)
            mesh_cam, smpl_joint_cam = self.get_smpl_coord(smpl_pose, smpl_shape)
            has_param = np.array([1])
        else:
            smpl_pose, smpl_shape = np.zeros((smpl.joint_num*3,)), np.zeros((smpl.shape_param_dim,))
            smpl_joint_cam = np.zeros((smpl.joint_num, 3))
            has_param = np.array([0])
        
        if self.data_split == 'train':
            img = self.transform(img.astype(np.float32))
            
            # convert joint set / #다른 데이터 셋에서 joint 수, 순서 등이 다른 걸 통일시킴
            joint_img = transform_joint_to_other_db(joint_img, self.joint_set['joints_name'], smpl.joints_name)
            joint_cam = transform_joint_to_other_db(joint_cam, self.joint_set['joints_name'], smpl.joints_name)
            joint_valid = transform_joint_to_other_db(joint_valid, self.joint_set['joints_name'], smpl.joints_name)

            batch = {
                'img': img,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'smpl_joint_cam': smpl_joint_cam,
                'joint_valid': joint_valid,
                'has_3D': has_3D,
                'pose': smpl_pose,
                'shape': smpl_shape,
                'has_param': has_param
            }
        else:
            img = self.transform(img.astype(np.float32))

            mesh_cam = np.zeros((smpl.vertex_num, 3))
            joint_cam = joint_cam * 1000
                
            batch = {
                'img': img,
                'joint_cam': joint_cam,
                'mesh_cam': mesh_cam
            }
        
        return batch
    
    #smpl pose, shape로 부터 smpl joint, mesh 좌표 얻기
    def get_smpl_coord(self, smpl_pose, smpl_shape):
        root_pose, body_pose, smpl_shape = torch.tensor(smpl_pose[:3]).reshape(1,-1), torch.tensor(smpl_pose[3:]).reshape(1,-1), torch.tensor(smpl_shape).reshape(1,-1)
        output = smpl.layer['neutral'](betas=smpl_shape, body_pose=body_pose, global_orient=root_pose)
        smpl_mesh_cam = output.vertices[0].numpy()
        smpl_joint_cam = np.dot(smpl.joint_regressor, smpl_mesh_cam)
        smpl_joint_cam = smpl_joint_cam - smpl_joint_cam[smpl.root_joint_idx]
        return smpl_mesh_cam, smpl_joint_cam