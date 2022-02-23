import os
import torch
import random
import math
import numpy as np
import os.path as osp
import json
import copy
from pycocotools.coco import COCO

from core.config import cfg
from coord_utils import world2cam, cam2pixel, process_bbox
from base_dataset import BaseDataset

#human36m의 데이터를 읽어서 dataset을 만드는 클래스

class Human36M(BaseDataset):
    def __init__(self, transform, data_split):
        super(Human36M, self).__init__()
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join('data', 'Human36M', 'images')
        self.annot_path = osp.join('data', 'Human36M', 'annotations')

        self.joint_set = {
            'name': 'Human36M',
            'joint_num': 17,
            'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
            'flip_pairs': ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)),
            'skeleton': ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        }
        # flip pairs : 이미지를 flip할때 joint의 번호를 바꾸어야함(오른쪽 팔꿈치 <-> 왼쪽 팔꿈치 등) 그때 대응되는 joint 번호
        # skeleton : joint끼리 이어지는 경로

        self.root_joint_idx = self.joint_set['joints_name'].index('Pelvis')

        # 일반적으로 골반 = pelvis가 기준이됨
        
        self.has_joint_cam = True
        self.has_smpl_param = cfg.TRAIN.use_pseudo_GT
        self.datalist = self.load_data()
        #데이터 읽기
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    #train/test일때 다른 세트 사용
    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject



    def load_data(self):

        """
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # data의 annotation json파일이 coco data format와 유사하게 되어있음(https://cocodataset.org/#format-data) 
        # 이 annotation json파일에서 이미지의 여러 정보들을 읽음
        db = COCO()
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            if cfg.TRAIN.use_pseudo_GT:
                # smpl parameter load
                with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPL_NeuralAnnot.json'),'r') as f:
                    smpl_params[str(subject)] = json.load(f)
            else:
                smpl_params = None
        
        db.createIndex()
        """

        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # data의 annotation json파일이 coco data format와 유사하게 되어있음(https://cocodataset.org/#format-data) 
        # 이 annotation json파일에서 이미지의 여러 정보들을 읽음
        db = COCO()  #pycocotools 라이브러리
        cameras = {}
        joints = {}
        smpl_params = {}
        for subject in subject_list:
            # data load
            # 데이터 폴더에 있는 annotation json 파일 읽는 과정
            with open(osp.join(self.annot_path, f'Human36M_subject{subject}_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, f'Human36M_subject{subject}_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, f'Human36M_subject{subject}_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            if cfg.TRAIN.use_pseudo_GT:
                # smpl parameter load
                with open(osp.join(self.annot_path, f'Human36M_subject{subject}_SMPL_NeuralAnnot.json'),'r') as f:
                    smpl_params[str(subject)] = json.load(f)
            else:
                smpl_params = None
        
        #수동으로 직접 넣은 데이터를 사용해 db.anns/db.imgs를 정리 - anns에 indx를 넣으면 데이터가 나오도록
        #https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        db.createIndex()
        
        
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            # 이미지 로딩

            # annotation json파일에 있는 여러 데이터들 파이썬 변수로 옮기기
            
            # check subject and frame_idx
            frame_idx = img['frame_idx']
            if frame_idx % sampling_ratio != 0:
                continue

            # check smpl parameter exist
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
            if smpl_params is not None:
                try:
                    smpl_param = smpl_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]
                except KeyError:
                    smpl_param = None
            else:
                smpl_param = None
                
            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
            
            # bbox를 우리가 사용할 bbox의 가로세로 비율에 맞게 조정. coco bbox : xywh
            # ex) 256*192 비율이 되도록 해야하는데 400*200이면 가로를 확장 + 여유확장 : (400*1.25, 300*1.25)
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t) # 절대 좌표계에서 카메라 좌표계로 변환
            joint_img = cam2pixel(joint_cam, f, c)[:,:2] #카메라 좌표계에서 카메라 사영좌표계로 변환
            joint_valid = np.ones((self.joint_set['joint_num'],))
            #joint_world : 절대 좌표계에서 joint 좌표
            #joint_cam : 카메라 좌표계에서 joint 좌표
            #joint_img : 카메라에 사영된 joint의 2차원 좌표 

            #각종 데이터 읽고 datalist에 추가
            
            datalist.append({
                'ann_id': aid,
                'img_id': image_id,
                'img_path': img_path,
                'img_shape': (img['height'], img['width']),
                'bbox': bbox,
                'joint_img': joint_img, 
                'joint_cam': joint_cam/1000,
                'joint_valid': joint_valid,
                'smpl_param': smpl_param,
                'cam_param': cam_param
                })

        return datalist