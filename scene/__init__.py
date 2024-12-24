#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import cv2
import torch
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[2, 1, 0], resize_to_orig=False,
                 load_train = True, load_ply = False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = []
        self.test_cameras = []

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        print("Loading Training Cameras")
        if load_train:
            self.train_cameras, _ = cameraList_from_camInfos(scene_info.train_cameras, resolution_scales, args, resize_to_original=resize_to_orig)

        print("Loading Test Cameras")
        self.test_cameras, _ = cameraList_from_camInfos(scene_info.test_cameras, resolution_scales, args, resize_to_original=resize_to_orig)
        self.cur_resolution = 0
        self.resolution_scales=resolution_scales
        if self.loaded_iter:
            self.gaussians.load(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)
                                                           ),load_ply =load_ply)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, only_ply=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        os.makedirs(point_cloud_path, exist_ok=True)
        self.gaussians.save(point_cloud_path, only_ply = only_ply)
        #self.gaussians.save_mlp_checkpoints(point_cloud_path)

    def getTrainCameras(self):
        return self.train_cameras[self.cur_resolution]
    
    def clear_image(self):
        if self.cur_resolution-1>=0:
            self.train_cameras[self.cur_resolution-1] = None
            torch.cuda.empty_cache()
    
    def getTestCamerasOrig(self):
        return self.test_cameras[-1]
    def getTrainCamerasOrig(self):
        return self.train_cameras[-1]
    
    def getTestCameras(self):
        return self.test_cameras[self.cur_resolution]

    def up_one_resolution(self):
        if self.cur_resolution + 1 < len(self.resolution_scales):
            self.cur_resolution += 1

    @property
    def get_cur_resolution(self):
        return self.resolution_scales[self.cur_resolution]
    @property
    def isNotLastStage(self):
        if self.cur_resolution != len(self.resolution_scales)-1:
            return True
        return False