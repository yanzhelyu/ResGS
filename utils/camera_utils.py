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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
import cv2 as cv

WARNED = False

def loadCam(args, id, cam_info, resolution_scales,resize_to_original=True):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(1.0 * args.resolution)), round(orig_h/(1.0 * args.resolution))
        #print("Resolution: ", args.resolution)  
        
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(1.0)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    downed_image = gt_image.permute(1, 2, 0).cpu().numpy()
    image_pyramid = []
    up_image_pyramid = []
    last_size=None
    size_list=[]
    last_resolution=0
    for i in range(resolution_scales[0] + 1):
        if i in resolution_scales:
            if resize_to_original:
                uped_resized_image=downed_image
                for j in range(1,i+1):
                    cur_size=size_list[-j]
                    uped_resized_image=cv.pyrUp(uped_resized_image,dstsize=(cur_size[1],cur_size[0]))
                gt_image=torch.from_numpy(uped_resized_image).permute(2, 0, 1).float()
            else:
                gt_image = torch.from_numpy(downed_image).permute(2, 0, 1).float()
            if i==0:
                up_image_pyramid.append(None)
            else:
                if resize_to_original is False:
                    uped_image=cv.pyrUp(downed_image,dstsize=(last_size[1],last_size[0]))
                    for j in range(1,i-last_resolution):
                        cur_size=size_list[-(j+1)]
                        uped_image=cv.pyrUp(uped_image,dstsize=(cur_size[1],cur_size[0]))
                else:
                    uped_image=uped_resized_image

                uped_image_torch=torch.from_numpy(uped_image).permute(2, 0, 1).float().cuda()
                up_image_pyramid.append(Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=uped_image_torch, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device))
            loaded_mask = None
            image_pyramid.append(Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device))
            last_resolution=i
        last_size=downed_image.shape
        size_list.append(last_size)
        downed_image = cv.pyrDown(downed_image)

    up_image_pyramid.reverse()
    image_pyramid.reverse()

    return image_pyramid, up_image_pyramid

def cameraList_from_camInfos(cam_infos, resolution_scales, args, resize_to_original=True):
    camera_list = []
    pry_up_camera_list = []
    for i in range(len(resolution_scales)):
        camera_list.append([])
        pry_up_camera_list.append([])
    for id, c in enumerate(cam_infos):
        cam_pyramid, cam_up_pyramid = loadCam(args, id, c, resolution_scales,resize_to_original=resize_to_original)
        for i in range(len(resolution_scales)):
            camera_list[i].append(cam_pyramid[i])
            pry_up_camera_list[i].append(cam_up_pyramid[i])

    return camera_list, pry_up_camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
