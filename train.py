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
import numpy as np

import torch
import torch.backends
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from time import time
# torch.set_num_threads(32)

def savefiles(path):
    path=os.path.join(path, "backup")
    os.makedirs(path, exist_ok=True)
    shutil.copy("arguments/__init__.py", path)
    shutil.copy("scene/gaussian_model.py", path)
    #shutil.copyfile("scene/__init__.py", os.path.join(path, "scene"))
    shutil.copy("train.py", path)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    return tb_writer

def loss_fn(image, gt_image, lambda_dssim):
    Ll1 = l1_loss(image, gt_image)
    ssim_loss = (1.0 - ssim(image, gt_image))
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * ssim_loss
    return Ll1, ssim_loss, loss

def get_change_split_iter (last_change_iter, next_change_iter, stage_split):
    return (next_change_iter - last_change_iter) // stage_split

def training(dataset, opt:OptimizationParams, pipe, testing_iterations, saving_iterations, logger=None):
    first_iter = 0
    lpips_fn = lpips.LPIPS(net='vgg').to("cuda")
    _ = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    #max_level=len(dataset.feat_dim)
    scene = Scene(dataset, gaussians, shuffle=False, resolution_scales=opt.resolution_scales, resize_to_orig=opt.resize_to_original)
    gaussians.training_setup(opt)
    max_level = len(opt.resolution_scales)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    change_iter = opt.change_iter
    warm_up_iter = 0
    update_value = opt.stage_hr_factor**(1.0/opt.stage_split)
    cur_level = 1
    last_change_iter = opt.update_from
    next_change_iter = change_iter[0]
    change_split_level_iter = get_change_split_iter(last_change_iter, next_change_iter, opt.stage_split)


    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            net_image_bytes = None
            custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            if custom_cam != None:
                net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer)["render"]
                net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            network_gui.send(net_image_bytes, dataset.source_path)
            #print("sent web image")
            if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                break
            
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        cur_pop_id=randint(0, len(viewpoint_stack)-1)

        # Render
        is_densify_iter = (iteration > opt.update_from and iteration % opt.update_interval == 0)
        viewpoint_cam=viewpoint_stack.pop(cur_pop_id)

        render_pkg=render(viewpoint_cam, gaussians, pipe, background)
        image,viewspace_point_tensor,visibility_filter=render_pkg["render"],render_pkg["viewspace_points"],render_pkg["visibility_filter"]
        gt_image=viewpoint_cam.original_image.cuda()


        # Calculate loss
        Ll1, ssim_loss, image_loss=loss_fn(image, gt_image, opt.lambda_dssim)
        loss=image_loss
        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, None, l1_loss, testing_iterations, scene, render, (pipe, background), logger, lpips_fn)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)



            # densification
            if iteration < opt.update_until and iteration > opt.start_stat and (warm_up_iter==0):
                # add statis
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # densification
                if is_densify_iter:
                    gaussians.adjust_gaussian(opt.densify_grad_threshold, update_value,opt.min_opacity, cur_stage=cur_level
                                              ,opacity_reduce_weight=opt.opacity_reduce_weight, residual_split_scale_div=opt.residual_split_scale_div)


            if iteration in change_iter:
                scene.up_one_resolution()
                warm_up_iter=opt.warm_up_iter
                if scene.cur_resolution is not max_level-1:
                    next_change_iter = change_iter[scene.cur_resolution]
                    last_change_iter = change_iter[scene.cur_resolution-1] + warm_up_iter
                else:
                    next_change_iter = opt.update_until
                    last_change_iter = change_iter[scene.cur_resolution-1] + warm_up_iter
                cur_level = opt.stage_split*(scene.cur_resolution) + 1
                change_split_level_iter = get_change_split_iter(last_change_iter, next_change_iter, opt.stage_split)
                viewpoint_stack = scene.getTrainCameras().copy()
                scene.clear_image()

            if  scene.cur_resolution!=0 and opt.use_opacity_reduce and iteration < opt.prune_until:
                if iteration % opt.opacity_reduce_interval == 0:
                    if iteration >= opt.update_until:
                        gaussians.prune_opacity(opt.min_opacity)
                    gaussians.reduce_opacity()

            elif iteration == opt.update_until:
                del gaussians.xyz_gradient_accum
                del gaussians.denom
                del gaussians.xyz_gradient_accum_abs
                torch.cuda.empty_cache()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if warm_up_iter is not 0:
                warm_up_iter-=1
            if change_split_level_iter is not 0 and warm_up_iter is 0 and iteration > opt.update_from and iteration < opt.update_until:
                change_split_level_iter-=1
                if change_split_level_iter is 0:
                    cur_level+=1
                    change_split_level_iter = get_change_split_iter(last_change_iter, next_change_iter, opt.stage_split)


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

def training_report(iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, renderArgs, logger, lpips_fn):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test=0.0
                lpips_test=0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_fn(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
        torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--device",type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)


    model_path = args.model_path
    savefiles(model_path)
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    #os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    logger.info(f'args: {args}')


    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]


    logger.info("Optimizing " + args.model_path)

    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    torch.cuda.set_device(args.device)
    # training
    lpargs=lp.extract(args)

    opargs = op.extract(args)

    training(lpargs,opargs, pp.extract(args), args.test_iterations, args.save_iterations, logger)