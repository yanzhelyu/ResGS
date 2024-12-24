from argparse import ArgumentParser, Namespace
import sys
import os
import time
def run_script(dataset_path, save_path, model_name, device, args):
    mip_outdoor = ["bicycle", "flowers", "garden", "stump", "treehill"]
    mip_indoor = ["room", "counter", "kitchen", "bonsai"]

    dataset_name = dataset_path.split("/")[-1]

    for name in os.listdir(dataset_path):
    #check if the path is a directory
        print(name)
        data_path=os.path.join(dataset_path, name)
        cur_save_path=os.path.join(save_path, name)
        image_args = ""
        if name in mip_outdoor and dataset_name == "MipNeRF":
            image_args = "-i images_4"
        if name in mip_indoor and dataset_name == "MipNeRF":
            image_args = "-i images_2"
        cmd=f"python train.py -s {data_path} -m {cur_save_path} --eval --device {device} {args} {image_args}"
        print(cmd)
        os.system(cmd)


parser=ArgumentParser(description="Script parameters")
parser.add_argument('--device', type=int, default=0)
parser.add_argument("--name", type=str, default="default")
parser.add_argument("--args", type=str, default="")
parser.add_argument("--dataset", type=str, default="")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--save_path", type=str,default="trained/")
parser.add_argument("--dataset_path", type=str,default="data/")
args = parser.parse_args(sys.argv[1:])
base_save_path=args.save_path
base_dataset_path=args.dataset_path

if args.dataset=="tt" or args.dataset=="":
    run_script(os.path.join(base_dataset_path,"tandt"), os.path.join(base_save_path, args.name, "tandt"), args.name, args.device, args.args)
    if args.eval:
        os.system(f"python script_eval.py --device {args.device} --name {args.name} --save_path {args.save_path} --dataset_path {args.dataset_path} --dataset tt")

if args.dataset=="db" or args.dataset=="":
    run_script(os.path.join(base_dataset_path,"db"), os.path.join(base_save_path, args.name, "db"), args.name, args.device, args.args)
    if args.eval:
        os.system(f"python script_eval.py --device {args.device} --name {args.name} --save_path {args.save_path} --dataset_path {args.dataset_path} --dataset db")

if args.dataset=="mip" or args.dataset=="":
    run_script(os.path.join(base_dataset_path,"MipNeRF"), os.path.join(base_save_path, args.name, "MipNeRF"), args.name, args.device, args.args)
    if args.eval:
        os.system(f"python script_eval.py --device {args.device} --name {args.name} --save_path {args.save_path} --dataset_path {args.dataset_path} --dataset mip")