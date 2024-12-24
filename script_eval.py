from argparse import ArgumentParser, Namespace
import sys
import os
import json
def read_file(path):
    with open(path, "r") as f:
        tp = f.read()
        lst = tp.split()
        return float(lst[0]), float(lst[1]), float(lst[2]), int(lst[3])
def run_script(dataset_path, save_path, device, out_path):

    dataset_name = dataset_path.split("/")[-1]
    dataset_log= open(os.path.join(out_path, f"{dataset_name}_eval.txt"), "w")
    total_psnr=0.0
    total_ssim=0.0
    total_lpips=0.0
    cnt = 0
    total_size = 0.0
    for name in os.listdir(dataset_path):
        print(name)
        data_path=os.path.join(dataset_path, name)
        cur_save_path=os.path.join(save_path, name)

        cmd=f"python render.py -s {data_path} -m {cur_save_path} --device {device} --eval --skip_train"
        print(cmd)
        os.system(cmd)
        cmd = f"python metrics.py -m {cur_save_path} --device {device}"
        print(cmd)
        os.system(cmd)
        cur_out_path=os.path.join(cur_save_path,"results.json")
        json_file = json.load(open(cur_out_path))
        cur_psnr = json_file["ours_30000"]["PSNR"]
        cur_ssim = json_file["ours_30000"]["SSIM"]
        cur_lpips = json_file["ours_30000"]["LPIPS"]
        cnt+=1
        total_lpips+=cur_lpips
        total_ssim+=cur_ssim
        total_psnr+=cur_psnr
        size = os.path.getsize(os.path.join(cur_save_path,"point_cloud","iteration_30000","point_cloud.ply"))
        cur_size = size/1024/1024
        total_size += cur_size
        dataset_log.write(f"{name} PSNR: {cur_psnr} SSIM: {cur_ssim} LPIPS: {cur_lpips} STORAGE: {cur_size}\n")

    dataset_log.write(f"Average PSNR: {total_psnr/cnt} SSIM: {total_ssim/cnt} LPIPS: {total_lpips/cnt} STORAGE: {total_size/cnt}")
    dataset_log.close()

parser=ArgumentParser(description="Script parameters")
parser.add_argument('--device', type=int, default=0)
parser.add_argument("--name", type=str, default="default")
parser.add_argument("--dataset",type=str,default="")
parser.add_argument("--save_path", type=str,default="", required=True)
parser.add_argument("--dataset_path", type=str,default="", required=True)
args = parser.parse_args(sys.argv[1:])
base_save_path=args.save_path
base_dataset_path=args.dataset_path
out_path = os.path.join(base_save_path, args.name)
os.makedirs(out_path, exist_ok=True)
if args.dataset=="db" or args.dataset=="":
    run_script(os.path.join(base_dataset_path,"db"), os.path.join(base_save_path, args.name, "db"), args.device, out_path)
if args.dataset=="tt" or args.dataset=="":
    run_script(os.path.join(base_dataset_path,"tandt"), os.path.join(base_save_path, args.name, "tandt"), args.device, out_path)
if args.dataset=="mip" or args.dataset=="":
    run_script(os.path.join(base_dataset_path,"MipNeRF"), os.path.join(base_save_path, args.name, "MipNeRF"), args.device, out_path)