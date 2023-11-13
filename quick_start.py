import os
import sys
from os.path import join
from subprocess import call
from typing import List, Tuple

from pyhocon import ConfigFactory

import stage1.dataloading as dl

'''
This script aims to simplify the process of running all steps and stages of psnerf.

Troubleshooting:
I had to follow this:
    https://github.com/conda-forge/pytorch_sparse-feedstock/issues/21#issuecomment-1056418260
    https://github.com/pytorch/pytorch/issues/69894
as the original repos environment.yaml file threw the error mentioned in above issue when running stage1/shape_extract.py
'''

file_dir = os.path.dirname(os.path.realpath(sys.argv[0]))  # directory where this file lies


def SAVE_CALL(cmd: str, args: List[str], goto: str, dry: bool = False):
    print(
        f"Call: {' '.join([f'cd {os.path.join(file_dir, goto)};', cmd, *args, ';', f'cd {file_dir}'])}",
        flush=True)
    if not dry:
        try:
            retcode = call(" ".join(
                [f'cd {os.path.join(file_dir, goto)};', cmd, *args, ";", f'cd {file_dir}']),
                shell=True)
            if retcode != 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)
    return


def preprocessing(obj_name: str, gpu_id: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    cmd = f"python test.py"
    args = [
        f"--retrain {join('data', 'models', 'LCNet_CVPR2019.pth.tar')}",
        f"--retrain_s2 {join('data', 'models', 'NENet_CVPR2019.pth.tar')}",
        f"--benchmark UPS_Custom_Dataset",
        f"--bm_dir {join('..', 'dataset', obj_name)}",
    ]
    SAVE_CALL(cmd, args, 'preprocessing')
    return


def light_avg(obj_name: str):
    cmd = f"python light_avg.py"
    args = [
        f"--obj {obj_name}",
        f"--path dataset",
        f"--light_intnorm",
        f"--sdps",
    ]
    SAVE_CALL(cmd, args, ".")
    return


def stage1(obj_name: str, gpu_id: int, calls: Tuple[str], gt_mesh: str = ""):
    cfg = dl.load_config(join("stage1", "configs", f"{obj_name}.yaml"))
    expfolder = cfg['training']['out_dir'].rstrip(os.sep).split(os.sep)[0]  # "out"
    expname = cfg['training']['out_dir'].rstrip(os.sep).split(os.sep)[-1]  # "test_1"

    def train():
        cmd = f"python train.py"
        args = [
            f"{join('configs', f'{obj_name}.yaml')}",
            f"--gpu {gpu_id}",
        ]
        SAVE_CALL(cmd, args, "stage1")
        return

    def shape_extract():
        cmd = f"python shape_extract.py"
        args = [
            f"--gpu {gpu_id}",
            f"--exp_folder {expfolder}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
            f"--visibility",
            f"--vis_plus",
        ]
        SAVE_CALL(cmd, args, "stage1")
        return

    def eval():
        cmd = f"python eval.py"
        args = [
            f"--gpu {gpu_id}",
            f"--exp_folder {expfolder}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
        ]
        SAVE_CALL(cmd, args, "stage1")
        return

    def extract_mesh():
        '''
        Run before:
        $ python setup.py build_ext --inplace
        '''
        cmd = f"python extract_mesh.py"
        args = [
            f"--gpu {gpu_id}",
            f"--exp_folder {expfolder}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
        ]
        SAVE_CALL(cmd, args, "stage1")
        return

    def chamfer_dist():
        cmd = f"python {join('..', 'chamfer_dist.py')}"
        args = [
            f"--mesh_gt {gt_mesh}",
            f"--mesh_pred PRED_MESH_PATH",  # todo: probably from previous routine extract_mesh
        ]
        SAVE_CALL(cmd, args, "stage1")
        return

    if "train" in calls:
        train()
    if "shape_extract" in calls:
        shape_extract()

    if "eval" in calls:
        eval()  # todo: test
    if "extract_mesh" in calls:
        extract_mesh()
    if "chamfer_dist" in calls and os.path.isfile(gt_mesh):
        chamfer_dist()  # todo: test
    return


def stage2(obj_name: str, gpu_id: int, calls: Tuple[str]):
    conf = ConfigFactory.parse_file(join("stage2", "confs", f"{obj_name}.conf"))
    expname = conf.get_string('train.expname')

    def train():
        cmd = f"python train.py"
        args = [
            f"--conf {join('confs', f'{obj_name}.conf')}",
            f"--gpu {gpu_id}",
            f"--is_continue",
        ]
        SAVE_CALL(cmd, args, "stage2")
        return

    def eval_envmap():
        # For rendering with environment lighting
        cmd = f"python eval.py"
        args = [
            f"--gpu {gpu_id}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
            f"--render_envmap",
        ]
        SAVE_CALL(cmd, args, "stage2")
        return

    def eval():
        # For material editing
        cmd = f"python eval.py"
        args = [
            f"--gpu {gpu_id}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
        ]
        SAVE_CALL(cmd, args, "stage2")
        return

    if "train" in calls:
        train()
    if "eval_envmap" in calls:
        eval_envmap()
    if "eval" in calls:
        eval()
    return


def evaluation(obj_name: str):
    conf = ConfigFactory.parse_file(join('stage2', 'confs', f'{obj_name}.conf'))
    expname = conf.get_string('train.expname')
    cmd = f"python evaluation.py"
    args = [
        f"--obj {obj_name}",
        f"--expname {expname}",
    ]
    SAVE_CALL(cmd, args, ".")
    return


def parser():
    import argparse
    parser = argparse.ArgumentParser(prog='PS-NeRF: Quick start',
                                     description='Executes the whole PS-NeRF pipeline')
    parser.add_argument('-n', '--obj_name', required=True, type=str, help='object name')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    return parser.parse_args()


def main(cli):
    preprocessing(cli.obj_name, cli.gpu_id)
    light_avg(cli.obj_name)
    stage1(cli.obj_name, cli.gpu_id, calls=("train"))
    stage1(cli.obj_name, cli.gpu_id, calls=("shape_extract", "extract_mesh"))
    stage2(cli.obj_name, cli.gpu_id, calls=("train"))
    # evaluation(cli.obj_name)
    return


if __name__ == "__main__":
    '''
    EG: python quick_start.py -n obj_name
    '''
    main(parser())
