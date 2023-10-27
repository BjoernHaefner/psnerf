import os
import sys
from subprocess import call
from typing import List


def SAVE_CALL(cmd: str, args: List[str], dry: bool = False):
    print(f"Call: {' '.join([cmd, *args])}")
    if not dry:
        try:
            retcode = call(" ".join([cmd, *args]), shell=True)
            if retcode < 0:
                print("Child was terminated by signal", -retcode, file=sys.stderr)
            else:
                print("Child returned", retcode, file=sys.stderr)
        except OSError as e:
            print("Execution failed:", e, file=sys.stderr)
    return


def preprocessing(obj_name: str):
    cmd = f"python preprocessing/test.py"
    args = [
        f"--retrain preprocessing/data/models/LCNet_CVPR2019.pth.tar",
        f"--retrain_s2 preprocessing/data/models/NENet_CVPR2019.pth.tar",
        f"--benchmark UPS_Custom_Dataset",
        f"--bm_dir dataset/{obj_name}",
    ]
    SAVE_CALL(cmd, args)
    return


def light_avg(obj_name: str):
    cmd = f"python light_avg.py"
    args = [
        f"--obj {obj_name}",
        f"--path dataset",
    ]
    SAVE_CALL(cmd, args)
    return


def stage1(obj_name: str, expname: str, gpu_id: int, test: bool = False, gt_mesh: str = ""):
    def train():
        cmd = f"python stage1/train.py"
        args = [
            f"stage1/configs/{obj_name}.yaml",
            f"--gpu {gpu_id}",
        ]
        SAVE_CALL(cmd, args)
        return

    def shape_extract():
        cmd = f"python stage1/shape_extract.py"
        args = [
            f"--gpu {gpu_id}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
            f"--visibility",
            f"--vis_plus",
        ]
        SAVE_CALL(cmd, args)
        return

    train()
    shape_extract()

    if test:
        '''
        Run before: 
        $ python setup.py build_ext --inplace
        '''

        def eval():
            cmd = f"python stage1/eval.py"
            args = [
                f"--gpu {gpu_id}",
                f"--obj_name {obj_name}",
                f"--expname {expname}",
            ]
            SAVE_CALL(cmd, args)
            return

        def extract_mesh():
            cmd = f"python stage1/extract_mesh.py"
            args = [
                f"--gpu {gpu_id}",
                f"--obj_name {obj_name}",
                f"--expname {expname}",
            ]
            SAVE_CALL(cmd, args)
            return

        def chamfer_dist():
            cmd = f"python chamfer_dist.py"
            args = [
                f"--mesh_gt {gt_mesh}",
                f"--mesh_pred PRED_MESH_PATH",
            ]
            SAVE_CALL(cmd, args)
            return

        eval()
        extract_mesh()
        if os.path.isfile(gt_mesh):
            chamfer_dist()
    return


def stage2(obj_name: str, expname: str, gpu_id: int):
    def train():
        cmd = f"python stage2/train.py"
        args = [
            f"--conf stage2/confs/{obj_name}.conf",
            f"--gpu {gpu_id}",
        ]
        SAVE_CALL(cmd, args)
        return

    def eval_envmap():
        # For rendering with environment lighting
        cmd = f"python stage2/eval.py"
        args = [
            f"--gpu {gpu_id}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
            f"--render_envmap",
        ]
        SAVE_CALL(cmd, args)
        return

    def eval_material():
        # For material editing
        cmd = f"python stage2/eval.py"
        args = [
            f"--gpu {gpu_id}",
            f"--obj_name {obj_name}",
            f"--expname {expname}",
        ]
        SAVE_CALL(cmd, args)
        return

    train()
    eval_envmap()
    eval_material()
    return


def evaluation(obj_name: str, expname: str, test_out_dir: str):
    cmd = f"python evaluation.py"
    args = [
        f"--obj {obj_name}",
        f"--expname {expname}",
        f"--test_out_dir {test_out_dir}",
    ]
    SAVE_CALL(cmd, args)
    return


def parser():
    import argparse
    parser = argparse.ArgumentParser(prog='PS-NeRF: Quick start',
                                     description='Executes the whole PS-NeRF pipeline')
    parser.add_argument('-n', '--obj_name', type=str, help='object name')
    parser.add_argument('-g', '--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('-e', '--expname', type=str, help='Experiment name')
    parser.add_argument('-o', '--test_out_dir', type=str, help='Output directory')
    return parser.parse_args(['-n', 'bear',
                              '-e', 'initial_test_bear',
                              '-o', 'initial_test_bear_out',
                              ])


def main(cli):
    preprocessing(cli.obj_name)
    light_avg(cli.obj_name)
    stage1(cli.obj_name, cli.expname, cli.gpu_id)
    stage2(cli.obj_name, cli.expname, cli.gpu_id)
    evaluation(cli.obj_name, cli.expname, cli.test_out_dir)
    return


if __name__ == "__main__":
    main(parser())
