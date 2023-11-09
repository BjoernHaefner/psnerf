import json
import logging
import shutil
import sys
from logging import Logger
from os import listdir, makedirs, sep
from os.path import abspath, isdir, isfile, join

import cv2
import numpy as np
from PIL import Image
from imageio import imread

'''
cat dataset/convert.log | grep -E "near|far|Prepare"
'''

class Logging:
    rootLogger: Logger

    def __init__(self, filename='convert.log', mode='w'):
        '''

        Args:
            filename: filename of log file
            mode: 'w' (write), 'a' (append)
        '''
        self.filename = filename

        logFormatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        self.rootLogger = logging.getLogger()
        self.rootLogger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(self.filename, mode=mode)  # mode='w' to overwrite
        fileHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        self.rootLogger.addHandler(consoleHandler)
        return

    def getLogger(self):
        return self.rootLogger

    def getFilename(self):
        return self.filename


logging_ = Logging()
logger = logging_.getLogger()


class Convert2PSNeRF:
    '''Converts our data to the data format of PSNeRF'''
    _radius:float = 2.0

    def __init__(self, path2dataset: str):
        self.path2dataset = path2dataset
        self.datasets = {}
        logger.info(f''.center(100, "#"))
        logger.info(f'Find all datasets'.center(100, "#"))
        logger.info(f''.center(100, "#"))
        for name in sorted(listdir(self.path2dataset)):
            if name=="hard_2_views":
                continue
            if isdir(join(self.path2dataset, name)):
                logger.info(f'Found dataset: {abspath(join(self.path2dataset, name))}')
                self.datasets[name] = {'path': self.path2dataset,
                                       'name': name}
                dataset = self.datasets[name]
                data = listdir(join(dataset['path'], dataset['name']))
                for d in sorted(data):
                    if 'cameras' in d:
                        dataset['cameras'] = d
                    if 'image' in d:
                        dataset['images'] = {'name': d}
                    if 'directional_light' in d:
                        dataset['directional_lights'] = d
                    if 'mask' in d:
                        dataset['masks'] = {'name': d}
                logger.info(dataset)
        return

    def __call__(self, output_path: str, dry: bool = False, force: bool = True):
        for dataset in self.datasets.values():
            logger.info(f''.center(100, "#"))
            logger.info(f'Prepare {dataset["name"]}'.center(100, "#"))
            logger.info(f''.center(100, "#"))

            height, width = self._convert_images(dataset, dry, force, output_path)
            self._convert_masks(dataset, dry, force, output_path)
            self._build_params_json(dataset, dry, force, height, output_path, width)
        return

    def _convert_images(self, dataset, dry, force, output_path):
        logger.info(f'Convert images'.center(100, "#"))
        for vi, view in enumerate(self._list_files(
                join(dataset['path'], dataset['name'], dataset['images']['name'])), 1):
            imgs = []
            for img in self._list_files(
                    join(dataset['path'], dataset['name'], dataset['images']['name'], view)):
                img_name = join(dataset['path'], dataset['name'], dataset['images']['name'],
                                view, img)
                if 'ambient' in img:
                    logger.info(f"Read ambient: {img_name}")
                    ambient_image = imread(img_name).astype(np.float32) / 255.0
                else:
                    logger.info(f"Read PS+ambient: {img_name}")
                    imgs.append(imread(img_name).astype(np.float32) / 255.0)
            logger.info(f"Substract ambient image: I - I_ambient")
            imgs = np.stack(imgs, axis=0) - ambient_image  # [n, h, w, c]

            height, width = imgs.shape[1:3]

            filedir = join(output_path, dataset['name'], 'img', f'view_{vi:02d}')
            makedirs(filedir, exist_ok=True)
            for ii, img in enumerate(imgs, 1):
                # psnerf data structure
                img_out_path = join(filedir, f'{ii:03d}.png')
                if not isfile(img_out_path) or force:
                    logger.info(f'Write PS: {img_out_path} [{img.shape}]')
                    if not dry:
                        Image.fromarray((img * 255).astype(np.uint8)).save(img_out_path)
        return height, width

    def _convert_masks(self, dataset, dry, force, output_path):
        logger.info(f'Convert masks'.center(100, "#"))
        filedir = join(output_path, dataset['name'], 'mask')
        filedir_norm = join(output_path, dataset['name'], 'norm_mask')
        makedirs(filedir, exist_ok=True)
        makedirs(filedir_norm, exist_ok=True)
        for mi, mask in enumerate(self._list_files(
                join(dataset['path'], dataset['name'], dataset['masks']['name'])), 1):
            mask_name = join(dataset['path'], dataset['name'], dataset['masks']['name'], mask)

            # make sure mask is of size [H, W]
            logger.info(f"Read mask: {mask_name}")
            mask = imread(mask_name).astype(np.float32) / 255.0
            if mask.ndim < 2 or 3 < mask.ndim:
                raise NotImplementedError(f"mask of ndim = {mask.ndim} not yet handled")
            elif mask.ndim == 3:
                logger.info(
                    f"Convert mask from shape {mask.shape} to {mask.shape[:-1]} [averaging over axis=-1]")
                mask = mask.mean(axis=-1)

            assert mask.ndim == 2, f"mask still does not have ndim==2, but mask.ndim={mask.ndim}"

            mask_out_path = join(filedir, f'view_{mi:02d}.png')
            if not isfile(mask_out_path) or force:
                logger.info(f"Write: {mask_out_path} [{mask.shape}]")
                if not dry:
                    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_out_path)

            norm_mask_out_path = join(filedir_norm, f'view_{mi:02d}.png')
            if not isfile(norm_mask_out_path) or force:
                logger.info(f"Write: {norm_mask_out_path} [{mask.shape}]")
                if not dry:
                    Image.fromarray((mask * 255).astype(np.uint8)).save(norm_mask_out_path)
        return

    def _build_params_json(self, dataset, dry, force, height, output_path, width):
        logger.info(f'Build params.json'.center(100, "#"))
        n_view = len(
            self._list_files(join(dataset['path'], dataset['name'], dataset['masks']['name'])))
        logger.info(
            f"Add obj_name, n_view, imhw, gt_normal_world, view_train, view_test, light_is_same to params.json")
        params = {
            "obj_name": dataset['name'],
            "n_view": n_view,
            "imhw": [height, width],
            "gt_normal_world": True,
            "view_train": list(range(n_view)),
            "view_test": [0],
            "light_is_same": False,
        }

        # cameras
        logger.info(f"Read: {join(dataset['path'], dataset['name'], dataset['cameras'])}")
        cameras = np.load(join(dataset['path'], dataset['name'], dataset['cameras']))
        scale_mats = [cameras[f'scale_mat_{idx}'].astype(np.float32) for idx in range(n_view)]
        # world2camera projections
        w2c_projs = [cameras[f'world_mat_{idx}'].astype(np.float32) for idx in range(n_view)]
        Ks = []  # intrinsics
        c2w_mats = []  # camera2world matrices
        translations = []
        for vi, (scale_mat, w2c_proj) in enumerate(zip(scale_mats, w2c_projs)):
            logger.info(f"Compute K, R, t of view {vi}")
            P_w2c = (w2c_proj @ scale_mat)[:3, :4]
            K, c2w = self.__load_K_Rt_from_P(None, P_w2c)  # will already be inverted
            Ks.append(K[:3, :3])
            c2w_mats.append(c2w.tolist())
            translations.append(c2w[:3, -1])  # [3]
        logger.info(f"Attach pose_c2w and K to params.json")
        translations = np.stack(translations, axis=0)  # [n_view, 3]
        logger.info(f"near: {int(min(np.linalg.norm(translations, axis=-1)) - self._radius)}")
        logger.info(f"far: {int(max(np.linalg.norm(translations, axis=-1)) + self._radius + 1)}") # + 1 is for ceiling
        params['pose_c2w'] = c2w_mats
        params['K'] = np.array(Ks).mean(axis=0).tolist()

        # light
        if 'directional_lights' in dataset:
            logger.info(f"Found directional lights")
            lights = np.load(
                join(dataset['path'], dataset['name'], dataset['directional_lights']))
            lights_dir = lights['L_dir']  # [views, imgs, 3]
            lights_color = lights['L_color']  # [views, imgs, channels]
            lights_dir /= np.linalg.norm(lights_dir, axis=-1, keepdims=True)  # normalize
            lights_dir *= lights_color.mean(axis=-1, keepdims=True) # shape [num_views, num_imgs, 3]
            logger.info(f"Attach light_direction to params.json")
            params['light_direction'] = lights_dir.tolist()
        else:
            logger.info(f"DID NOT find directional lights: Use dummy lights for now")
            n_imgs = len(self._list_files(
                join(dataset['path'], dataset['name'], dataset['images']['name'], self._list_files(
                    join(dataset['path'], dataset['name'], dataset['images']['name']))[
                    0]))) - 1  # - 1 to neglect the ambient image
            lights_dir = np.ones((n_view, n_imgs, 3)) # shape [num_views, num_imgs, 3]
            logger.info(f"Attach dummy light_direction to params.json")
            params['light_direction'] = lights_dir.tolist()

        # write json file params.json
        filedir = join(output_path, dataset['name'])
        makedirs(filedir, exist_ok=True)
        params_file_out = join(filedir, 'params.json')
        if not isfile(params_file_out) or force:
            logger.info(f"Write: {params_file_out}")
            if not dry:
                with open(params_file_out, 'w') as f:
                    json.dump(params, f)
        return

    def _list_files(self, path: str):
        return sorted(listdir(abspath(path)))

    def __load_K_Rt_from_P(self, filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K / K[2, 2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        return intrinsics, pose


if __name__ == "__main__":
    path2dataset = abspath(
        join(sep, 'storage', 'group', 'cvpr', 'brahimi', 'multiview_PS_project', 'data',
             'PS_final_evaluation_dataset', 'photometric'))
    convert = Convert2PSNeRF(path2dataset)

    output_path = abspath('dataset')
    convert(output_path, dry=False, force=True)

    logger.info(f"mv {logging_.getFilename()} {join(output_path, 'convert.log')}")
    shutil.move(logging_.getFilename(), join(output_path, 'convert.log'))
