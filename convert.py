import json
import logging
import shutil
import sys
from collections import defaultdict
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
    _radius: float = 2.0
    psnerf_datasets = defaultdict(lambda: {
        "images": None,  # [V, N, H, W, 3]
        "masks": None,  # [V, H, W]
        "norm_masks": None,  # [V, H, W]
        "obj_name": None,
        "n_view": None,
        "imhw": None,  # [H, W]
        "gt_normal_world": None,
        "view_train": None,
        "view_test": None,
        "light_is_same": None,
        "pose_c2w": None,  # [V, 4, 4]
        "K": None,  # [3,3]
        "light_direction": None,
        # [N, 3] if light_is_same==True else {[V, N, 3] or [V*N, 3] not sure: https://github.com/ywq/psnerf/issues/8}
    })

    def __init__(self):
        return

    def save(self, name: str, output_path: str, dry: bool = False, force: bool = True):
        logger.info(f''.center(100, "#"))
        logger.info(f'Save {name}'.center(100, "#"))
        logger.info(f''.center(100, "#"))

        self._validate(self.psnerf_datasets[name])
        self._save_images(name, self.psnerf_datasets[name], output_path, dry, force)
        self._save_masks(name, self.psnerf_datasets[name], output_path, dry, force)
        self._save_params_json(name, self.psnerf_datasets[name], output_path, dry, force)
        return

    def _validate(self, dataset):
        '''
        Checks if the provided data are of correct shape and data type
        Still some open questions as mentioned here: https://github.com/ywq/psnerf/issues/8
        '''

        # check data types
        assert type(dataset['images']) == np.ndarray and dataset['images'].dtype == np.float32
        assert type(dataset['masks']) == np.ndarray and dataset['masks'].dtype == bool
        assert type(dataset['norm_masks']) == np.ndarray and dataset['norm_masks'].dtype == bool
        assert type(dataset['obj_name']) == str
        assert type(dataset['n_view']) == int
        assert type(dataset['imhw']) == list
        assert type(dataset['gt_normal_world']) == bool
        assert type(dataset['view_train']) == list
        assert type(dataset['view_test']) == list
        assert type(dataset['light_is_same']) == bool
        assert type(dataset['pose_c2w']) == np.ndarray and dataset['pose_c2w'].dtype == np.float32
        assert type(dataset['K']) == np.ndarray and dataset['K'].dtype == np.float64
        assert type(dataset['light_direction']) == np.ndarray and dataset[
            'light_direction'].dtype == np.float64

        # Check shapes, sizes, and values
        v, n, h, w, c = dataset['images'].shape
        assert (0 <= dataset['images']).all and (dataset['images'] <= 1).all()
        assert (v, h, w) == dataset['masks'].shape
        assert (v, h, w) == dataset['norm_masks'].shape
        assert [h, w] == dataset['imhw']
        assert (v, 4, 4) == dataset['pose_c2w'].shape
        assert (3, 3) == dataset['K'].shape

        for vi in dataset['view_train']:
            assert vi < dataset['n_view']
        for vi in dataset['view_test']:
            assert vi < dataset['n_view']

        if dataset['light_is_same']:
            assert (n, 3) == dataset['light_direction'].shape
        else:
            assert (v, n, 3) == dataset['light_direction'].shape

    def _save_images(self, name, dataset, output_path, dry, force):
        '''psnerf data structure: eg obj_name/img/view_01/001.png'''
        logger.info(f'Save images'.center(100, "#"))
        images = dataset['images']  # [V, N, H, W, 3]
        for vi, view in enumerate(images, 1):  # iterate over views
            filedir = join(output_path, name, 'img', f'view_{vi:02d}')
            makedirs(filedir, exist_ok=True)
            for ii, img in enumerate(view, 1):  # iterate over images
                # psnerf data structure
                img_out_path = join(filedir, f'{ii:03d}.png')
                if not isfile(img_out_path) or force:
                    logger.info(f'Write PS [8-bit]: {img_out_path} [{img.shape}]')
                    if not dry:
                        Image.fromarray((img * 255).astype(np.uint8)).save(img_out_path)
        return

    def _save_masks(self, name, dataset, output_path, dry, force):
        def __convert_masks(mask_name: str):
            filedir = join(output_path, name, mask_name)
            makedirs(filedir, exist_ok=True)
            masks = dataset['masks']  # [V, H, W]
            for mi, mask in enumerate(masks, 1):
                mask_path = join(filedir, f'view_{mi:02d}.png')
                if not isfile(mask_path) or force:
                    logger.info(f"Write: {mask_path} [{mask.shape}]")
                    if not dry:
                        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

            return

        logger.info(f'Save masks'.center(100, "#"))
        __convert_masks('mask')
        __convert_masks('norm_mask')
        return

    def _save_params_json(self, name, dataset, output_path, dry, force):
        logger.info(f'Save params.json'.center(100, "#"))
        params = {
            "obj_name": dataset['obj_name'],
            "n_view": dataset['n_view'],
            "imhw": dataset['imhw'],
            "gt_normal_world": dataset['gt_normal_world'],
            "view_train": dataset['view_train'],
            "view_test": dataset['view_test'],
            "light_is_same": dataset['light_is_same'],
        }
        for k in sorted(params.keys()):
            logger.info(f"\t{k}: {params[k]}")

        # cameras
        params['pose_c2w'] = dataset['pose_c2w'].tolist()
        logger.info(f"\t'pose_c2w': {dataset['pose_c2w'].shape}")
        params['K'] = dataset['K'].tolist()
        logger.info(f"\t'K': {dataset['K'].shape}")
        params['light_direction'] = dataset['light_direction'].tolist()
        logger.info(f"\t'light_direction': {dataset['light_direction'].shape}")

        # write json file params.json
        filedir = join(output_path, name)
        makedirs(filedir, exist_ok=True)
        params_file_out = join(filedir, 'params.json')
        if not isfile(params_file_out) or force:
            logger.info(f"Write: {params_file_out}")
            if not dry:
                with open(params_file_out, 'w') as f:
                    json.dump(params, f)
        return

    def add2dataset(self, name, attribute, value):
        self.psnerf_datasets[name][attribute] = value
        return

    def _list_files_sorted(self, path: str):
        return sorted(listdir(abspath(path)))


class Ours2PSNeRF(Convert2PSNeRF):
    _bit_depth: int = 16.

    def __init__(self, path2dataset: str):
        super(Ours2PSNeRF, self).__init__()
        self.path2dataset = path2dataset
        self.datasets = {}
        logger.info(f''.center(100, "#"))
        logger.info(f'Find all datasets'.center(100, "#"))
        logger.info(f''.center(100, "#"))
        for name in sorted(listdir(self.path2dataset)):
            if name == "hard_2_views":
                continue
            if isdir(join(self.path2dataset, name)):
                logger.info(f'Found dataset: {abspath(join(self.path2dataset, name))}')
                self.datasets[name] = {'path': self.path2dataset, 'name': name}
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

    def get_dataset_names(self):
        return sorted(self.datasets.keys())

    def load_and_convert(self, name: str):
        logger.info(f''.center(100, "#"))
        logger.info(f'Load and convert {name}'.center(100, "#"))
        logger.info(f''.center(100, "#"))

        self._load_and_convert_images(self.datasets[name])
        self._load_and_convert_masks(self.datasets[name])
        self._build_params(self.datasets[name])
        return

    def _load_and_convert_images(self, dataset: dict):
        logger.info(f'Load and convert images'.center(100, "#"))
        dataset_name = dataset['name']
        path2images = join(dataset['path'], dataset_name, dataset['images']['name'])
        images = []  # images for all views
        for view_name in self._list_files_sorted(path2images):
            imgs_v = []  # images for a single view
            for img_name in self._list_files_sorted(join(path2images, view_name)):
                path2image = join(path2images, view_name, img_name)
                if 'ambient' in img_name:
                    logger.info(f"Read ambient [{int(self._bit_depth)}-bit]: {path2image}")
                    ambient_image = cv2.imread(path2image, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    ambient_image /= 2 ** self._bit_depth - 1
                else:
                    logger.info(f"Read PS+ambient [{int(self._bit_depth)}-bit]: {path2image}")
                    img = cv2.imread(path2image, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    img /= 2 ** self._bit_depth - 1
                    imgs_v.append(img)
            logger.info(f"Substract ambient image: min(1, max(0, I - I_ambient))")
            imgs_v = (np.stack(imgs_v, axis=0) - ambient_image).clip(min=0, max=1)  # [n, h, w, c]
            images.append(imgs_v)

        images = np.stack(images, axis=0)  # [v, n, h, w, c]
        self.add2dataset(dataset_name, 'images', images)
        logger.info(f'Images loaded: {images.shape} [{images.dtype}]')
        return

    def _load_and_convert_masks(self, dataset: dict):
        logger.info(f'Load and convert masks'.center(100, "#"))
        dataset_name = dataset['name']
        path2masks = join(dataset['path'], dataset_name, dataset['masks']['name'])
        masks = []
        norm_masks = []
        for mask_name in self._list_files_sorted(path2masks):
            path2mask = join(path2masks, mask_name)

            logger.info(f"Read mask: {path2mask}")
            mask = imread(path2mask).astype(np.float32) / 255.0

            # make sure mask is of size [H, W]
            if mask.ndim < 2 or 3 < mask.ndim:
                raise NotImplementedError(f"mask of ndim = {mask.ndim} not yet handled")
            elif mask.ndim == 3:
                logger.info(
                    f"Convert mask from shape {mask.shape} to {mask.shape[:-1]} [averaging over axis=-1]")
                mask = mask.mean(axis=-1)
            masks.append(mask.astype(bool))
            norm_masks.append(mask.astype(bool))

        masks = np.stack(masks, axis=0)  # [v, h, w]
        self.add2dataset(dataset_name, 'masks', masks)
        logger.info(f'Masks loaded: {masks.shape} [{masks.dtype}]')

        norm_masks = np.stack(norm_masks, axis=0)  # [v, h, w]
        self.add2dataset(dataset_name, 'norm_masks', norm_masks)
        logger.info(f'Norm_masks loaded: {norm_masks.shape} [{norm_masks.dtype}]')
        return

    def _build_params(self, dataset: dict):
        logger.info(f'Build params.json'.center(100, "#"))
        images = self.psnerf_datasets[dataset['name']]['images']
        n_view = images.shape[0]
        self.add2dataset(dataset['name'], 'obj_name', dataset['name'])
        self.add2dataset(dataset['name'], 'n_view', n_view)
        self.add2dataset(dataset['name'], 'imhw', list(images.shape[2:4]))  # [h, w]
        self.add2dataset(dataset['name'], 'gt_normal_world', True)
        self.add2dataset(dataset['name'], 'view_train', list(range(n_view)))
        self.add2dataset(dataset['name'], 'view_test', [0])
        self.add2dataset(dataset['name'], 'light_is_same', False)
        logger.info(f"Build params:")
        logger.info(f"\tobj_name: {self.psnerf_datasets[dataset['name']]['obj_name']}")
        logger.info(f"\tn_view: {self.psnerf_datasets[dataset['name']]['n_view']}")
        logger.info(f"\timhw: {self.psnerf_datasets[dataset['name']]['imhw']}")
        logger.info(
            f"\tgt_normal_world: {self.psnerf_datasets[dataset['name']]['gt_normal_world']}")
        logger.info(f"\tview_train: {self.psnerf_datasets[dataset['name']]['view_train']}")
        logger.info(f"\tview_test: {self.psnerf_datasets[dataset['name']]['view_test']}")
        logger.info(f"\tlight_is_same: {self.psnerf_datasets[dataset['name']]['light_is_same']}")

        # cameras
        path2cameras = join(dataset['path'], dataset['name'], dataset['cameras'])
        logger.info(f"Read: {path2cameras}")
        cameras = np.load(path2cameras)
        scale_mats = [cameras[f'scale_mat_{idx}'].astype(np.float32) for idx in range(n_view)]
        # world2camera projections
        w2c_projs = [cameras[f'world_mat_{idx}'].astype(np.float32) for idx in range(n_view)]
        Ks = []  # intrinsics
        c2w_mats = []  # camera2world matrices
        translations = []
        for vi, (scale_mat, w2c_proj) in enumerate(zip(scale_mats, w2c_projs)):
            logger.info(f"Compute K, R, t of view {vi} in OpenGL format")
            P_w2c = (w2c_proj @ scale_mat)[:3, :4]
            K, c2w = self._load_K_Rt_from_P(None, P_w2c)  # will already be inverted
            # OpenCV to OpenGL: https://github.com/ywq/psnerf/blob/96217612eb17975e82e9caaee294714a00c7f7db/stage1/dataloading/dataset.py#L54-L56
            c2w[:3, 1:3] *= -1
            Ks.append(K[:3, :3])
            c2w_mats.append(c2w)
            translations.append(c2w[:3, -1])  # [3]
        pose_c2w = np.stack(c2w_mats, axis=0)
        self.add2dataset(dataset['name'], 'pose_c2w', pose_c2w)
        logger.info(f'Poses loaded: {pose_c2w.shape} [{pose_c2w.dtype}]')
        K = np.stack(Ks, axis=0).mean(axis=0)
        self.add2dataset(dataset['name'], 'K', K)
        logger.info(f'Intrinsics loaded: {K.shape} [{K.dtype}]')

        logger.info(f"Compute near and far value (has to be put  in stage1/configs/*.yaml): ")
        translations = np.stack(translations, axis=0)  # [n_view, 3]
        logger.info(f"\tnear = int(min(norm(translations)) - self._radius) = "
                    f"{int(min(np.linalg.norm(translations, axis=-1)) - self._radius)}")
        logger.info(f"\tfar = int(max(norm(translations)) + self._radius + 1) = "
                    f"{int(max(np.linalg.norm(translations, axis=-1)) + self._radius + 1)} "
                    f"[+ 1 is for ceiling]")  # + 1 is for ceiling

        # light
        if 'directional_lights' in dataset:
            logger.info(f"Found directional lights")
            path2lights = join(dataset['path'], dataset['name'], dataset['directional_lights'])
            lights = np.load(path2lights)
            light_direction = lights['L_dir']  # [views, imgs, 3]
            lights_color = lights['L_color']  # [views, imgs, channels]
            light_direction /= np.linalg.norm(light_direction, axis=-1, keepdims=True)  # normalize
            light_direction *= lights_color.mean(axis=-1,
                                                 keepdims=True)  # shape [num_views, num_imgs, 3]
            self.add2dataset(dataset['name'], 'light_direction', light_direction)
            logger.info(
                f'Directional lights loaded: {light_direction.shape} [{light_direction.dtype}]')
        else:
            logger.info(f"DID NOT find directional lights: Use dummy lights for now")
            n_imgs = images.shape[1]
            light_direction = np.ones((n_view, n_imgs, 3))
            self.add2dataset(dataset['name'], 'light_direction', light_direction)
            logger.info(
                f'Dummy lights: {light_direction.shape} [{light_direction.dtype}]')
        return

    def _load_K_Rt_from_P(self, filename, P=None):
        '''all credits to: https://github.com/lioryariv/volsdf/blob/a974c883eb70af666d8b4374e771d76930c806f3/code/utils/rend_util.py#L31'''
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
    output_path = abspath('dataset')
    ours2psnerf = Ours2PSNeRF(path2dataset)
    
    for name in ours2psnerf.get_dataset_names():
        ours2psnerf.load_and_convert(name)
        ours2psnerf.save(name, output_path, dry=True, force=True)

    logger.info(f"mv {logging_.getFilename()} {join(output_path, 'convert.log')}")
    shutil.move(logging_.getFilename(), join(output_path, 'convert.log'))
