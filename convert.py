import json
import logging
import shutil
import sys
from os import listdir, sep
from os.path import abspath, isdir, isfile, join

import numpy as np
from PIL.Image import Image
from imageio import imread

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.INFO)

convert_log_file = 'convert.log'
fileHandler = logging.FileHandler(convert_log_file)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)


class Convert2PSNeRF:
    '''Converts our data to the data format of PSNeRF'''

    def __init__(self, path2dataset: str):
        self.path2dataset = path2dataset
        self.datasets = {}
        for name in sorted(listdir(self.path2dataset)):
            if isdir(join(self.path2dataset, name)):
                rootLogger.info(f'Found dataset: {abspath(join(self.path2dataset, name))}')
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
                rootLogger.info(dataset)
        return

    def __call__(self, output_path: str, dry: bool = True, force: bool = False):
        for dataset in self.datasets.values():
            rootLogger.info(f'Prepare {dataset["name"]}')
            # images
            for vi, view in enumerate(self._list_files(
                    join(dataset['path'], dataset['name'], dataset['images']['name'])), 1):
                imgs = []
                for img in self._list_files(
                        join(dataset['path'], dataset['name'], dataset['images']['name'], view)):
                    img_name = join(dataset['path'], dataset['name'], dataset['images']['name'],
                                    view, img)
                    if 'ambient' in img:
                        rootLogger.info(f"Read ambient: {img_name}")
                        ambient_image = imread(img_name).astype(np.float32) / 255.0
                    else:
                        rootLogger.info(f"Read PS+ambient: {img_name}")
                        imgs.append(imread(img_name).astype(np.float32) / 255.0)
                rootLogger.info(f"Substract ambient image")
                imgs = np.stack(imgs, axis=0) - ambient_image  # [n, h, w, c]
                height, width = imgs.shape[1:3]
                for ii, img in enumerate(imgs, 1):
                    # psnerf data structure
                    img_out_path = join(output_path,
                                        dataset['name'], 'img', f'view_{vi:02d}', f'{ii:03d}.png')
                    if not isfile(img_out_path) or force:
                        rootLogger.info(f'Write: {img_out_path}')
                        if not dry:
                            Image.fromarray(img).save(img_out_path)
            # masks
            for mi, mask in enumerate(self._list_files(
                    join(dataset['path'], dataset['name'], dataset['masks']['name'])), 1):
                mask_name = join(dataset['path'], dataset['name'], dataset['masks']['name'], mask)
                mask_out_path = join(output_path, dataset['name'], 'mask', f'view_{mi:02d}.png')
                if not isfile(mask_out_path) or force:
                    rootLogger.info(f"cp {mask_name} {mask_out_path}")
                    if not dry:
                        shutil.copy2(mask_name, mask_out_path)

                norm_mask_out_path = join(output_path, dataset['name'], 'norm_mask',
                                          f'view_{mi:02d}.png')
                if not isfile(norm_mask_out_path) or force:
                    rootLogger.info(f"cp {mask_name} {norm_mask_out_path}")
                    if not dry:
                        shutil.copy2(mask_name, norm_mask_out_path)
            n_view = len(
                self._list_files(join(dataset['path'], dataset['name'], dataset['masks']['name'])))
            params = {
                "obj_name": dataset['name'],
                "n_view": n_view,
                "imhw": [height, width],
                "gt_normal_world": None,
                "view_train": list(range(1, n_view + 1)),
                "view_test": list(range(1, n_view + 1)),
                "K": None,
                "pose_c2w": None,
                "light_is_same": False,
                "light_direction": None,
            }
            # cameras

            # light

            # write json file params.json
            if not isfile(norm_mask_out_path) or force:
                rootLogger.info(f"Write: {join(output_path, dataset['name'], 'params.json')}")
                if not dry:
                    with open(join(output_path, dataset['name'], 'params.json'), 'w') as f:
                        json.dump(params, f)
        return

    def _list_files(self, path: str):
        return sorted(listdir(abspath(path)))


if __name__ == "__main__":
    path2dataset = abspath(
        join(sep, 'storage', 'group', 'cvpr', 'brahimi', 'multiview_PS_project', 'data',
             'PS_final_evaluation_dataset', 'photometric'))
    convert = Convert2PSNeRF(path2dataset)

    output_path = abspath('dataset')
    convert(output_path)

# rootLogger.info(f"mv {convert_log_file} {join(output_path, 'convert.log')}")
# shutil.move(convert_log_file, join(output_path, 'convert.log'))
