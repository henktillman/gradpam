import argparse
import os
import pprint
import random
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

from collections import Sequence
import cv2
import random
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval
from utils.gradcam import GradPAM, GradPAMWhole, SegNormGrad, SegNormGradWhole, \
    GradCAM, SegGradCAM
from utils.modelsummary import get_model_summary
from utils.utils import create_logger
from collections import defaultdict

METHODS = {
    'GradPAM': GradPAM,
    'GradPAMWhole': GradPAMWhole,
    'SegNormGrad': SegNormGrad,
    'SegNormGradWhole': SegNormGradWhole,
    'GradCAM': GradCAM,
    'SegGradCAM': SegGradCAM
}

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize GradCAM')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--vis-mode', type=str, default='GradCAM',
                        choices=METHODS.keys(),
                        help='Type of gradient visualization')
    parser.add_argument('--image-index', type=int, default=0,
                        help='Index of input image for GradCAM')
    parser.add_argument('--image-index-range', type=int, nargs=3,
                        help='Expects [start, end) and step.')
    parser.add_argument('--crop-size', type=int, default=-1,
                        help='Size of crop around the center pixel')
    parser.add_argument('--pixel-max-num-random', type=int, default=10,
                        help='Maximum number of pixels to randomly sample from '
                        'an image, when fetching all predictions for a class.')
    parser.add_argument('--pixel-i', type=int, default=0, nargs='*',
                        help='i coordinate of pixel from which to compute GradCAM')
    parser.add_argument('--pixel-j', type=int, default=0, nargs='*',
                        help='j coordinate of pixel from which to compute GradCAM')
    parser.add_argument('--pixel-i-range', type=int, nargs=3,
                        help='Range for pixel i. Expects [start, end) and step.')
    parser.add_argument('--pixel-j-range', type=int, nargs=3,
                        help='Range for pixel j. Expects [start, end) and step.')
    parser.add_argument('--pixel-cartesian-product', action='store_true',
                        help='Compute cartesian product between all is and js '
                             'for the full list of pixels.')
    parser.add_argument('--suffix', default='',
                        help='Appended to each image filename.')
    parser.add_argument('--target-layers', type=str,
                        help='List of target layers from which to compute GradCAM')
    parser.add_argument('--nbdt-node-wnids-for', type=str,
                        help='Class NAME. Automatically computes nodes leading '
                             'up to particular class leaf.')
    parser.add_argument('--crop-for', type=str,
                        help='Class to crop for')
    parser.add_argument('--nbdt-node-wnid', type=str, default='', nargs='*',
                        help='WNID of NBDT node from which to compute output logits')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--skip-save-npy', action='store_true',
                        help="Don't save the npy file.")

    args = parser.parse_args()
    update_config(config, args)

    return args

class_names = [
    'road', 'sidewalk', 'building', 'wall', 'fence', \
    'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
    'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
    'motorcycle', 'bicycle'
]

def get_pixels(pixel_i, pixel_j, pixel_i_range, pixel_j_range, cartesian_product):
    assert not (pixel_i and pixel_i_range), \
        'Can only specify list of numbers (--pixel-i) OR a range (--pixel-i-range)'
    pixel_is = pixel_i or range(*pixel_i_range)

    assert not (pixel_j and pixel_j_range), \
        'Can only specify list of numbers (--pixel-j) OR a range (--pixel-j-range)'
    pixel_js = pixel_j or range(*pixel_j_range)

    if cartesian_product:
        return sum([ [(i, j) for i in pixel_is] for j in pixel_js ], [])
    return list(zip(pixel_is, pixel_js))

def compute_output_coord(pixel_i, pixel_j, image_shape, output_shape):
    ratio_i, ratio_j = output_shape[0]/image_shape[0], output_shape[1]/image_shape[1]
    out_pixel_i = int(np.floor(pixel_i * ratio_i))
    out_pixel_j = int(np.floor(pixel_j * ratio_j))
    return out_pixel_i, out_pixel_j

def retrieve_raw_image(dataset, index):
    item = dataset.files[index]
    image = cv2.imread(os.path.join(dataset.root,'cityscapes',item["img"]),
                       cv2.IMREAD_COLOR)
    return image

def save_gradcam(save_path, gradcam, raw_image, paper_cmap=False,
        minimum=None, maximum=None, save_npy=True):
    gradcam = gradcam.cpu().numpy()
    np_save_path = save_path.replace('.jpg', '.npy')
    if save_npy:
        np.save(np_save_path, gradcam)
    gradcam = GradCAM.normalize_np(gradcam, minimum=minimum, maximum=maximum)[0,0]
    cmap = cm.hot(gradcam)[..., 2::-1] * 255.0
    if paper_cmap:
        alpha = gradcam[..., None]
        gradcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gradcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(save_path, np.uint8(gradcam), [cv2.IMWRITE_JPEG_QUALITY, 50])

def generate_output_dir(output_dir, vis_mode, target_layer, use_nbdt,
        nbdt_node_wnid, crop_size=0, cls=None):
    vis_mode = vis_mode.lower()
    target_layer = target_layer.replace('model.', '')

    dir = os.path.join(output_dir, f'{vis_mode}_{target_layer}')
    if use_nbdt:
        dir += f'_{nbdt_node_wnid}'
    if cls:
        dir += f'_cls{cls}'
    if crop_size > 0:
        dir += f'_crop{crop_size}'
    os.makedirs(dir, exist_ok=True)
    return dir

def generate_save_path(output_dir, gradcam_kwargs, suffix='', ext='jpg'):
    fname = generate_fname(gradcam_kwargs)
    save_path = os.path.join(output_dir, f'{fname}.{ext}')
    return save_path

def generate_fname(kwargs, order=('image', 'pixel_i', 'pixel_j', 'class_name')):
    parts = []
    kwargs = kwargs.copy()
    for key in order:
        if key not in kwargs:
            continue
        parts.append(f'{key}-{kwargs.pop(key)}')
    for key in sorted(kwargs):
        parts.append(f'{key}-{kwargs.pop(key)}')
    return '-'.join(parts)

def compute_overlap(label, gradcam):
    cls_to_mass = {}
    gradcam = GradCAM.normalize(gradcam)[0,0]
    for cls in map(int, np.unique(label.tolist())):
        selector = label == cls
        cls_to_mass[cls] = gradcam[selector].sum() / selector.sum()
    cls_to_mass.pop(255)  # the 'ignore' label
    return cls_to_mass

def save_overlap(save_path_overlap, save_path_plot, gradcam, label, k=5, save_npy=True):
    overlap = compute_overlap(label, gradcam)
    max_keys = list(reversed(sorted(overlap, key=lambda key: overlap[key])))[:k]
    max_labels = [class_names[key] for key in max_keys]
    max_values = [overlap[key] for key in max_keys]
    if save_npy:
        np.save(save_path_overlap, overlap)

    plt.figure()
    plt.title('Average saliency per class')
    plt.barh(max_labels, max_values)
    plt.xlabel('Average Pixel Normalized Saliency')
    plt.savefig(save_path_plot)
    plt.close()

def get_image_indices(image_index, image_index_range):
    if image_index_range:
        return range(*image_index_range)
    return [image_index]

def crop(i, j, size, image, is_tensor=True):
    half = size // 2
    slice_i = slice( max(i - half, 0) , i + half)
    slice_j = slice( max(j - half, 0), j + half)
    if is_tensor:
        return image[..., slice_i, slice_j]
    return image[slice_i, slice_j, ...]

def get_random_pixels(n, pixels, bin_size=300, seed=0):
    random.seed(seed)

    bin_to_pixels = defaultdict(lambda: [])
    for (i, j) in pixels:
        bin_to_pixels[(i // bin_size, j // bin_size)].append((i, j))

    if n >= len(bin_to_pixels):
        pixels_per_bins = bin_to_pixels.values()
    else:
        bins = random.sample(bin_to_pixels.keys(), n)
        pixels_per_bins = [bin_to_pixels[bin] for bin in bins]

    pixels = []
    for pixels_per_bin in pixels_per_bins:
        indices = random.sample(range(len(pixels_per_bin)), 1)
        pixels.append(pixels_per_bin[indices[0]])
    return pixels


# def occlusion_sensitivity(
#     model, images, sal, mean=None, patch=90, stride=90, n_batches=2
# ):
#     """
#     "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
#     https://arxiv.org/pdf/1610.02391.pdf
#     Look at Figure A5 on page 17
#     Originally proposed in:
#     "Visualizing and Understanding Convolutional Networks"
#     https://arxiv.org/abs/1311.2901
#     """
#     torch.set_grad_enabled(False)
#     model.eval()
#     mean = mean if mean else 0
#     patch_H, patch_W = patch if isinstance(patch, Sequence) else (patch, patch)
#     pad_H, pad_W = patch_H // 2, patch_W // 2

#     # Padded image
#     images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
#     B, _, H, W = images.shape
#     new_H = (H - patch_H) // stride + 1
#     new_W = (W - patch_W) // stride + 1

#     # Prepare sampling grids
#     anchors = []
#     grid_h = 0
#     while grid_h <= H - patch_H:
#         grid_w = 0
#         while grid_w <= W - patch_W:
#             grid_w += stride
#             anchors.append((grid_h, grid_w))
#         grid_h += stride

#     __import__('ipdb').set_trace()
#     pred_probs, pred_labels = sal.forward(images)
#     # baseline = np.where(pred_labels[:,:,180,1000].numpy()[0]==2)[0].item()
#     baseline = torch.where(pred_labels[0,:,180,1000]==2)[0].item()
#     baseline = pred_probs[0,baseline,180,1000].item()
#     # Baseline score without occlusion
#     # baseline = model(images).detach().gather(1, ids)

#     # Compute per-pixel logits
#     scoremaps = []
#     for i in tqdm(range(0, len(anchors), n_batches), leave=False):
#         batch_images = []
#         batch_ids = []
#         for grid_h, grid_w in anchors[i : i + n_batches]:
#             images_ = images.clone()
#             images_[..., grid_h : grid_h + patch_H, grid_w : grid_w + patch_W] = mean
#             batch_images.append(images_)
#             # batch_ids.append(ids)
#         batch_images = torch.cat(batch_images, dim=0)
#         # batch_ids = torch.cat(batch_ids, dim=0)
#         __import__('ipdb').set_trace()
#         probs, labels = sal.forward(batch_images)
#         scoremaps += torch.where(labels[:,:,180,1000]==2)[1].tolist()
#         # scores = model(batch_images).detach().gather(1, batch_ids)
#         # scoremaps += list(torch.split(scores, B))

#     diffmaps = torch.cat(scoremaps, dim=1) - baseline
#     diffmaps = diffmaps.view(B, new_H, new_W)

#     return diffmaps






def main():
    random.seed(11)
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'vis_gradcam')

    # logger.info(pprint.pformat(args))
    # logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # build model
    model = eval('models.'+config.MODEL.NAME +
                 '.get_seg_model')(config)

    dump_input = torch.rand(
        (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'best.pth')
    model_state_file = 'pretrained_models/hrnet_w18_small_model_v1.pth'
    logger.info('=> loading model from {}'.format(model_state_file))
    # __import__('ipdb').set_trace()
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    # for k, _ in pretrained_dict.items():
    #     logger.info(
    #         '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Wrap original model with NBDT
    if config.NBDT.USE_NBDT:
        from nbdt.model import SoftSegNBDT
        model = SoftSegNBDT(
            config.NBDT.DATASET, model, hierarchy=config.NBDT.HIERARCHY,
            classes=class_names)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = model.to(device).eval()
    # for x in model.named_modules():
    #     print(x)
    # Retrieve input image corresponding to args.image_index
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_dataset = eval('datasets.'+config.DATASET.DATASET)(
                        root=config.DATASET.ROOT,
                        list_path=config.DATASET.TEST_SET,
                        num_samples=None,
                        num_classes=config.DATASET.NUM_CLASSES,
                        multi_scale=False,
                        flip=False,
                        ignore_label=config.TRAIN.IGNORE_LABEL,
                        base_size=config.TEST.BASE_SIZE,
                        crop_size=test_size,
                        downsample_rate=1)

    # Define target layer as final convolution layer if not specified
    if args.target_layers:
        target_layers = args.target_layers.split(',')
    else:
        for name, module in list(model.named_modules())[::-1]:
            if isinstance(module, nn.Conv2d):
                target_layers = [name]
            break
    logger.info('Target layers set to {}'.format(str(target_layers)))

    # Append model. to target layers if using nbdt
    if config.NBDT.USE_NBDT:
        target_layers = ['model.' + layer for layer in target_layers]

    dmg_by_cls = [[] for _ in class_names]
    def generate_and_save_saliency(
            image_index, pixel_i=None, pixel_j=None, crop_size=None,
            cls_index=0, normalize=False, previous_pred=1):
        """too lazy to move out to global lol"""
        nonlocal maximum, minimum, label
        # Generate GradCAM + save heatmap
        # raw_image = retrieve_raw_image(test_dataset, image_index)

        # should_crop = crop_size is not None and pixel_i is not None and pixel_j is not None
        # if should_crop:
        #     raw_image = crop(pixel_i, pixel_j, crop_size, raw_image, is_tensor=False)

        for layer in target_layers:
            gradcam_region = gradcam.generate(target_layer=layer, normalize=False)

            # if should_crop:
            #     gradcam_region = crop(pixel_i, pixel_j, crop_size, gradcam_region, is_tensor=True)

            maximum = max(float(gradcam_region.max()), maximum)
            minimum = min(float(gradcam_region.min()), minimum)
            logger.info(f'=> Bounds: ({minimum}, {maximum})')


            # print('start')
            image = test_dataset[image_index][0]
            # coords = gradcam_region.nonzero().cpu().numpy()

            # __import__('ipdb').set_trace()
            # centered_coords = coords - np.array([0, 0, pixel_i, pixel_j])
            # centered_coords = np.abs(centered_coords)
            # # Filter function for determining whether pixel is within receptive field
            # f = lambda x: x[2] < 200 and x[3] < 200
            # within_receptive_field = np.apply_along_axis(f, 1, centered_coords)
            # coords = coords[within_receptive_field]
            # gradcam_region = gradcam_region.cpu()
            # def v(x):
            #     return gradcam_region[x[0], x[1], x[2], x[3]]
            # within_weight = np.apply_along_axis(v, 1, coords).sum()

            # count = 0
            # inside_mass = 0
            # total_mass = 0
            # for coord in foo:
            #     total_mass += gradcam_region[coord[0], coord[1], coord[2], coord[3]]
            #     image[:, coord[2], coord[3]] = torch.zeros(1, 3)
            #     # 400px receptive field
            #     if abs(coord[2] - pixel_i) < 200 and abs(coord[3] - pixel_j) < 200:
            #         count += 1
            #         inside_mass += gradcam_region[coord[0], coord[1], coord[2], coord[3]]

            # percent_mass_inside = within_weight / gradcam_region.sum()
            # percent_mass_outside = 1 - percent_mass_inside
            # percent_pixels_inside = np.sum(within_receptive_field) / len(within_receptive_field)
            # percent_pixels_outside = 1 - percent_pixels_inside

            # pct_mass_inside_list.append(percent_mass_inside)
            # pct_pixels_inside_list.append(percent_pixels_inside)

            image = torch.from_numpy(image).unsqueeze(0).to(gradcam_region.device)


            # Zero out pixels marked by the saliency map
            coords_to_zero = gradcam_region != 0
            image[:, 0:1, :, :][coords_to_zero] = 0
            image[:, 1:2, :, :][coords_to_zero] = 0
            image[:, 2:3, :, :][coords_to_zero] = 0

            pred_probs, pred_labels = gradcam.forward(image)
            # New location of the target class after occlusion
            new_index = np.where(pred_labels[0, :, pixel_i, pixel_j].cpu().numpy() == cls_index)[0][0]
            damage = previous_pred - pred_probs[0, new_index, pixel_i, pixel_j].item()
            # dmg_by_cls[cls_index].append(damage)
            dmg_by_cls[cls_index].append((damage, image_index, [pixel_i, pixel_j]))
            del image
            del pred_probs
            del pred_labels
            torch.cuda.empty_cache()

            # bar = pred_probs[:,:,pixel_i,pixel_j]
            # baz = pred_labels[:,:,pixel_i,pixel_j]

            # print("percent_mass_inside ", percent_mass_inside)
            # print("percent_pixels_inside ", percent_pixels_inside)

            # __import__('ipdb').set_trace()
            # image = test_dataset[image_index][0]
            # image = torch.from_numpy(image).unsqueeze(0).to(gradcam_region.device)
            # pred_probs, pred_labels = gradcam.forward(image)
            # ordered_class_names = [class_names[i] for i in pred_labels[0,:,180,1000]]
            # target_index = ordered_class_names.index('building')
            # images = image

            # pred_labels[:, [target_index], :, :]
            # occlusion_map = occlusion_sensitivity(model, images, gradcam)

            # output_dir = generate_output_dir(final_output_dir, args.vis_mode, layer, config.NBDT.USE_NBDT, nbdt_node_wnid, args.crop_size, args.nbdt_node_wnids_for)
            # save_path = generate_save_path(output_dir, gradcam_kwargs)
            # logger.info('Saving {} heatmap at {}...'.format(args.vis_mode, save_path))

            # if normalize:
            #     gradcam_region = GradCAM.normalize(gradcam_region)
            #     save_gradcam(save_path, gradcam_region, raw_image, save_npy=not args.skip_save_npy)
            # else:
            #     save_gradcam(save_path, gradcam_region, raw_image, minimum=minimum, maximum=maximum, save_npy=not args.skip_save_npy)

            # output_dir_original = output_dir + '_original'
            # os.makedirs(output_dir_original, exist_ok=True)
            # save_path_original = generate_save_path(output_dir_original, gradcam_kwargs, ext='jpg')
            # logger.info('Saving {} original at {}...'.format(args.vis_mode, save_path_original))
            # cv2.imwrite(save_path_original, raw_image)

        #     if crop_size and pixel_i and pixel_j:
        #         continue
        #     output_dir += '_overlap'
        #     os.makedirs(output_dir, exist_ok=True)
        #     save_path_overlap = generate_save_path(output_dir, gradcam_kwargs, ext='npy')
        #     save_path_plot = generate_save_path(output_dir, gradcam_kwargs, ext='jpg')
        #     if not args.skip_save_npy:
        #         logger.info('Saving {} overlap data at {}...'.format(args.vis_mode, save_path_overlap))
        #     logger.info('Saving {} overlap plot at {}...'.format(args.vis_mode, save_path_plot))
        #     save_overlap(save_path_overlap, save_path_plot, gradcam_region, label, save_npy=not args.skip_save_npy)
        # if len(heatmaps) > 1:
        #     combined = torch.prod(torch.stack(heatmaps, dim=0), dim=0)
        #     combined /= combined.max()
        #     save_path = generate_save_path(final_output_dir, args.vis_mode, gradcam_kwargs, 'combined', config.NBDT.USE_NBDT, nbdt_node_wnid)
        #     logger.info('Saving combined {} heatmap at {}...'.format(args.vis_mode, save_path))
        #     save_gradcam(save_path, combined, raw_image)

    nbdt_node_wnids = args.nbdt_node_wnid or []
    cls = args.nbdt_node_wnids_for
    if cls:
        assert config.NBDT.USE_NBDT, 'Must be using NBDT'
        from nbdt.data.custom import Node
        assert hasattr(model, 'rules') and hasattr(model.rules, 'nodes'), \
            'NBDT must have rules with nodes'
        logger.info("Getting nodes leading up to class leaf {}...".format(cls))
        leaf_to_path_nodes = Node.get_leaf_to_path(model.rules.nodes)

        cls_index = class_names.index(cls)
        leaf = model.rules.nodes[0].wnids[cls_index]
        path_nodes = leaf_to_path_nodes[leaf]
        nbdt_node_wnids = [item['node'].wnid for item in path_nodes if item['node']]

    def run():
        nonlocal maximum, minimum, label, gradcam_kwargs
        np.random.seed(13)
        for image_index in get_image_indices(args.image_index, args.image_index_range):
            image, label, _, name = test_dataset[image_index]
            image = torch.from_numpy(image).unsqueeze(0).to(device)
            logger.info("Using image {}...".format(name))

            pred_probs, pred_labels = gradcam.forward(image)

            maximum, minimum = -1000, 0
            logger.info(f'=> Starting bounds: ({minimum}, {maximum})')

            if args.crop_for and class_names.index(args.crop_for) not in label:
                print(f'Skipping image {image_index} because no {args.crop_for} found')
                continue

            if getattr(Saliency, 'whole_image', False):
                assert not (
                        args.pixel_i or args.pixel_j or args.pixel_i_range
                        or args.pixel_j_range), \
                    'the "Whole" saliency method generates one map for the whole ' \
                    'image, not for specific pixels'
                gradcam_kwargs = {'image': image_index}
                if args.suffix:
                    gradcam_kwargs['suffix'] = args.suffix
                gradcam.backward(pred_labels[:,[0],:,:])

                generate_and_save_saliency(image_index)

                if args.crop_size <= 0:
                    continue

            if args.crop_for:
                cls_index = class_names.index(args.crop_for)
                label = torch.Tensor(label).to(pred_labels.device)
                # is_right_class = pred_labels[0,0,:,:] == cls_index
                # is_correct = pred_labels == label
                is_right_class = is_correct = label == cls_index  #TODO:tmp
                pixels = (is_right_class * is_correct).nonzero()

                pixels = get_random_pixels(args.pixel_max_num_random, pixels, seed=cls_index)
            else:
                assert (args.pixel_i or args.pixel_i_range) and (args.pixel_j or args.pixel_j_range)
                pixels = get_pixels(
                    args.pixel_i, args.pixel_j, args.pixel_i_range, args.pixel_j_range,
                    args.pixel_cartesian_product)
            logger.info(f'Running on {len(pixels)} pixels.')

            # for pixel_index in range(len(pixels)):
            # for cls_index in range(len(class_names)):
            for _ in range(1):
                cls_index = 17 # hardcode person
                # pixel_i, pixel_j = int(pixels[pixel_index][0]), int(pixels[pixel_index][1])
                # pixel_i, pixel_j = int(args.pixel_i[0]), int(args.pixel_j[0])
                # list of all coords where the label matches the current class
                # index.
                matching_coords = np.random.permutation(np.array(np.where(label==cls_index)).T)
                if matching_coords.shape[0] == 0:
                    continue
                pixel_i, pixel_j = 0, 0
                found_matching = False
                for coord in matching_coords:
                    pixel_i, pixel_j = coord[0], coord[1]
                    if pred_labels[0, 0, pixel_i, pixel_j].item() == cls_index:
                        found_matching = True
                        break

                if not found_matching:
                    continue
                assert pred_labels[0, 0, pixel_i, pixel_j].item() == cls_index

                # pixel_i, pixel_j = ps[pixel_index]

                # top_class_index = pred_labels[0,0,pixel_i,pixel_j]
                # top_class_name = class_names[top_class_index]
                # print("Running on pixel ({}, {})".format(pixel_i, pixel_j))
                assert pixel_i < test_size[0] and pixel_j < test_size[1], \
                    "Pixel ({},{}) is out of bounds for image of size ({},{})".format(
                        pixel_i,pixel_j,test_size[0],test_size[1])

                # Get the current prediction confidence for the top class.
                # Will use to compute occlusion damage later.
                curr_pred = pred_probs[0, 0, pixel_i, pixel_j].item()
                # Run backward pass
                # Note: Computes backprop wrt most likely predicted class rather than gt class

                # gradcam_kwargs = {'image': image_index, 'pixel_i': pixel_i, 'pixel_j': pixel_j, 'class_name': top_class_name}
                # if args.suffix:
                #     gradcam_kwargs['suffix'] = args.suffix
                # logger.info(f'Running {args.vis_mode} on image {image_index} at pixel ({pixel_i},{pixel_j}). Using filename suffix: {args.suffix}')
                output_pixel_i, output_pixel_j = compute_output_coord(pixel_i, pixel_j, test_size, pred_probs.shape[2:])
                target_index = 0
                if not getattr(Saliency, 'whole_image', False):
                    gradcam.backward(pred_labels[:, [target_index], :, :], output_pixel_i, output_pixel_j)

                # gradcam_region = gradcam.generate(target_layer=target_layers[0], normalize=False).cpu()
                # __import__('ipdb').set_trace()

                # now `gradcam_region` and `second_region` contain both saliency maps for two pixels of the
                # same class but at least 800px apart.




                # ordered_class_names = [class_names[i] for i in pred_labels[0,:,pixel_i,pixel_j]]
                # target_index = ordered_class_names.index('car')
                # print("TOP CLASS", pred_labels[0, 0, pixel_i, pixel_j], pred_probs[0,0,pixel_i,pixel_j], class_names[pred_labels[0, 0, pixel_i, pixel_j]])
                # print("######################################")

                if args.crop_size <= 0:
                    generate_and_save_saliency(image_index, pixel_i, pixel_j, cls_index=cls_index, previous_pred=curr_pred)
                else:
                    generate_and_save_saliency(image_index, pixel_i, pixel_j, args.crop_size, cls_index=cls_index, previous_pred=curr_pred)

            # for i in range(len(dmg_by_cls)):
            #     if len(dmg_by_cls[i]) > 0 and i == 16:
            #         # print(i, np.mean(dmg_by_cls[i]))
            #         # logger.info('{}, {}'.format(i, np.mean(dmg_by_cls[i])))
            #         logger.info('{}, {}'.format(i, dmg_by_cls[i]))

            # logger.info(f'=> Final bounds are: ({minimum}, {maximum})')

    # Instantiate wrapper once, outside of loop
    Saliency = METHODS[args.vis_mode]
    gradcam = Saliency(model=model, candidate_layers=target_layers,
        use_nbdt=config.NBDT.USE_NBDT, nbdt_node_wnid=None)

    maximum, minimum, label, gradcam_kwargs = -1000, 0, None, {}
    for nbdt_node_wnid in nbdt_node_wnids:
        if config.NBDT.USE_NBDT:
            logger.info("Using logits from node with wnid {}...".format(nbdt_node_wnid))
        gradcam.set_nbdt_node_wnid(nbdt_node_wnid)
        run()

    if not nbdt_node_wnids:
        nbdt_node_wnid = None
        run()
        # for i in range(len(dmg_by_cls)):
        #     if len(dmg_by_cls[i]) > 0:
        #         logger.info('{}, {}'.format(i, np.mean(dmg_by_cls[i])))
        for i in range(len(dmg_by_cls)):
            if len(dmg_by_cls[i]) > 0:
                # print(i, np.mean(dmg_by_cls[i]))
                # logger.info('{}, {}'.format(i, np.mean(dmg_by_cls[i])))
                logger.info('{}, {}'.format(i, dmg_by_cls[i]))
        # __import__('ipdb').set_trace()
        print('done')


if __name__ == '__main__':
    main()
