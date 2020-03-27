import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval
from utils.modelsummary import get_model_summary
from utils.utils import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize effective receptive field')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--out-i', type=int, default=0, 
                        help='i coordinate of output to compute ERF')
    parser.add_argument('--out-j', type=int, default=0, 
                        help='j coordinate of output to compute ERF')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args

def main():
    args = parse_args()

    logger, final_output_dir, _ = create_logger(
        config, args.cfg, 'vis_erf')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

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
    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'best.pth')
    logger.info('=> loading model from {}'.format(model_state_file))
        
    pretrained_dict = torch.load(model_state_file)
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                        if k[6:] in model_dict.keys()}
    for k, _ in pretrained_dict.items():
        logger.info(
            '=> loading {} from pretrained model'.format(k))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    # Generate input
    logger.info('Generating input of ones...')
    input = torch.ones((1,3,config.TEST.IMAGE_SIZE[1],config.TEST.IMAGE_SIZE[0])).float()
    input = torch.autograd.Variable(input, requires_grad=True)

    # Run input through network + compute backward pass
    logger.info('Computing backward pass...')
    output = model(input)
    output_grad = torch.zeros_like(output)
    output_grad[0,0,args.out_i,args.out_j] = 1.
    output.backward(gradient=output_grad)
    input_grad = input.grad[0,0].data.numpy()

    # Save plot of gradient
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(input_grad, cmap='coolwarm')

if __name__ == '__main__':
    main()
