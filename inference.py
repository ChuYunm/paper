import argparse
import math
import logging

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from dataset import create_dataset
from model import MODEL_NAMES, create_model
from helper import setup_default_logging

import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', type=str, metavar='DIR',
                    help='directory to data')
parser.add_argument('--arch', default='resnet50', type=str, metavar='MODEL', choices=MODEL_NAMES,
                    help='model architecture: ' + ' | '.join(MODEL_NAMES))
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multi', action='store_true', default=False, help='multiple label')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('-b', '--batch-size', type=int, default=4,
                    help='batch size')
parser.add_argument('--num-visualize', type=int, default=None, help='num of images to visualize')
parser.add_argument('--model', type=str, metavar='PATH',
                    help='path to model')

args = parser.parse_args()

Face_expression = ['NO','Yes']
Face_color = ['cyan','red','yellow','white','black','normal']
Face_emotion = ['happy','sad','paceful']

np.seterr(divide='ignore',invalid='ignore')


def visualization(input_tensor, output_tensor, i, nrows, ncols, mean=None, std=None, cuda=True):
    if mean is not None:
        mean = np.array(mean)
    else:
        mean = np.array([0, 0, 0])
    if std is not None:
        std = np.array(std)
    else:
        std = np.array([1, 1, 1])

    if cuda:
        imgs = input_tensor.cpu().detach().numpy()
        outputs = output_tensor.cpu().detach().numpy()
    else:
        imgs = input_tensor.detach().numpy()
        outputs = output_tensor.detach().numpy()

    for j in range(imgs.shape[0]):
        img = imgs[j].transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        color_outputs = outputs[j][:6]
        expression_outputs = outputs[j][6:9]
        emotion_outputs = outputs[j][9:11]

        color_score = np.exp(emotion_outputs) / np.sum(np.exp(emotion_outputs))
        expression_score = np.exp(expression_outputs) / np.sum(np.exp(expression_outputs))
        emotion_score = np.exp(color_outputs) / np.sum(np.exp(color_outputs))

        color_pred = np.argmax(color_score)
        expression_pred = np.argmax(expression_score)
        emotion_pred = np.argmax(emotion_score)

        plt.subplot(nrows, ncols, i + j)
        plt.imshow(img)
        title = 'face_color: {}\nface_expression: {}\nface_emotion: {}'.format(
            face_emotion_labels[color_pred.item()], face_expression_labels[expression_pred.item()], face_color_labels[emotion_pred.item()])
        plt.title(title)
        plt.axis('off')
    plt.savefig(f'./test/predict.jpg')

def inference():
    setup_default_logging()
    _logger = logging.getLogger('inference')

    _logger.info("create model '{}'".format(args.arch))
    model = create_model(args.arch, args.num_classes, pretrained=False)

    # windows 无法使用nccl，下只能用gloo
    # dist.init_process_group(
    #     backend='gloo',
    #     init_method='file:///f:tmp/sharefile',
    #     world_size=1,
    #     rank=0,
    # )

    _logger.info("loading '{}'".format(args.model))
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        checkpoint = torch.load(args.model, map_location='cuda:{}'.format(args.gpu))
    else:
        checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    _logger.info("loaded '{}'".format(args.model))

    model.eval()

    dataset = create_dataset(args.data, mean=args.mean, std=args.std, multi=args.multi, train=False, evaluate=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    n = 16
    if args.num_visualize is not None:
        if 0 < args.num_visualize < n:
            n = args.num_visualize
    ncols = 4
    nrows = math.ceil(n / ncols)
    width = 224 * ncols
    height = 336 * nrows
    dpi = 100
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)

    imgs_so_far = 0
    last = 0
    steps = len(loader)
    pics = 0
    with torch.no_grad():
        for i, (img, fname) in enumerate(loader):
            if args.gpu is not None:
                img = img.cuda(args.gpu)
            output = model(img)
            idx = i * args.batch_size - pics * n + 1
            visualization(img, output, idx, nrows, ncols, args.mean, args.std, cuda=args.gpu is not None)
            imgs_so_far += args.batch_size
            if args.num_visualize is not None:
                if imgs_so_far - last >= n:
                    last = imgs_so_far
                    pics += 1
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    plt.show()
                    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                    if imgs_so_far >= args.num_visualize:
                        exit(0)
                elif imgs_so_far >= args.num_visualize:
                    pics += 1
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    plt.show()
                    exit(0)
            else:
                if i == steps - 1:
                    pics += 1
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    plt.show()
                    exit(0)
                elif imgs_so_far - last >= n:
                    last = imgs_so_far
                    pics += 1
                    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                    plt.show()
                    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)


if __name__ == '__main__':
        start_time = time.time()  # 记录开始时间
    inference()
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算代码执行时间
    print(f"代码执行时间：{execution_time} 秒")
