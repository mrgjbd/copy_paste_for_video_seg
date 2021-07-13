#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 上午11:09
# @Author  : guoxz
# @Site    : 
# @File    : copy_paste.py
# @Software: PyCharm
# @Description

import argparse
import os
import random
import shutil
from glob import glob
from multiprocessing import Pool

import numpy as np
from PIL import Image

from palette import palette

randint = random.randint


def construct_hyper_param():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir",
                        default='/opt/disk1/image/全球视频云创新挑战赛/PreRoundData',
                        type=str,
                        help="src_dir")

    parser.add_argument("--target_dir",
                        default='/opt/disk1/image/全球视频云创新挑战赛/copy_paste_same',
                        type=str,
                        help="target_dir")

    parser.add_argument("--num_worker", default=4, type=int,
                        help="seed")

    parser.add_argument("--seed", default=123, type=int,
                        help="seed")

    args = parser.parse_args()
    return args


class Pos(object):
    tx_start = None
    ty_start = None
    tx_end = None
    ty_end = None

    x_start = None
    y_start = None
    x_end = None
    y_end = None

    x_center = None
    y_center = None

    target_x = None
    target_y = None

    scale = None


def process_single_img(img, img_p, target_w=1280, target_h=720, is_background=True, pos=None):
    w, h = img.size
    target_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 192
    target_img_p = np.zeros((target_h, target_w), dtype=np.uint8)

    if pos is None:

        pos = Pos()

        # 处理背景色
        scale = randint(1, 20) / 10

        pos.scale = scale

        w1 = int(w * scale)
        h1 = int(h * scale)
        img = img.resize((w1, h1), Image.ANTIALIAS)
        img_p = img_p.resize((w1, h1), Image.ANTIALIAS)

        img = np.array(img)
        img_p = np.array(img_p)
        num_objs = len(np.unique(img_p)) - 1

        # 找x方向目标区域的中点
        x_array = img_p.sum(axis=0)
        start = -1
        end = -1
        for i, v in enumerate(x_array):
            if v != 0 and start == -1:
                start = i
            if start != -1 and v == 0:
                end = i
                break
        x_center = (start + end) // 2
        # 找y方向目标区域的中点
        y_array = img_p.sum(axis=1)
        start = -1
        end = -1
        for i, v in enumerate(y_array):
            if v != 0 and start == -1:
                start = i
            if start != -1 and v == 0:
                end = i
                break
        y_center = (start + end) // 2

        target_x, target_y = randint(int(0.2 * target_w), int(0.8 * target_w)), randint(int(0.2 * target_h),
                                                                                        int(0.8 * target_h))

        pos.x_center = x_center
        pos.y_center = y_center

        pos.target_x = target_x
        pos.target_y = target_y

        if target_y - y_center <= 0:
            ty_start = 0
            y_start = abs(y_center - target_y)
        else:
            ty_start = abs(y_center - target_y)
            y_start = 0

        if target_x - x_center <= 0:
            tx_start = 0
            x_start = abs(x_center - target_x)
        else:
            tx_start = abs(x_center - target_x)
            x_start = 0

        if target_y - y_center + h1 >= target_h:
            ty_end = target_h
            y_end = target_h - (target_y - y_center)
        else:
            ty_end = target_y - y_center + h1
            y_end = h1

        if target_x - x_center + w1 >= target_w:
            tx_end = target_w
            x_end = target_w - (target_x - x_center)
        else:
            tx_end = target_x - x_center + w1
            x_end = w1
        pos.tx_start = tx_start
        pos.ty_start = ty_start
        pos.tx_end = tx_end
        pos.ty_end = ty_end

        pos.x_start = x_start
        pos.y_start = y_start
        pos.x_end = x_end
        pos.y_end = y_end
    else:
        tx_start = pos.tx_start
        ty_start = pos.ty_start
        tx_end = pos.tx_end
        ty_end = pos.ty_end

        x_start = pos.x_start
        y_start = pos.y_start
        x_end = pos.x_end
        y_end = pos.y_end

        scale = pos.scale

        w1 = int(w * scale)
        h1 = int(h * scale)
        img = img.resize((w1, h1), Image.ANTIALIAS)
        img_p = img_p.resize((w1, h1), Image.ANTIALIAS)

        img = np.array(img)
        img_p = np.array(img_p)
        num_objs = len(np.unique(img_p)) - 1

    target_img[ty_start:ty_end, tx_start:tx_end, :] = img[y_start:y_end, x_start:x_end, :]
    target_img_p[ty_start:ty_end, tx_start:tx_end] = img_p[y_start:y_end, x_start:x_end]

    if not is_background:
        target_img[target_img_p == 0, :] = 0

    return target_img, target_img_p, num_objs, pos


def get_num_objs(img_p_files):
    return max([len(np.unique(np.array(Image.open(img_p_file).convert('P')))) - 1 for img_p_file in img_p_files])


def process(x):
    video_idx1, video_idx2, ii = x

    f_img_files = sorted(glob(os.path.join(args.src_dir, f'JPEGImages/{video_idx1}/*.jpg')))
    if len(f_img_files) < 60:
        print(f'{video_idx1} less than 60')
        return

    f_img_p_files = sorted(glob(os.path.join(args.src_dir, f'Annotations/{video_idx1}/*.png')))

    b_img_files = sorted(glob(os.path.join(args.src_dir, f'JPEGImages/{video_idx2}/*.jpg')))
    b_img_p_files = sorted(glob(os.path.join(args.src_dir, f'Annotations/{video_idx2}/*.png')))

    start = randint(0, 20)
    if randint(0, 1) == 0:
        f_img_files = f_img_files[start:]
        f_img_p_files = f_img_p_files[start:]
        b_img_files = b_img_files[:len(f_img_files)]
        b_img_p_files = b_img_p_files[:len(f_img_files)]
    else:
        b_img_files = b_img_files[start:]
        b_img_p_files = b_img_p_files[start:]
        f_img_files = f_img_files[:len(b_img_files)]
        f_img_p_files = f_img_p_files[:len(b_img_p_files)]

    if len(b_img_p_files) == 0 or len(f_img_p_files) == 0:
        return

    b_num_objs = get_num_objs(b_img_p_files)

    b_pos = None
    f_pos = None

    video_id = str(ii)
    video_id = '0' * (6 - len(video_id)) + video_id

    for idx, (f_img_file, f_img_p_file, b_img_file, b_img_p_file) in enumerate(
            zip(f_img_files, f_img_p_files, b_img_files, b_img_p_files)):
        f_img = Image.open(f_img_file).convert('RGB')
        f_img_p = Image.open(f_img_p_file).convert('P')

        b_img = Image.open(b_img_file).convert('RGB')
        b_img_p = Image.open(b_img_p_file).convert('P')

        target_w, target_h = b_img.size

        target_img, target_img_p, _, b_pos = process_single_img(b_img, b_img_p, target_w=target_w,
                                                                target_h=target_h,
                                                                is_background=True,
                                                                pos=b_pos)

        f_img, f_img_p, _, f_pos = process_single_img(f_img, f_img_p, target_w=target_w,
                                                      target_h=target_h,
                                                      is_background=False,
                                                      pos=f_pos)

        target_img[f_img_p != 0, :] = 0
        target_img += f_img
        f_img_p[f_img_p != 0] += b_num_objs
        target_img_p[f_img_p != 0] = f_img_p[f_img_p != 0]

        img_save = Image.fromarray(target_img).convert('RGB')
        img_p_save = Image.fromarray(target_img_p).convert('P')
        img_p_save.putpalette(palette)

        jpg_dir = os.path.join(target_dir, 'JPEGImages', video_id)
        p_dir = os.path.join(target_dir, 'Annotations', video_id)
        os.makedirs(jpg_dir, exist_ok=True)
        os.makedirs(p_dir, exist_ok=True)

        img_save.save(os.path.join(jpg_dir, f'{"0" * (6 - len(str(idx))) + str(idx)}.jpg'), quality=95)
        img_p_save.save(os.path.join(p_dir, f'{"0" * (6 - len(str(idx))) + str(idx)}.png'))


if __name__ == '__main__':

    args = construct_hyper_param()

    random.seed(args.seed)
    target_dir = args.target_dir
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    video_idx_list1 = os.listdir(os.path.join(args.src_dir, 'JPEGImages'))
    videl_id_pairs = [(x, x, i) for i, x in enumerate(video_idx_list1)]

    with Pool(processes=args.num_worker) as pool:
        pool.map(process, videl_id_pairs)
