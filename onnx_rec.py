import time
import math
import cv2
import numpy as np
from PIL import Image
from onnxruntime import InferenceSession

from preprocess import resize_norm_img
from postprocess import RecPred

from collections import Counter
from itertools import groupby


def split_overlength(img):
    ratio = 0.01
    num_threshold = 3  # The minimum number of [num_threshold] will be set to 0
    step_threshold = 3  # [step_threshold] number of pixel means effective index

    # bgr2gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # binarization
    bin_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
    # display(Image.fromarray(bin_img))

    # count blank pixel
    h, w = bin_img.shape
    pixel = [0] * w
    for x in range(w):
        for y in range(h):
            if bin_img[y, x] == 0:
                pixel[x] += 1

    # calculate threshold
    # 获取可切割位置
    counter = Counter(pixel)
    wrap_threshold = []
    new_pixel = [i for i in set(pixel) if counter.get(i) >= (w * ratio)]
    cut_threshold = new_pixel[num_threshold]

    cut_idx = []
    count_idx = 0
    cut_pixel = [0 if i <= cut_threshold else i for i in pixel]
    for val, lst in groupby(cut_pixel):
        _len = len(list(lst))
        if val == 0 and _len >= step_threshold:
            cut_idx.append(count_idx + int(_len / 2))
        count_idx += _len

    imgs = []
    if not cut_idx:
        return imgs

    # paint all idx
    # paint_img =  cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # for idx in cut_idx:
    #     ptStart = (idx, 0)
    #     ptEnd = (idx, height)
    #     point_color = (255, 0, 0)
    #     thickness = 1
    #     lineType = 4
    #     cv2.line(paint_img, ptStart, ptEnd, point_color, thickness, lineType)
    # display(Image.fromarray(paint_img))

    def get_x_idx(diff, lst):
        # get split x coordinate
        diff_lst = [abs(i - diff) for i in lst]
        return lst[diff_lst.index(min(diff_lst))]

    # split x_idx
    # print('len_cut_idx', len(cut_idx))
    if len(cut_idx) > 12:
        # split to 3 images
        x_idx1 = get_x_idx(w / 3 * 1, cut_idx)
        x_idx2 = get_x_idx(w / 3 * 2, cut_idx)
        img1 = img[0:h, 0:x_idx1]
        img2 = img[0:h, x_idx1:x_idx2]
        img3 = img[0:h, x_idx2:w]
        imgs += [img1, img2, img3]

    else:
        # split to 2 images
        x_idx = get_x_idx(w / 2, cut_idx)
        img1 = img[0:h, 0:x_idx]
        img2 = img[0:h, x_idx:w]
        imgs += [img1, img2]

    return imgs


def onnx_rec_img(img, model, pred_func, padding=True, rec_image_shape=(3, 32, 480)):
    # gray2bgr
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    resized_img = resize_norm_img(img, padding=padding, rec_image_shape=rec_image_shape)
    resized_img = resized_img[np.newaxis, :]

    pred = model.run(output_names=None, input_feed={'x': resized_img})
    return pred_func(pred[0])[0]


def onnx_rec_img_list(img_list, model, pred_func, batch_num=8, padding=True, rec_image_shape=(3, 32, 480)):
    basic_acc = 0.9
    max_width = 320
    max_multiple = 12  # width/height
    img_num = len(img_list)
    # Calculate the aspect ratio of all text bars
    width_list = []
    for img in img_list:
        width_list.append(img.shape[1] / float(img.shape[0]))
    # Sorting can speed up the recognition process
    # sort by width of image
    indices = np.argsort(np.array(width_list))
    rec_res = [['', 0.0]] * img_num
    st = time.time()

    for beg_img_no in range(0, img_num, batch_num):
        end_img_no = min(img_num, beg_img_no + batch_num)
        norm_img_batch = []
        norm_origin_img_batch = []
        imgC, imgH, imgW = rec_image_shape[:3]
        max_wh_ratio = imgW / imgH
        # max_wh_ratio = 0
        for ino in range(beg_img_no, end_img_no):
            h, w = img_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = resize_norm_img(img_list[indices[ino]], padding=padding, rec_image_shape=rec_image_shape)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)
            norm_origin_img_batch.append(img_list[indices[ino]])
        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()

        input_dict = {}
        input_dict['x'] = norm_img_batch
        outputs = model.run(output_names=None, input_feed=input_dict)
        preds = outputs[0]

        rec_result = pred_func(preds)
        not_padding = not padding

        for rno in range(len(rec_result)):
            # print(beg_img_no, rno)
            # 结果修正
            label, acc = rec_result[rno]
            cur_img = norm_origin_img_batch[rno]
            cur_h, cur_w = cur_img.shape[0:2]

            if acc < basic_acc:
                # 尝试拆分图片
                if cur_w > max_width and (cur_w / cur_h) > max_multiple:
                    # print('尝试拆分图片')
                    split_imgs = split_overlength(cur_img)

                    # padding
                    new_label = ''
                    new_acc = 0
                    for im in split_imgs:
                        tmp_label, tmp_acc = onnx_rec_img(im, model, pred_func, padding=padding, rec_image_shape=rec_image_shape)
                        new_label += tmp_label
                        new_acc += tmp_acc
                    new_acc = new_acc / len(split_imgs)
                    if new_acc > acc:
                        label, acc = new_label, new_acc

                    # not padding
                    if acc < basic_acc:
                        new_label = ''
                        new_acc = 0
                        for im in split_imgs:
                            tmp_label, tmp_acc = onnx_rec_img(im, model, pred_func, padding=not_padding, rec_image_shape=rec_image_shape)
                            new_label += tmp_label
                            new_acc += tmp_acc
                        new_acc = new_acc / len(split_imgs)
                        if new_acc > acc:
                            label, acc = new_label, new_acc

            if acc < basic_acc:
                # 尝试不进行padding， return (label, acc)
                # print('尝试不进行padding')
                new_label, new_acc = onnx_rec_img(cur_img, model, pred_func, padding=not_padding, rec_image_shape=rec_image_shape)
                if new_acc > acc:
                    label, acc = new_label, new_acc
            rec_res[indices[beg_img_no + rno]] = (label, acc)

    return rec_res, time.time() - st


if __name__ == '__main__':

    pred_process = RecPred('models/vocab.txt', 'ch', use_space_char=True)

    path = 'models/svtr_tiny.onnx'
    # OpenVINOExecutionProvider
    # model = InferenceSession(path, providers=['CUDAExecutionProvider'])
    rec_model = InferenceSession(path, providers=['CPUExecutionProvider'])

    print(rec_model.get_providers())

    img_path = 'images/crop1.png'
    img = cv2.imread(img_path)

    print(onnx_rec_img_list([img,], rec_model, pred_process, padding=True, rec_image_shape=(3, 32, 400)))


