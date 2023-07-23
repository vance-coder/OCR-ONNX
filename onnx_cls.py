import onnxruntime as ort
import numpy as np
import cv2
import time
import os


# onnx inference

# preprocess
def orient_preproc(img, input_size):
    # img = img[:, :, ::-1] #BGR to RGB
    img = cv2.resize(img, input_size, interpolation=1)  # unified resize
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean = np.array(mean).reshape((1, 1, 3)).astype('float32')  # broadcast
    std = np.array(std).reshape((1, 1, 3)).astype('float32')  # broadcast
    img = (img.astype('float32') * np.float32(1.0 / 255.0) - mean) / std  # normalize scale:1.0/255.0
    img = img.transpose(2, 0, 1).astype('float32')  # whc to chw
    return img.reshape(1, 3, input_size[0], input_size[1])


def orient_correct(model, img):
    label_list = ['0', '90', '180', '270']
    rotate_list = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]

    norm_img = orient_preproc(img.copy(), (224, 224))  # w,h
    outputs = model.run(None, {model.get_inputs()[0].name: norm_img})
    res_idx = np.argmax(outputs[0][0])
    print('orientation:', label_list[res_idx])

    angle = rotate_list[res_idx]
    if angle is not None:
        img = cv2.rotate(img, angle)
    return img


# 添加 acc
# gradio
# 检测模型训练

if __name__ == '__main__':
    pass
