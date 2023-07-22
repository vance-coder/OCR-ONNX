import cv2
import onnx
import numpy as np
import onnxruntime
from PIL import Image
from postprocess import DBPostProcess

# 大部分代码来自PaddleOCR
model_shape_w = 48


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect


def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points


# shape_part_list = [661 969 7.74583964e-01 6.60474716e-01]
def filter_tag_det_res(dt_boxes, shape_part_list):
    img_height, img_width = shape_part_list[0], shape_part_list[1]
    dt_boxes_new = []
    for box in dt_boxes:
        box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        rect_width = int(np.linalg.norm(box[0] - box[1]))
        rect_height = int(np.linalg.norm(box[0] - box[3]))
        if rect_width <= 3 or rect_height <= 3:
            continue
        dt_boxes_new.append(box)
    dt_boxes = np.array(dt_boxes_new)
    return dt_boxes


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        print(op_name, param)
        op = eval(op_name)(**param)
        ops.append(op)
    return ops


def get_rotate_crop_image(img, points):
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


def get_boxes(img, onnx_model, preprocess, postprocess):
    input_name = onnx_model.get_inputs()[0].name
    img_part = img.copy()
    data_part = {'image': img_part}
    data_part = transform(data_part, preprocess)
    img_part, shape_part_list = data_part
    img_part = np.expand_dims(img_part, axis=0)
    shape_part_list = np.expand_dims(shape_part_list, axis=0)

    inputs_part = {input_name: img_part}
    outs_part = onnx_model.run(None, inputs_part)
    outs_part = outs_part[0]

    post_res_part = postprocess(outs_part, shape_part_list)
    dt_boxes_part = post_res_part[0]['points']
    dt_boxes_part = filter_tag_det_res(dt_boxes_part, shape_part_list[0])
    dt_boxes_part = sorted_boxes(dt_boxes_part)

    return dt_boxes_part, img_part


if __name__ == '__main__':
    det_db_thresh = 0.3
    det_db_box_thresh = 0.3
    max_candidates = 1000
    unclip_ratio = 1.6
    use_dilation = True
    # DetResizeForTest 定义检测模型前处理规则

    pre_process_list = [{
        'DetResizeForTest': {
            # 'limit_side_len': 2500,
            # 'limit_type': 'max',
            'resize_long': 640  # 512
            # 'image_shape':[640,640],
            # 'keep_ratio':True,
        }
    }, {
        'NormalizeImage': {
            'std': [1.0, 1.0, 1.0],
            'mean':
                [0.48109378172549, 0.45752457890196, 0.40787054090196],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }, {
        'ToCHWImage': None
    }, {
        'KeepKeys': {
            'keep_keys': ['image', 'shape']
        }
    }]

    preprocess = create_operators(pre_process_list)
    postprocess = DBPostProcess(det_db_thresh, det_db_box_thresh, max_candidates, unclip_ratio, use_dilation)

    model_path = 'models/model_det.onnx'
    image_file = 'images/10.jpg'

    det_model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    img = cv2.imread(image_file)

    dt_boxes_part, img_part = get_boxes(img, det_model, preprocess, postprocess)
    print(dt_boxes_part)
