from onnx_rec import *
from onnx_det import *
from onnx_cls import *

from PIL import Image

# detect
det_db_thresh = 0.3
det_db_box_thresh = 0.3
max_candidates = 1000
unclip_ratio = 1.6
use_dilation = True
# DetResizeForTest 定义检测模型前处理规则
pre_process_list = [{
    'DetResizeForTest': {
        'resize_long': 736  # 512
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

# init detection model
det_model_path = 'models/model_det.onnx'
preprocess = create_operators(pre_process_list)
postprocess = DBPostProcess(det_db_thresh, det_db_box_thresh, max_candidates, unclip_ratio, use_dilation)
det_model = onnxruntime.InferenceSession(det_model_path, providers=['CPUExecutionProvider'])

# init rec model
pred_process = RecPred('vocab.txt', 'ch', use_space_char=True)
svtr_path = 'models/svtr_small.onnx'
svtr_rec_model = InferenceSession(svtr_path, providers=['CPUExecutionProvider'])
# crnn_path = 'models/crnn.onnx'
# crnn_rec_model = InferenceSession(crnn_path, providers=['CPUExecutionProvider'])

# init orientation model
cls_path = 'models/orientation.onnx'
cls_model = InferenceSession(cls_path, providers=['CPUExecutionProvider'])


def ocr_proc(image):
    image = orient_correct(cls_model, image)
    dt_boxes_part, img_part = get_boxes(image, det_model, preprocess, postprocess)

    data = []

    for idx, box in enumerate(dt_boxes_part):
        img_crop = get_rotate_crop_image(image, box)
        text, acc = onnx_rec_img(img_crop, svtr_rec_model, pred_process, padding=True, rec_image_shape=(3, 32, 400))
        # if acc <= 0.93:
        #     text1, acc1 = onnx_rec_img(img_crop, crnn_rec_model, pred_process, padding=True, rec_image_shape=None)
        #     print(text1, acc1)
        #     if acc1 > acc:
        #         text, acc = text1, acc1
        data.append([idx, text, round(float(acc), 3)])

    return image, data
