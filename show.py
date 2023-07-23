import numpy as np

import gradio as gr
from gradio import components as comp

from onnx_ocr import ocr_proc

# document
# https://developer.aliyun.com/article/1202975
# https://blog.51cto.com/u_15485092/6223566
"""
DB++

# input
crnn, tiny, small, base, large

# output
output image
OCR Value (idx, text, accuracy)

"""

model_list = ["tiny", "small", "base", "large"]


def ocr_process(model_option, image):
    print(model_option)
    print(image.shape)

    output_df = [
        [1, 'value1', 0.98],
        [2, 'value2', 0.94],
        [3, 'value3', 0.97],
        [4, 'value4', 0.99],
    ]
    # return image, output_df
    image, data = ocr_proc(image)
    print(image.shape, data)
    return image, data


with gr.Blocks() as demo:
    # introduction
    comp.Markdown("OCR application demo, DB detection model + SVTR model")
    # Radio
    model_option = comp.Radio(model_list)

    input_image = comp.Image()

    comp.Markdown("Output:")

    output_image = comp.Image()

    output_df = comp.Dataframe(
        headers=["index", "text", "accuracy"],
        datatype=["number", "str", "number"],
    )

    btn = comp.Button("submit")

    btn.click(fn=ocr_process, inputs=[model_option, input_image], outputs=[output_image, output_df])

if __name__ == "__main__":
    demo.launch()
