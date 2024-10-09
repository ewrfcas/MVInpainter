import os.path

import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from typing import List
from matplotlib.patches import Polygon
import json
from PIL import Image

def show_points(ax, points: List[List[float]], top_points=[], size=375):
    points_np = np.array(points)
    ax.scatter(points_np[:, 0], points_np[:, 1], color="red", marker='o', s=size, edgecolor='white', linewidth=1.25)

    if len(points) == 4:  # draw the bottom bbox
        polygon = Polygon(points, closed=True, fill=None, edgecolor='black', linewidth=2)
        ax.add_patch(polygon)

    if len(top_points) == 4:
        top_points_np = np.array(top_points)
        ax.scatter(top_points_np[:, 0], top_points_np[:, 1], color="blue", marker='o', s=size, edgecolor='white', linewidth=1.25)
        polygon = Polygon(top_points, closed=True, fill=None, edgecolor='black', linewidth=2)
        ax.add_patch(polygon)

        for pb, pt in zip(points, top_points):
            ax.plot([pb[0], pt[0]], [pb[1], pt[1]], color="black", linewidth=2)


def get_select_coords(img, mask, bottom_points, default_offset, evt: gr.SelectData):
    dpi = plt.rcParams['figure.dpi']
    img = np.asarray(img)
    height, width = img.shape[:2]

    fig = plt.figure(figsize=(int(width / dpi), int(height / dpi)))
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()

    if len(bottom_points) < 4:
        x = int(evt.index[0])
        y = int(evt.index[1])
        bottom_points.append([x, y])
    else:
        print("Exceed 4 points, ignored...")

    if len(bottom_points) == 4 and mask is not None:
        mask_bbox = np.where(mask > 0)
        h_min, h_max = mask_bbox[0].min(), mask_bbox[0].max()
        # w_min, w_max = mask_bbox[1].min(), mask_bbox[1].max()
        bottom_h_min = min([b[1] for b in bottom_points])
        offset = int(bottom_h_min - h_min + default_offset)
        top_points = [[b[0], b[1] - offset] for b in bottom_points]
    else:
        top_points = [[b[0], b[1] - default_offset] for b in bottom_points]

    show_points(plt.gca(), bottom_points.copy(), top_points.copy(), size=(width * 0.02) ** 2)

    print(bottom_points)

    return fig, bottom_points, top_points


def clear_point(init_img):
    gr.Info("Clear all points!")
    height, width = init_img.shape[:2]
    dpi = plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(int(width / dpi), int(height / dpi)))
    plt.imshow(init_img)
    plt.axis('off')
    plt.tight_layout()

    return fig, [], []


def save_point(init_img, bottom_points, top_points):
    i = 0
    os.makedirs("./bbox", exist_ok=True)
    while os.path.exists(f"./bbox/{str(i).zfill(4)}.json"):
        i += 1
    fname = f"./bbox/{str(i).zfill(4)}.json"
    with open(fname, "w") as w:
        json.dump({"bottom": bottom_points, "top": top_points}, w, indent=2)
    init_img = Image.fromarray(init_img)
    init_img.save(f"./bbox/{str(i).zfill(4)}.png")
    gr.Warning("Save Success!")

if __name__ == "__main__":
    with gr.Blocks(title='DEMO') as block:
        with gr.Row():
            with gr.Column(min_width=512):
                bottom_points = gr.State([])
                top_points = gr.State([])
                init_img = gr.Image(source='upload', label="Input image", type="numpy").style(height=512, width=512)
            with gr.Column():
                mask = gr.Image(source='upload', label="Input mask", type="numpy").style(height=512, width=512)
            with gr.Column(min_width=512):
                img_pointed = gr.Plot(label='Pointed image', max_size=512)
        with gr.Row():
            clear_point_button = gr.Button(value="Clear points", label="Clear points")
            offset = gr.Slider(value=50, label='h offset', minimum=0, maximum=200, interactive=True)
            save_point_button = gr.Button(value="Save points", label="Save points")

        init_img.select(fn=get_select_coords, inputs=[init_img, mask, bottom_points, offset], outputs=[img_pointed, bottom_points, top_points])
        clear_point_button.click(fn=clear_point, inputs=[init_img], outputs=[img_pointed, bottom_points, top_points])
        save_point_button.click(fn=save_point, inputs=[init_img, bottom_points, top_points])

    block.launch(share=False, server_name='localhost', server_port=8894)
