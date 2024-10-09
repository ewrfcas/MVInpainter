from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import torch
import os
from utils.masking import random_irregular_object_mask
import argparse
from roma import roma_outdoor
import torch
from tqdm import tqdm
import json


def draw_dashed_polyline(img, pts, color, thickness, dash_length):
    """
    Draw a dashed polygon.
    :param img: The image to draw on.
    :param pts: An array of polygon vertex points.
    :param color: Line color.
    :param thickness: Line width.
    :param dash_length: Length of the dashed segments.
    """
    # To ensure the first and last points are connected, copy the first point to the end.
    pts = np.concatenate([pts, pts[:1]])
    # Iterate through all point pairs and draw the dashed segments for each.
    for i in range(len(pts) - 1):
        pt1 = pts[i]
        pt2 = pts[i + 1]
        # Calculate the length of the line segment.
        dist = np.linalg.norm(pt2 - pt1)
        # Calculate the number of dashes based on dash_length.
        num_dashes = int(np.floor(dist / dash_length))
        # Loop to draw individual dashed segments.
        for j in range(num_dashes):
            # Calculate the start and end points of the dashed segment.
            start = pt1 + (pt2 - pt1) * (float(j) / num_dashes)
            end = pt1 + (pt2 - pt1) * ((float(j) + 0.5) / num_dashes)
            # Draw the dashed segment.
            cv2.line(img, tuple(np.int32(start)), tuple(np.int32(end)), color, thickness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mask adaption")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--edited_index", type=int, default=0,
                        help="the index of obj_bbox, default: 0-->0000.json, 1-->0001.json...")
    parser.add_argument("--n_sample", type=int, default=100, help="random points sampled from matching")
    parser.add_argument("--h", type=int, default=512)
    parser.add_argument("--w", type=int, default=512)
    parser.add_argument("--mask_dilate", type=int, default=20)
    parser.add_argument("--no_irregular_mask", action="store_true",
                        help="you can disable the irregular mask for more precise warp masks (usually used for long and slender things)")
    args = parser.parse_args()

    # load data
    clean_fs = glob(f"{args.input_path}/removal/*")
    edited_fs = glob(f"{args.input_path}/obj_bbox/*.png")
    bbox_fs = glob(f"{args.input_path}/obj_bbox/*.json")
    plane_mask_fs = glob(f"{args.input_path}/plane_masks/*")
    nframe = len(clean_fs)

    roma_model = roma_outdoor(device="cuda")

    # warpaffine with unmasked matching
    masked_flows = []
    Ms = []

    for i in tqdm(range(nframe - 1)):
        if i > len(plane_mask_fs) - 1:
            mask = np.ones((args.h, args.w), dtype=np.float32)
        else:
            mask = cv2.imread(plane_mask_fs[i])[:, :, 0] / 255
            mask = cv2.resize(mask, [args.h, args.w], interpolation=cv2.INTER_NEAREST)
            mask[mask > 0] = 1

        # src, dst points from dense matching
        imA_path = clean_fs[i]
        imB_path = clean_fs[i + 1]
        with torch.no_grad():
            warp, certainty = roma_model.match(imA_path, imB_path, device="cuda")
            mask = cv2.resize(mask, [certainty.shape[1], certainty.shape[0]], interpolation=cv2.INTER_NEAREST)
            certainty2 = certainty.clone()
            certainty2[mask == 1] = 0
            matches, certainty = roma_model.sample(warp, certainty2)
            src_pts, dst_pts = roma_model.to_pixel_coordinates(matches, args.h, args.w, args.h, args.w)

        src_pts, dst_pts = src_pts.cpu().numpy(), dst_pts.cpu().numpy()
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        Ms.append(M)

    # load bottom points
    print("Processing", bbox_fs[args.edited_index])
    info = json.load(open(bbox_fs[args.edited_index]))

    all_pts = [np.array(info['bottom']).reshape(-1, 1, 2)]
    pts_h = np.array(info['bottom'])[:, 1].mean() - np.array(info['top'])[:, 1].mean()

    for M in Ms:
        dst = cv2.perspectiveTransform(all_pts[-1].astype(np.float32), M)
        all_pts.append(dst)

    print("Drawing bbox")
    convex_masks = []
    res = []
    for i, pts in enumerate(all_pts):
        if i == 0:
            img = plt.imread(bbox_fs[args.edited_index].replace(".json", ".png"))
        else:
            img = plt.imread(clean_fs[i])
        img = img[:, :, :3]
        img = cv2.resize(img, [args.h, args.w])
        pts = pts.reshape(-1, 1, 2).astype(np.int32)
        pts_top = pts.copy()
        pts_top[:, :, 1] -= pts_h.astype(np.int32)
        hull = cv2.convexHull(np.concatenate([pts_top, pts], axis=0).reshape(1, -1, 2)).astype(np.int32)
        for pt in hull[:, 0]:
            cv2.circle(img, (pt[0], pt[1]), radius=10, color=(1, 1, 0), thickness=-1)

        if i == 0:
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 1), thickness=3)
            cv2.polylines(img, [pts_top], isClosed=True, color=(1, 0, 0), thickness=3)
        else:
            draw_dashed_polyline(img, pts.reshape(-1, 2), color=(0, 0, 1), thickness=3, dash_length=25)
            draw_dashed_polyline(img, pts_top.reshape(-1, 2), color=(1, 0, 0), thickness=3, dash_length=25)

        new_mask = np.zeros((args.h, args.w, 3), dtype=np.uint8)
        # fill the polygon
        cv2.fillConvexPoly(new_mask, hull, color=(1, 1, 1))
        new_mask = new_mask[:, :, 0]
        img[new_mask == 1, :] = np.clip(img[new_mask == 1, :] - 0.15, 0, 1.0)
        convex_masks.append(new_mask)

        res.append(img)

    resc = np.concatenate(res, axis=1)
    plt.imsave(f"{args.input_path}/{str(args.edited_index).zfill(4)}_bbox.png", resc)

    # building masks
    print("Getting masks")
    kernel = np.ones((args.mask_dilate, args.mask_dilate))
    enlarged_convex_masks = []
    for i, mask in enumerate(convex_masks):
        mask2 = cv2.dilate(mask, kernel, iterations=1)
        if not args.no_irregular_mask:
            mask2 = random_irregular_object_mask(mask2, mask_enlarge=[0.01, 0.01], pts_size=[60, 70], width_range=[40, 60])
        enlarged_convex_masks.append(mask2)

    os.makedirs(f"{args.input_path}/warp_masks", exist_ok=True)
    for i, mask in enumerate(enlarged_convex_masks):
        cv2.imwrite(f"{args.input_path}/warp_masks/{str(i).zfill(4)}.png", (mask * 255).astype(np.uint8))
