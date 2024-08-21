import numpy as np
from typing import cast, Optional, Any
from PIL import Image
from rembg.bg import (fix_image_orientation,
                      post_process,
                      alpha_matting_cutout,
                      putalpha_cutout,
                      naive_cutout,
                      get_concat_v_multi)
from rembg.sessions.base import BaseSession


REMBG_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
]


def run_rembg(session: BaseSession,
              src_image: np.ndarray,
              post_process_mask: bool = False,
              alpha_matting: bool = False,
              alpha_matting_foreground_threshold: int = 240,
              alpha_matting_background_threshold: int = 10,
              alpha_matting_erode_size: int = 10,
              *args: Optional[Any],
              **kwargs: Optional[Any]):
    put_alpha = kwargs.pop("putalpha", False)
    img = cast(Image, Image.fromarray(src_image))
    # Fix image orientation
    img = fix_image_orientation(img)

    raw_masks = session.predict(img, *args, **kwargs)
    masks = []
    cutouts = []

    if post_process_mask:
        for mask in raw_masks:
            masks.append(Image.fromarray(post_process(np.array(mask))))
    else:
        masks = raw_masks

    for mask in masks:
        if alpha_matting:
            try:
                cutout = alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_size,
                )
            except ValueError:
                if put_alpha:
                    cutout = putalpha_cutout(img, mask)
                else:
                    cutout = naive_cutout(img, mask)
        else:
            if put_alpha:
                cutout = putalpha_cutout(img, mask)
            else:
                cutout = naive_cutout(img, mask)

        cutouts.append(cutout)

    if len(cutouts) > 0:
        cutout = get_concat_v_multi(cutouts)
    else:
        cutout = img

    if len(masks) > 0:
        mask_out = get_concat_v_multi(masks)
    else:
        w, h = src_image.shape
        mask_out = np.zeros((w, h))

    return np.asarray(mask_out), np.asarray(cutout)
