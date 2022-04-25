import numpy as np
import cv2

def draw_input(input):
    inputs = input[0].permute(1, 2, 0)*255
    text = {
        0: "depth",
        1: "compound",
        2: "boss",
        3: "hook",
        4: "rib",
        5: "ucut"
    }
    images = []
    for i in range(6):
        input = inputs[:, :, i]
        if i!= 0:
            input = input * 255
        input = np.expand_dims(input, -1)
        input = np.repeat(input, 3, -1)
        cv2.putText(input, text[i], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        images.append(input)
    return images

def overlay_mask(image, mask, text=None, color=[0, 0, 255]):
    mask = mask/255
    color_mask = np.array(color, dtype=np.uint8)
    vis = image.copy()
    mask = np.expand_dims(mask, -1)
    vis = np.uint8(vis * (1-mask) + color_mask * mask)
    # vis = np.uint8(color_mask * mask)
    if text is not None:
        vis = cv2.putText(vis.copy(), text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return vis.copy()
