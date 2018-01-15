import numpy as np

def generate_anchor_base(side_length=16, ratios=[0.5, 1, 2],
                         scales=[0.5, 1, 2], strides=16):
    """
	Generate anchors for a single 16*16 block. Then transform the anchors
	to the original image space.
	
	Input:
	side_length: block side length
	ratios
	scales
	strides: network strides

	Return
	anchor_base: base anchor of the original image
    """
    py = side_length / 2.
    px = side_length / 2.

    anchor_base = np.zeros((len(ratios) * len(scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(scales)):
            h = side_length * strides * scales[j] * np.sqrt(ratios[i])
            w = side_length * strides * scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base


def get_anchors(anchor_base, feat_stride, height, width):
    anchors_y = np.arange(height) * feat_stride
    anchors_x = np.arange(width) * feat_stride
    anchors_x, anchors_y = np.meshgrid(anchors_x, anchors_y)
    shift = np.stack((anchors_y.ravel(), anchors_x.ravel(),
                      anchors_y.ravel(), anchors_x.ravel()), axis=1)
    anchors = np.repeat(shift, repeats=len(anchor_base), axis=0) + \
        np.tile(anchor_base, [len(shift),1])
    return anchors