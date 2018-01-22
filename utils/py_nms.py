import numpy as np

def py_cpu_nms(rois, thresh):
    """
    Pure Python NMS baseline.
    Already Sorted

    return:
    keep: roi keep indice
    """
    y1 = rois[:, 0]
    x1 = rois[:, 1]
    y2 = rois[:, 2]
    x2 = rois[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    N = len(rois)
    order = np.array(range(N))

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep