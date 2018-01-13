import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import cv2

# ----------------------------------------------------------------
# VOC Objection Datasets
class VOCBboxDataset:
    """Bounding box dataset for PASCAL `VOC`_.

    .. _`VOC`: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

    The index corresponds to each image.

    When queried by an index, if :obj:`return_difficult == False`,
    this dataset returns a corresponding
    :obj:`img, bbox, label`, a tuple of an image, bounding boxes and labels.
    This is the default behaviour.
    If :obj:`return_difficult == True`, this dataset returns corresponding
    :obj:`img, bbox, label, difficult`. :obj:`difficult` is a boolean array
    that indicates whether bounding boxes are labeled as difficult or not.

    The bounding boxes are packed into a two dimensional tensor of shape
    :math:`(R, 4)`, where :math:`R` is the number of bounding boxes in
    the image. The second axis represents attributes of the bounding box.
    They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`, where the
    four attributes are coordinates of the top left and the bottom right
    vertices.

    The labels are packed into a one dimensional tensor of shape :math:`(R,)`.
    :math:`R` is the number of bounding boxes in the image.
    The class name of the label :math:`l` is :math:`l` th element of
    :obj:`VOC_BBOX_LABEL_NAMES`.

    The array :obj:`difficult` is a one dimensional boolean array of shape
    :math:`(R,)`. :math:`R` is the number of bounding boxes in the image.
    If :obj:`use_difficult` is :obj:`False`, this array is
    a boolean array with all :obj:`False`.

    The type of the image, the bounding boxes and the labels are as follows.

    * :obj:`img.dtype == numpy.float32`
    * :obj:`bbox.dtype == numpy.float32`
    * :obj:`label.dtype == numpy.int32`
    * :obj:`difficult.dtype == numpy.bool`

    Args:
        data_dir (string): Path to the root of the training data. 
            i.e. "/data/image/voc/VOCdevkit/VOC2007/"
        split ({'train', 'val', 'trainval', 'test'}): Select a split of the
            dataset. :obj:`test` split is only available for
            2007 dataset.
        year ({'2007', '2012'}): Use a dataset prepared for a challenge
            held in :obj:`year`.
        use_difficult (bool): If :obj:`True`, use images that are labeled as
            difficult in the original annotation.
        return_difficult (bool): If :obj:`True`, this dataset returns
            a boolean array
            that indicates whether bounding boxes are labeled as difficult
            or not. The default value is :obj:`False`.

    """

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes
            img: RGB image with shape [H, W, C], type float32

        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + '.xml'))
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when in not using difficult split, and the object is
            # difficult, skipt it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bndbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bndbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = plt.imread(img_file)
        img = img.astype(np.float32)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

# ----------------------------------------------------------------
# Datasets Using in Training and Testing
# 
# link: http://pytorch.org/docs/0.3.0/data.html?highlight=dataset
class VOCDataset(Dataset):
    """
    returned image: 
    scaled image (mean, std, /255), float32, HWC, RGB
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    """
    def __init__(self, opt, train=True):
        self.opt = opt
        self.train = train
        if train:
            self.db = VOCBboxDataset(opt.voc_data_dir)
        else:
            self.db = VOCBboxDataset(opt.voc_data_dir, split='test', use_difficult=True)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        
        # RESCALE ----------------------------
        # image rescale to [opt.min_size, opt.max_size]
        H0, W0, C = ori_img.shape
        scale = min(self.opt.min_size/min(H0, W0), self.opt.max_size/max(H0, W0))
        scaled_img = cv2.resize(ori_img, (0,0), fx=scale, fy=scale)
        H1, W1, _ = scaled_img.shape
        # bbox rescale
        bbox = scale * bbox
        
        # NORMALIZE ----------------------------
        normalized_img = scaled_img / 255.0
        normalized_img = (normalized_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # HORIZON FLIP ----------------------------
        if self.train and random.random() < 0.5:
            normalized_img = normalized_img[:,::-1,:]
            bbox[:,1], bbox[:,3] = W1 - bbox[:, 3], W1 - bbox[:, 1]
        return normalized_img.astype(np.float32), bbox, label, scale

    def __len__(self):
        return len(self.db)