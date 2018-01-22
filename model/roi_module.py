from collections import namedtuple
from string import Template

import cupy, torch
import cupy as cp
import torch as t
from torch import nn
from torch.autograd import Function

from utils import array_tool as at


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier.cuda()
        self.cls_loc = nn.Linear(4096, n_class * 4).cuda()
        self.score = nn.Linear(4096, n_class).cuda()

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = t.autograd.Variable(xy_indices_and_rois.contiguous())

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

Stream = namedtuple('Stream', ['ptr'])


@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    cp.cuda.runtime.free(0)
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024


def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


class RoI(Function):
    """
    NOTE：only CUDA-compatible
    """

    def __init__(self, outh, outw, spatial_scale):
        self.forward_fn = load_kernel('roi_forward', kernel_forward)
        self.backward_fn = load_kernel('roi_backward', kernel_backward)
        self.outh, self.outw, self.spatial_scale = outh, outw, spatial_scale

    def forward(self, x, rois):
        # NOTE: MAKE SURE input is contiguous too
        x = x.contiguous()
        rois = rois.contiguous()
        self.in_size = B, C, H, W = x.size()
        self.N = N = rois.size(0)
        output = t.zeros(N, C, self.outh, self.outw).cuda()
        self.argmax_data = t.zeros(N, C, self.outh, self.outw).int().cuda()
        self.rois = rois
        args = [x.data_ptr(), rois.data_ptr(),
                output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.spatial_scale, C, H, W,
                self.outh, self.outw,
                output.numel()]
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        self.forward_fn(args=args,
                        block=(CUDA_NUM_THREADS, 1, 1),
                        grid=(GET_BLOCKS(output.numel()), 1, 1),
                        stream=stream)
        return output

    def backward(self, grad_output):
        ##NOTE: IMPORTANT CONTIGUOUS
        # TODO: input
        grad_output = grad_output.contiguous()
        B, C, H, W = self.in_size
        grad_input = t.zeros(self.in_size).cuda()
        stream = Stream(ptr=torch.cuda.current_stream().cuda_stream)
        args = [grad_output.data_ptr(),
                self.argmax_data.data_ptr(),
                self.rois.data_ptr(),
                grad_input.data_ptr(),
                self.N, self.spatial_scale, C, H, W, self.outh, self.outw,
                grad_input.numel()]
        self.backward_fn(args=args,
                         block=(CUDA_NUM_THREADS, 1, 1),
                         grid=(GET_BLOCKS(grad_input.numel()), 1, 1),
                         stream=stream
                         )
        return grad_input, None


class RoIPooling2D(nn.Module):

    def __init__(self, outh, outw, spatial_scale):
        super(RoIPooling2D, self).__init__()
        self.RoI = RoI(outh, outw, spatial_scale)

    def forward(self, x, rois):
        return self.RoI(x, rois)


def test_roi_module():
    ## fake data###
    B, N, C, H, W, PH, PW = 2, 8, 4, 32, 32, 7, 7

    bottom_data = t.randn(B, C, H, W).cuda()
    bottom_rois = t.randn(N, 5)
    bottom_rois[:int(N / 2), 0] = 0
    bottom_rois[int(N / 2):, 0] = 1
    bottom_rois[:, 1:] = (t.rand(N, 4) * 100).float()
    bottom_rois = bottom_rois.cuda()
    spatial_scale = 1. / 16
    outh, outw = PH, PW

    # pytorch version
    module = RoIPooling2D(outh, outw, spatial_scale)
    x = t.autograd.Variable(bottom_data, requires_grad=True)
    rois = t.autograd.Variable(bottom_rois)
    output = module(x, rois)
    output.sum().backward()

    def t2c(variable):
        npa = variable.data.cpu().numpy()
        return cp.array(npa)

    def test_eq(variable, array, info):
        cc = cp.asnumpy(array)
        neq = (cc != variable.data.cpu().numpy())
        assert neq.sum() == 0, 'test failed: %s' % info

    # chainer version,if you're going to run this
    # pip install chainer 
    import chainer.functions as F
    from chainer import Variable
    x_cn = Variable(t2c(x))

    o_cn = F.roi_pooling_2d(x_cn, t2c(rois), outh, outw, spatial_scale)
    test_eq(output, o_cn.array, 'forward')
    F.sum(o_cn).backward()
    test_eq(x.grad, x_cn.grad, 'backward')
    print('test pass')


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


kernel_forward = '''
    extern "C"
    __global__ void roi_forward(const float* const bottom_data,const float* const bottom_rois,
                float* top_data, int* argmax_data,
                const double spatial_scale,const int channels,const int height, 
                const int width, const int pooled_height, 
                const int pooled_width,const int NN
    ){
        
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=NN)
        return;
    const int pw = idx % pooled_width;
    const int ph = (idx / pooled_width) % pooled_height;
    const int c = (idx / pooled_width / pooled_height) % channels;
    int num = idx / pooled_width / pooled_height / channels;
    const int roi_batch_ind = bottom_rois[num * 5 + 0];
    const int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);
    const int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);
    const int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);
    const int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);
    // Force malformed ROIs to be 1x1
    const int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    const int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    const float bin_size_h = static_cast<float>(roi_height)
                    / static_cast<float>(pooled_height);
    const float bin_size_w = static_cast<float>(roi_width)
                    / static_cast<float>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                    * bin_size_h));
        int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                    * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                                * bin_size_h));
        int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                                * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    float maxval = is_empty ? 0 : -1E+37;
    // If nothing is pooled, argmax=-1 causes nothing to be backprop'd
    int maxidx = -1;
    const int data_offset = (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
            int bottom_index = h * width + w;
            if (bottom_data[data_offset + bottom_index] > maxval) {
                maxval = bottom_data[data_offset + bottom_index];
                maxidx = bottom_index;
            }
        }
    }
    top_data[idx]=maxval;
    argmax_data[idx]=maxidx;
    }
'''
kernel_backward = '''
    extern "C"
    __global__ void roi_backward(const float* const top_diff,
         const int* const argmax_data,const float* const bottom_rois,
         float* bottom_diff, const int num_rois,
         const double spatial_scale, int channels,
         int height, int width, int pooled_height,
          int pooled_width,const int NN)
    {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ////Importtan >= instead of >
    if(idx>=NN)
        return;
    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx/ (width * height)) % channels;
    int num = idx / (width * height * channels);

    float gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
        // Skip if ROI's batch index doesn't match num
        if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
            continue;
        }

        int roi_start_w = round(bottom_rois[roi_n * 5 + 1]
                                * spatial_scale);
        int roi_start_h = round(bottom_rois[roi_n * 5 + 2]
                                * spatial_scale);
        int roi_end_w = round(bottom_rois[roi_n * 5 + 3]
                                * spatial_scale);
        int roi_end_h = round(bottom_rois[roi_n * 5 + 4]
                                * spatial_scale);

        // Skip if ROI doesn't include (h, w)
        const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                h >= roi_start_h && h <= roi_end_h);
        if (!in_roi) {
            continue;
        }

        int offset = (roi_n * channels + c) * pooled_height
                        * pooled_width;

        // Compute feasible set of pooled units that could have pooled
        // this bottom unit

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);

        float bin_size_h = static_cast<float>(roi_height)
                        / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width)
                        / static_cast<float>(pooled_width);

        int phstart = floor(static_cast<float>(h - roi_start_h)
                            / bin_size_h);
        int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                            / bin_size_h);
        int pwstart = floor(static_cast<float>(w - roi_start_w)
                            / bin_size_w);
        int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                            / bin_size_w);

        phstart = min(max(phstart, 0), pooled_height);
        phend = min(max(phend, 0), pooled_height);
        pwstart = min(max(pwstart, 0), pooled_width);
        pwend = min(max(pwend, 0), pooled_width);
        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int index_ = ph * pooled_width + pw + offset;
                if (argmax_data[index_] == (h * width + w)) {
                    gradient += top_diff[index_];
                }
            }
        }
    }
    bottom_diff[idx] = gradient;
    }
'''
