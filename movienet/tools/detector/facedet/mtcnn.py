import numpy as np
import torch
from torch import nn
from torchvision.ops.boxes import batched_nms

from .utils import (batched_nms_numpy, bbreg, generateBoundingBox, imresample,
                    pad, rerec)


class PNet(nn.Module):
    """MTCNN PNet."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet."""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


class MTCNN(nn.Module):
    """MTCNN face detection module. This class loads pretrained P-, R-, and
    O-nets and returns images cropped to include the face only, given raw input
    images of one of the following types:

    - PIL image or list of PIL images
    - numpy.ndarray (uint8) representing either a single image (3D) or \
        a batch of images (4D).
    """

    def __init__(self,
                 min_face_size=20,
                 thresholds=[0.6, 0.7, 0.7],
                 factor=0.709):
        super().__init__()

        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor

        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()

    def forward(self, imgs):
        """
        Args:
            imgs: torch tensor
        """

        batch_size, _, h, w = imgs.shape
        m = 12.0 / self.min_face_size
        minl = min(h, w)
        minl = minl * m

        # Create scale pyramid
        scale_i = m
        scales = []
        while minl >= 12:
            scales.append(scale_i)
            scale_i = scale_i * self.factor
            minl = minl * self.factor

        # First stage
        boxes = []
        image_inds = []
        all_inds = []
        all_i = 0
        for scale in scales:
            im_data = imresample(imgs,
                                 (int(h * scale + 1), int(w * scale + 1)))
            # im_data = (im_data - 127.5) * 0.0078125
            with torch.no_grad():
                reg, probs = self.pnet(im_data)

            boxes_scale, image_inds_scale = generateBoundingBox(
                reg, probs[:, 1], scale, self.thresholds[0])
            boxes.append(boxes_scale)
            image_inds.append(image_inds_scale)
            all_inds.append(all_i + image_inds_scale)
            all_i += batch_size

        boxes = torch.cat(boxes, dim=0)
        image_inds = torch.cat(image_inds, dim=0).cpu()
        all_inds = torch.cat(all_inds, dim=0)

        # NMS within each scale + image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], all_inds, 0.5)
        boxes, image_inds = boxes[pick], image_inds[pick]

        # NMS within each image
        pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
        boxes, image_inds = boxes[pick], image_inds[pick]

        regw = boxes[:, 2] - boxes[:, 0]
        regh = boxes[:, 3] - boxes[:, 1]
        qq1 = boxes[:, 0] + boxes[:, 5] * regw
        qq2 = boxes[:, 1] + boxes[:, 6] * regh
        qq3 = boxes[:, 2] + boxes[:, 7] * regw
        qq4 = boxes[:, 3] + boxes[:, 8] * regh
        boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
        boxes = rerec(boxes)
        y, ey, x, ex = pad(boxes, w, h)

        # Second stage
        if len(boxes) > 0:
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k],
                                 (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (24, 24)))
            im_data = torch.cat(im_data, dim=0)
            # im_data = (im_data - 127.5) * 0.0078125
            with torch.no_grad():
                out = self.rnet(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            score = out1[1, :]
            ipass = score > self.thresholds[1]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)),
                              dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            # NMS within each image
            pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
            boxes = bbreg(boxes, mv)
            boxes = rerec(boxes)

        # Third stage
        points = torch.zeros(0, 5, 2, device=imgs.get_device())
        if len(boxes) > 0:
            y, ey, x, ex = pad(boxes, w, h)
            im_data = []
            for k in range(len(y)):
                if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
                    img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k],
                                 (x[k] - 1):ex[k]].unsqueeze(0)
                    im_data.append(imresample(img_k, (48, 48)))
            im_data = torch.cat(im_data, dim=0)
            # im_data = (im_data - 127.5) * 0.0078125
            with torch.no_grad():
                out = self.onet(im_data)

            out0 = out[0].permute(1, 0)
            out1 = out[1].permute(1, 0)
            out2 = out[2].permute(1, 0)
            score = out2[1, :]
            points = out1
            ipass = score > self.thresholds[2]
            points = points[:, ipass]
            boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)),
                              dim=1)
            image_inds = image_inds[ipass]
            mv = out0[:, ipass].permute(1, 0)

            w_i = boxes[:, 2] - boxes[:, 0] + 1
            h_i = boxes[:, 3] - boxes[:, 1] + 1
            points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(
                5, 1) - 1
            points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(
                5, 1) - 1
            points = torch.stack((points_x, points_y)).permute(2, 1, 0)
            boxes = bbreg(boxes, mv)

            # NMS within each image using "Min" strategy
            # pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
            pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds,
                                     0.7, 'Min')
            boxes, image_inds, points = boxes[pick], image_inds[pick], points[
                pick]

        boxes = boxes.cpu().numpy()
        points = points.cpu().numpy()

        batch_boxes = []
        batch_points = []
        for b_i in range(batch_size):
            b_i_inds = np.where(image_inds == b_i)
            batch_boxes.append(boxes[b_i_inds].copy())
            batch_points.append(points[b_i_inds].copy())

        batch_boxes, batch_points = np.array(batch_boxes), np.array(
            batch_points)

        return batch_boxes, batch_points
