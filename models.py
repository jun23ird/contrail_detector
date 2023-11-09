import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import math
from tqdm.auto import tqdm
from timeit import default_timer as timer

import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.applications import ResNet50, ResNet101

import utils


class BinaryClassifier(keras.Model):

    def __init__(self):
        super().__init__()
        self.classifier = KM.Sequential([
            KL.Conv2D(16, (3,3), padding='same', activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal(), input_shape=(256,256,3)),
            KL.MaxPooling2D(2,2),
            KL.Conv2D(32, (3,3), padding='same', activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal()),
            KL.MaxPooling2D(2,2),
            KL.Conv2D(64, (3,3), padding='same', activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal()),
            KL.MaxPooling2D(2,2),
            KL.Conv2D(128, (3,3), padding='same', activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal()),
            KL.MaxPooling2D(2,2),
            KL.Conv2D(256, (3,3), padding='same', activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal()),
            KL.MaxPooling2D(2,2),
            KL.Conv2D(512, (3,3), padding='same', activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal()),
            KL.MaxPooling2D(2,2),

            KL.Flatten(),
            KL.Dense(512, activation=KL.LeakyReLU(alpha=0.1), kernel_initializer=GlorotNormal()),
            KL.Dense(1, activation='sigmoid')
])


class DoubleConv(keras.Model):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = KM.Sequential([
            KL.Conv2D(mid_channels, (3,3), padding='same', use_bias = True, kernel_initializer=GlorotNormal(), input_shape = (None, None, in_channels)),
            KL.BatchNormalization(),
            KL.Activation('relu'),
            KL.Conv2D(out_channels, (3,3), padding='same', use_bias = False, kernel_initializer=GlorotNormal(), input_shape = (None, None, mid_channels)),
            KL.BatchNormalization(),
            KL.Activation('relu')]
        )

    def call(self, x, training = False):
        return self.double_conv(x, training = training)

class Down(keras.Model):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = KM.Sequential([
            KL.MaxPooling2D((2, 2), strides = 2, padding="same"),
            DoubleConv(in_channels, out_channels)]
        )

    def call(self, x, training = False):
        return self.maxpool_conv(x, training = training)


class Up(keras.Model):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # use the normal convolutions to reduce the number of channels
        self.up = KL.UpSampling2D(size=(2,2), interpolation='bilinear')
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def call(self, x1, x2, training = False):
        x1 = self.up(x1)
        # input is HWC
        diffH = tf.shape(x2)[1] - tf.shape(x1)[1]
        diffW = tf.shape(x2)[2] - tf.shape(x1)[2]

        # Compute the paddings individually
        paddingH1 = diffH // 2
        paddingH2 = diffH - paddingH1

        paddingW1 = diffW // 2
        paddingW2 = diffW - paddingW1

        top_pad = tf.stack([paddingH1, paddingH2], axis=0)
        bottom_pad = tf.stack([paddingW1, paddingW2], axis=0)
        paddings = tf.stack([[0, 0], top_pad, bottom_pad, [0, 0]], axis=0)

        x1 = tf.pad(x1, paddings, "CONSTANT")

        x = tf.concat([x2, x1], axis=-1)
        return self.conv(x, training = training)


class OutConv(keras.Model):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = KL.Conv2D(out_channels, (1,1), kernel_initializer=GlorotNormal(), input_shape = (None, None, in_channels))

    def call(self, x):
        return self.conv(x)


class unet(keras.Model):
    def __init__(self, n_channels=3, n_classes=2, shape_input = (256,256,3)):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shape_input = shape_input

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))
        self.up1 = (Up(1024+512, 512))
        self.up2 = (Up(512+256, 256))
        self.up3 = (Up(256+128, 128))
        self.up4 = (Up(128+64, 64))
        self.outc = (OutConv(64, n_classes))

    # @tf.recompute_grad
    def call(self, x, training = False):
        x1 = self.inc(x, training = training)
        x2 = self.down1(x1, training = training)
        x3 = self.down2(x2, training = training)
        x4 = self.down3(x3, training = training)
        x5 = self.down4(x4, training = training)
        x = self.up1(x5, x4, training = training)
        x = self.up2(x, x3, training = training)
        x = self.up3(x, x2, training = training)
        x = self.up4(x, x1, training = training)
        logits = self.outc(x)
        probs = tf.math.sigmoid(logits)
        return probs
    

############################################################
#  Miscellaneous Functions
############################################################

def compute_backbone_shapes(config=None):
    """
    Compute the width and height of the feature maps.

    Args:
        config (object, optional): Configuration object with image_shape and backbone_strides attributes.

    Returns:
        list : List containing computed backbone shapes. Each shape is 1D array of length 2.
    """
    backbone_strides = config.backbone_strides if config else [4, 8, 16, 32, 64]
    return np.array(
        [np.array([int(math.ceil(config.image_shape[0] / stride)), int(math.ceil(config.image_shape[1] / stride))]) for stride in backbone_strides])


def get_anchors(config):
    """
    Compute anchor boxes for the given config.

    Args:
        config (object): Configuration object.

    Returns:
        np.ndarray: Normalized anchor boxes.
    """
    # Typo: compute_backbone_shapes was expecting one argument but two were provided.
    backbone_shapes = compute_backbone_shapes(config)
    a = utils.generate_pyramid_anchors(
      config.rpn_anchor_scales,
      config.rpn_anchor_ratios,
      backbone_shapes,
      config.backbone_strides,
      config.rpn_anchor_stride)
    return utils.norm_boxes(a, config.image_shape[:2])

############################################################
#  Batch Normalization
############################################################

class BatchNorm(KL.BatchNormalization):
    """
    Batch Normalization class that wraps Keras's BatchNormalization to provide consistent behavior.
    """
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)

  
############################################################
#  FPN (Feature Pyramid Network) Graph
############################################################

class fpn_graph(keras.Model):
    """
    Creates a Feature Pyramid Network.

    Attributes:
        backbone(keras.Model): The backbone model for the FPN.
    """
    
    def __init__(self, backbone):
        """
        Initializes the FPN graph.

        Args:
            backbone(keras.Model): A keras model object that returns a list of feature maps.
        """
        super(fpn_graph, self).__init__()  
        self.backbone = backbone

        self.conv_skip_5 = KL.Conv2D(256, (1, 1), kernel_initializer=GlorotNormal())

        self.add_4 = KL.Add()
        self.upsample_4 = KL.UpSampling2D(size=(2, 2))
        self.conv_skip_4 = KL.Conv2D(256, (1, 1), kernel_initializer=GlorotNormal())

        self.add_3 = KL.Add()
        self.upsample_3 = KL.UpSampling2D(size=(2, 2))
        self.conv_skip_3 = KL.Conv2D(256, (1, 1), kernel_initializer=GlorotNormal())

        self.add_2 = KL.Add()
        self.upsample_2 = KL.UpSampling2D(size=(2, 2))
        self.conv_skip_2 = KL.Conv2D(256, (1, 1), kernel_initializer=GlorotNormal())

        self.conv_out_2 = KL.Conv2D(256, (3, 3), padding="SAME", kernel_initializer=GlorotNormal())
        self.conv_out_3 = KL.Conv2D(256, (3, 3), padding="SAME", kernel_initializer=GlorotNormal())
        self.conv_out_4 = KL.Conv2D(256, (3, 3), padding="SAME", kernel_initializer=GlorotNormal())
        self.conv_out_5 = KL.Conv2D(256, (3, 3), padding="SAME", kernel_initializer=GlorotNormal())
        self.maxpool_6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2)

    def call(self, input_images):
        """
        Calls the FPN graph for processing the input.

        Args:
            input_images(tf.Tensor): A tensor representing the input images.
            stage5(boolean, optional): A boolean to include Stage 5 in the processing. Defaults to False.
            train_bn(boolean, optional): A boolean indicating whether to train batch normalization. Defaults to True.

        Returns:
            Tuple containing feature maps for RPN and RCNN.
        """
        # Get feature maps from backbone
        C1, C2, C3, C4, C5 = self.backbone(input_images)

        # Top-down Layers
        P5 = self.conv_skip_5(C5)
        P4 = self.add_4([self.upsample_4(P5), self.conv_skip_4(C4)])
        P3 = self.add_3([self.upsample_3(P4), self.conv_skip_3(C3)])
        P2 = self.add_2([self.upsample_2(P3), self.conv_skip_2(C2)])

        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = self.conv_out_2(P2) # dim = [n_batches, 256/4, 256/4, 256]
        P3 = self.conv_out_3(P3) # dim = [n_batches, 256/8, 256/8, 256]
        P4 = self.conv_out_4(P4) # dim = [n_batches, 256/16, 256/16, 256]
        P5 = self.conv_out_5(P5) # dim = [n_batches, 256/32, 256/32, 256]

        # P6 is used for the 5th anchor scale in RPN. Generated by subsampling from P5 with stride of 2.
        P6 = self.maxpool_6(P5)

        # Return feature maps for RPN and RCNN.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]

        return rpn_feature_maps, rcnn_feature_maps

############################################################
#  Region Proposal Network (RPN) Graph
############################################################

class rpn_graph(keras.Model):
    """
    Region Proposal Network (RPN) model that generates region proposals 
    based on feature maps provided by a backbone network.
    
    Attributes:
        conv_shared (KL.Layer): Shared convolutional layer.
        conv_class_raw (KL.Layer): Conv layer for raw class scores.
        activ_class (KL.Layer): Activation layer for class probabilities.
        conv_bbox (KL.Layer): Conv layer for bounding box deltas.
    """
    
    def __init__(self, anchors_per_location, anchor_stride):
        """
        Initializes the RPN model.

        Args:
            anchors_per_location (int): Number of anchors per feature map location.
            anchor_stride (int): Stride of the anchor grid.
        """
        super(rpn_graph, self).__init__()  # Initializing the superclass
        
        self.conv_shared = KL.Conv2D(512, (3, 3), padding='same', 
                                     activation='relu', strides=anchor_stride,
                                     kernel_initializer=GlorotNormal())
        self.conv_class_raw = KL.Conv2D(anchors_per_location*2, (1, 1), 
                                        padding='valid', activation='linear',
                                        kernel_initializer=GlorotNormal())
        self.activ_class = KL.Activation("softmax")
        self.conv_bbox = KL.Conv2D(anchors_per_location*4, (1, 1), 
                                   padding="valid", activation='linear',
                                   kernel_initializer=GlorotNormal())

    def call(self, feature_map):
        """
        Forward pass for the RPN model.

        Args:
            feature_map (tf.Tensor): Feature map from the backbone network.

        Returns:
            list: [rpn_class_logits, rpn_class_probs, rpn_deltas]
            rpn_class_logits: [batch, H * W * anchors_per_location, 2] Anchor classifier logits (before softmax)
            rpn_probs: [batch, H * W * anchors_per_location, 2] Anchor classifier probabilities.
            rpn_deltas: [batch, H * W * anchors_per_location, (dy, dx, log(dh), log(dw))] Deltas to be
                      applied to anchors in normalized coordinates.
        """
        # Shared convolutional base of the RPN
        shared = self.conv_shared(feature_map)  # dim = [n_batches, height, width, 512]
        
        # Class prediction
        x = self.conv_class_raw(shared)  # dim = [n_batches, height, width, anchors_per_location*2]
        rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])  # dim = [n_batches, height*width* anchors_per_location, 2]
        rpn_class_probs = self.activ_class(rpn_class_logits)  # dim = [n_batches, height*width* anchors_per_location, 2]
        
        # Bounding box prediction
        x = self.conv_bbox(shared)  # dim = [n_batches, height, width, 4 * anchors_per_location]
        rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])  # dim = [n_batches, height*width* anchors_per_location, 4]

        return [rpn_class_logits, rpn_class_probs, rpn_deltas]

############################################################
#  Proposal Layer
############################################################


class ProposalLayer(KL.Layer):
    """
    Defines the proposal layer.
    Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    
    Attributes:
        proposal_count (int): Max number of proposals to output.
        nms_threshold (float): Threshold for NMS.
        config (object): Configuration object.
    """

    def __init__(self, proposal_count, nms_threshold, config, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        """Generates proposal boxes.
        
        Args:
            inputs (list): [scores, deltas, anchors]
              scores(tf.Tensor): [batch, n_anchors, (bg prob, fg prob)]
              deltas(tf.Tensor): [batch, n_anchors, (dy, dx, log(dh), log(dw))]
              anchors(tf.Tensor): [batch, n_anchors, (y1, x1, y2, x2)] anchors in normalized coordinates
        
        Returns:
            proposal_bboxes(tf.Tensor): Proposals in normalized coordinates.
        """
        scores = inputs[0][:, :, 1]
        deltas = inputs[1]
        deltas = deltas * np.reshape(self.config.rpn_bbox_std_dev, [1, 1, 4])
        anchors = inputs[2]

        pre_nms_limit = tf.minimum(self.config.pre_nms_limit, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True).indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y), self.config.batch_size)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y), self.config.batch_size)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x), self.config.batch_size)

        bboxes = utils.batch_slice([pre_nms_anchors, deltas], lambda x, y: utils.apply_box_deltas_graph(x, y), self.config.batch_size)

        window = np.array([0, 0, 1, 1], dtype=np.float32)
        bboxes = utils.batch_slice(bboxes, lambda x: utils.clip_boxes_graph(x, window), self.config.batch_size)

        def nms(bboxes, scores):
            indices = tf.image.non_max_suppression(bboxes, scores, self.proposal_count, self.nms_threshold)
            proposal_bboxes = tf.gather(bboxes, indices)
            padding = tf.maximum(self.proposal_count - tf.shape(proposal_bboxes)[0], 0)
            proposal_bboxes = tf.pad(proposal_bboxes, [(0, padding), (0, 0)])
            return proposal_bboxes

        proposal_bboxes = utils.batch_slice([bboxes, scores], nms, self.config.batch_size)
        return proposal_bboxes

############################################################
#  Detection Target Layer
############################################################


def detection_targets_graph(proposal_bboxes, gt_class_ids, gt_bboxes, config):
    """
    Generates detection targets for proposals.

    Args:
    - proposal_bboxes(tf.Tensor): [config.post_nms_rois_training, (y1, x1, y2, x2)] tensor of proposed bounding boxes.
    - gt_class_ids(tf.Tensor): [config.max_gt_instances] tensor of ground truth class IDs.
    - gt_bboxes(tf.Tensor): [config.max_gt_instances, (y1, x1, y2, x2)] tensor of ground truth bounding boxes in normalized coordinates.
    - config(object): Configuration settings.

    Returns:
    - rois(tf.Tensor): [config.train_rois_per_image, (y1, x1, y2, x2)] padded tensor containing positive and negative proposal bounding boxes in normalized coordinates.
    - roi_gt_class_ids(tf.Tensor): [config.train_rois_per_image] ground truth class IDs corresponding to the rois.
    - deltas(tf.Tensor): [config.train_rois_per_image, (dy, dx, log(dh), log(dw))] deltas required for refining rois in normalized coordinates.
    """

    # Trim zeros from the proposals and ground truth bounding boxes
    proposal_bboxes, _ = utils.trim_zeros_graph(proposal_bboxes)
    gt_bboxes, non_zeros = utils.trim_zeros_graph(gt_bboxes)
    # Trim zeros from ground truth class IDs
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)

    # Compute IoU overlaps of proposals with ground truth bounding boxes
    overlaps = utils.overlaps_graph(proposal_bboxes, gt_bboxes)

    # Determine maximum IoU overlap for each proposal
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    
    # Indices of proposals with IoU >= 0.5 with any ground truth
    positive_indices = tf.where(roi_iou_max >= 0.5)[:, 0]
    # Indices of proposals with IoU < 0.5 with every ground truth
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Select a random subset of positive proposals
    positive_count = int(config.train_rois_per_image * config.roi_positive_ratio)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    
    # Calculate the number of negative samples required
    r = 1.0 / config.roi_positive_ratio
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    
    # Gather positive and negative proposals
    positive_rois = tf.gather(proposal_bboxes, positive_indices)
    negative_rois = tf.gather(proposal_bboxes, negative_indices)

    # Find the ground truth bounding boxes that match each positive proposal
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_bbox_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_bboxes = tf.gather(gt_bboxes, roi_gt_bbox_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_bbox_assignment)

    # Calculate the delta transformation required for refining the proposals
    deltas = utils.box_refinements_graph(positive_rois, roi_gt_bboxes)
    deltas /= config.bbox_std_dev

    # Combine positive and negative samples and pad tensors
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.train_rois_per_image - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_bboxes = tf.pad(roi_gt_bboxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

    return rois, roi_gt_class_ids, deltas

class DetectionTargetLayer(KL.Layer):
    """
    Subsamples proposals and generates detection targets for training

    """
    
    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        """
        Compute detection targets.

        Args:
        inputs(list): [proposals, gt_class_ids, gt_bboxes]
        - proposals(tf.Tensor): 
        - gt_class_ids(tf.Tensor): 
        - gt_bboxes(tf.Tensor): 

        Returns:
        outputs(list) : [rois, target_class_ids, target_deltas]
        - rois(tf.Tensor): Padded tensor containing positive and negative proposal bboxes.
        - target_class_ids(tf.Tensor): Ground truth class IDs corresponding to the rois.
        - target_deltas(tf.Tensor): Deltas required for refining rois.
        """
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_bboxes = inputs[2]

        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_bboxes],
            lambda w, x, y: detection_targets_graph(w, x, y, self.config),
            self.config.batch_size)
        return outputs

############################################################
#  ROIAlign Layer
############################################################


class PyramidROIAlign(KL.Layer):
    """
    Implementation of ROI Align on feature pyramids.
    
    Attributes:
    pool_shape (tuple): [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
    """

    def __init__(self, pool_shape, **kwargs):
        """
        Initializes the PyramidROIAlign layer.
        
        Args:
        pool_shape (tuple): [pool_height, pool_width] of the output pooled regions. Usually [7, 7]
        """
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        """
        Apply ROI Align operation.
        
        Args:
        inputs (list): [bboxes, image_shape, feature_map_2, feature_map_3, feature_map_4, feature_map_5]
           - bboxes(tf.Tensor): [n_batches, n_bboxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             bounding boxes to fill the array.
          - image_shape (tf.Tensor): ??
          - feature_maps(tf.Tensor): List of feature maps from different levels of the pyramid.
                          Each is [n_batches, height, width, n_channels]
        
        Returns:
        tf.Tensor: Pooled feature maps.  [n_batches, n_bboxes, pool_height, pool_width, n_channels]
        """
        # Input bounding boxes
        bboxes = inputs[0]
        
        # Input image shape
        image_shape = inputs[1]
        
        # Feature maps of different levels
        feature_maps = inputs[2:]
        
        # Split bounding box coordinates
        y1, x1, y2, x2 = tf.split(bboxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        
        # Compute the area of the image
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        # Compute the appropriate feature pyramid level for each box
        roi_level = utils.log2_graph(tf.sqrt(h * w) / (256.0 / tf.sqrt(image_area)))
        roi_level = tf.minimum(5, tf.maximum(2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Pooled features and indices
        pooled = []
        bbox_to_level = []
        
        for i, level in enumerate(range(2, 6)):
            ix = tf.where(tf.equal(roi_level, level))
            level_bboxes = tf.gather_nd(bboxes, ix)

            bbox_indices = tf.cast(ix[:, 0], tf.int32)
            bbox_to_level.append(ix)

            # Ensure the gradients don't flow through this path
            level_bboxes = tf.stop_gradient(level_bboxes)
            bbox_indices = tf.stop_gradient(bbox_indices)

            # Use crop and resize to pool the features
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_bboxes, bbox_indices, self.pool_shape,
                method="bilinear"))

        # Combine the pooled features
        pooled = tf.concat(pooled, axis=0)
        bbox_to_level = tf.concat(bbox_to_level, axis=0)

        # Sort the pooled features using the bounding box indices
        sorting_tensor = bbox_to_level[:, 0] * 100000 + bbox_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(bbox_to_level)[0]).indices[::-1]
        pooled = tf.gather(pooled, ix)

        # Reshape to match the original shape
        shape = tf.concat([tf.shape(bboxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        
        return pooled

############################################################
#  Feature Pyramid Network(FPN) Head
############################################################

class fpn_classifier_graph(keras.Model):
    """
    Feature Pyramid Network Classifier Head for object detection.

    Attributes:
    config (Config): Configuration for the model.
    roialign (PyramidROIAlign): ROI Align layer.
    conv_1 (Layer): First convolutional layer.
    bn_1 (Layer): Batch normalization for the first convolutional layer.
    conv_2 (Layer): Second convolutional layer.
    bn_2 (Layer): Batch normalization for the second convolutional layer.
    pool_squeeze (Layer): Lambda layer to squeeze pooled feature dimensions.
    class_dense (Layer): Fully connected layer for class logits.
    class_activ (Layer): Activation layer for class probabilities.
    delta_dense (Layer): Fully connected layer for bounding box refinements.
    reshape (Layer): Lambda layer to reshape bounding box refinements.
    """

    def __init__(self, config, pool_size, n_classes=2, fc_layers_size=1024):
        """
        Initializes the FPN classifier head.

        Args:
        config (Config): Configuration for the model.
        pool_size (int): Size of the pooling layer.
        n_classes (int): Number of classes for classification.
        fc_layers_size (int, optional): Number of neurons in the fully connected layers. Defaults to 1024.
        """
        super(fpn_classifier_graph, self).__init__()

        self.config = config
        self.roialign = PyramidROIAlign([pool_size, pool_size])
        self.conv_1 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding="valid",
                                                   kernel_initializer=GlorotNormal()))
        self.bn_1 = KL.TimeDistributed(BatchNorm())
        self.conv_2 = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (1, 1), 
                                                   kernel_initializer=GlorotNormal()))
        self.bn_2 = KL.TimeDistributed(BatchNorm())
        self.pool_squeeze = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2))
        self.class_dense = KL.TimeDistributed(KL.Dense(n_classes, kernel_initializer=GlorotNormal()))
        self.class_activ = KL.TimeDistributed(KL.Activation("softmax"))
        self.delta_dense = KL.TimeDistributed(KL.Dense(n_classes * 4, activation='linear',
                                                       kernel_initializer=GlorotNormal()))
        self.reshape = KL.Lambda(lambda x: tf.reshape(x, [-1, tf.shape(x)[1], n_classes, 4]))

    def call(self, rois, feature_maps, train_bn=True):
        """
        Forward pass for the FPN classifier head.

        Args:
        rois (tf.Tensor): Regions of interest.
        feature_maps (list): List of feature maps from the backbone.
        train_bn (bool, optional): Whether batch norm layers are in training mode. Defaults to True.

        Returns:
        rcnn_class_logits(tf.Tensor): [n_batches, n_rois, n_classes] classifier logits (before softmax)
        rcnn_class_probs(tf.Tensor): [n_batches, n_rois, n_classes] classifier probabilities
        rcnn_deltas(tf.Tensor):  [n_batches, n_rois, n_classes, (dy, dx, log(dh), log(dw))] Deltas to apply to
                     proposal bounding boxes in normalized coordinates.
        """
        x = self.roialign([rois, tf.convert_to_tensor(self.config.image_shape)] + feature_maps) # shape: [n_batches, n_rois, pool_size, pool_size, channels]
        x = self.conv_1(x) # shape: [n_batches, n_rois, 1, 1, fc_layers_size]
        x = self.bn_1(x, training=train_bn)
        x = KL.Activation('relu')(x)
        x = self.conv_2(x)  # shape: [n_batches, n_rois, 1, 1, fc_layers_size]
        x = self.bn_2(x, training=train_bn)
        x = KL.Activation('relu')(x)

        shared = self.pool_squeeze(x) # shape: [n_batches, n_rois, fc_layers_size]
        rcnn_class_logits = self.class_dense(shared) # shape: [n_batches, n_rois, n_classes]
        rcnn_probs = self.class_activ(rcnn_class_logits) # shape: [n_batches, n_rois, n_classes]
        x = self.delta_dense(shared) # shape: [n_batches, n_rois, n_classes * 4]
        rcnn_deltas = self.reshape(x) # shape: [n_batches, n_rois, n_classes, 4]

        return rcnn_class_logits, rcnn_probs, rcnn_deltas

############################################################
#  Build RPN Targets
############################################################


def build_rpn_targets(anchors, gt_bboxes, config):
    """Given the anchors and ground truth bounding boxes, compute anchor IOU scores and
    get RPN targets for training.

    Args:
        anchors(np.ndarray): [n_anchors, (y1, x1, y2, x2)] anchors defined in image coordinates.
        gt_class_ids(np.array): [n_gt_bboxes] Integer class IDs.
        gt_bboxes(np.ndarray): [n_gt_bboxes, (y1, x1, y2, x2)]
        config(Config): Configuration object.

    Returns:
        rpn_matches(np.array): [N] (int32) matches between anchors and ground truth bounding boxes.
        rpn_deltas(np.ndarray): [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    
    # Anchors
    rpn_matches = np.zeros([anchors.shape[0]], dtype=np.int32)
    rpn_deltas = np.zeros((config.rpn_train_anchors_per_image, 4))

    # Compute overlaps [n_anchors, n_gt_bboxes]
    overlaps = utils.compute_overlaps(anchors, gt_bboxes)

    # Match anchors to ground truth bounding boxes
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]

    # Negative anchors have < 0.3 IoU
    rpn_matches[anchor_iou_max < 0.3] = -1

    # Anchors with max IoU 
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:, 0]
    rpn_matches[gt_iou_argmax] = 1

    # Positive anchors have >= 0.7 IoU
    rpn_matches[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    ids = np.where(rpn_matches == 1)[0]
    extra = len(ids) - (config.rpn_train_anchors_per_image // 2)
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_matches[ids] = 0

    ids = np.where(rpn_matches == -1)[0]
    extra = len(ids) - (config.rpn_train_anchors_per_image - np.sum(rpn_matches == 1))
    if extra > 0:
        ids = np.random.choice(ids, extra, replace=False)
        rpn_matches[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT bounding boxes.
    ids = np.where(rpn_matches == 1)[0]
    ix = 0  
    for i, a in zip(ids, anchors[ids]):
        gt = gt_bboxes[anchor_iou_argmax[i]]
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w

        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        rpn_deltas[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        rpn_deltas[ix] /= config.rpn_bbox_std_dev
        ix += 1

    return rpn_matches, rpn_deltas

############################################################
#  Loss Functions
############################################################

def batch_pack_graph(x, counts, n_rows):
    """
    Combine batches of various sizes into a single tensor.
    
    Args:
        x (tf.Tensor): Input tensor.
        counts (tf.Tensor): Number of valid entries for each batch.
        n_rows (int): Number of rows in the tensor.

    Returns:
        tf.Tensor: Combined tensor.
    """
    outputs = []
    for i in range(n_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)

def smooth_l1_loss(y_true, y_pred):
    """
    Implement the Smooth L1 loss.
    
    Args:
        y_true (tf.Tensor): Ground truth values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Loss value.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_matches, rpn_class_logits):
    """
    RPN anchor classifier loss.

    Args:
        rpn_matches (tf.Tensor): [n_batches, n_anchors, 1]. Anchor match type. 1=positive, -1=negative, 0=neutral anchor.
        rpn_class_logits (tf.Tensor): [n_batches, n_anchors, 2]. Logits for RPN classifier.

    Returns:
        tf.Tensor: Loss value.
    """
    # Anchor class. Convert anchor match type to class type (1=positive, 0=negative, ignores neutral)
    rpn_matches = tf.squeeze(rpn_matches, -1)
    anchor_class = K.cast(K.equal(rpn_matches, 1), tf.int32)
    indices = tf.where(K.not_equal(rpn_matches, 0))
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Use categorical cross-entropy on the positive and negative anchors
    loss = K.sparse_categorical_crossentropy(target=anchor_class,
                                             output=rpn_class_logits,
                                             from_logits=True)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss



def rpn_bbox_loss_graph(config, target_rpn_deltas, rpn_matches, rpn_deltas):
    """
    Compute the RPN bounding box loss graph.

    Args:
        config (Config): Configuration object with various hyperparameters.
        target_rpn_deltas (tf.Tensor): [n_batches, max_positive_anchors, (dy, dx, log(dh), log(dw))]. 
            Ground truth bounding box deltas.
        rpn_matches (tf.Tensor): [n_batches, n_anchors, 1]. Match values (1=positive anchor, -1=negative, 0=neutral).
        rpn_deltas (tf.Tensor): [n_batches, n_anchors, (dy, dx, log(dh), log(dw))]. 
            Predicted bounding box deltas.

    Returns:
        tf.Tensor: RPN bounding box loss.
    """
    # Squeeze to remove last dimension
    rpn_matches = K.squeeze(rpn_matches, -1)
    # Get the positive anchors
    indices = tf.where(K.equal(rpn_matches, 1))
    # Get predicted bounding box deltas for the positive anchors
    rpn_deltas = tf.gather_nd(rpn_deltas, indices)
    # Count the number of positive anchors
    batch_counts = K.sum(K.cast(K.equal(rpn_matches, 1), tf.int32), axis=1)
    # Pack the ground truth bounding boxes according to batch counts
    target_rpn_deltas = utils.batch_pack_graph(target_rpn_deltas, batch_counts, config.batch_size)
    # Compute the Smooth L1 loss
    loss = smooth_l1_loss(target_rpn_deltas, rpn_deltas)
    # If no positive anchors, return 0
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def rcnn_class_loss_graph(target_class_ids, rcnn_class_logits):
    """
    Compute the RCNN class loss graph.

    Args:
        target_class_ids (tf.Tensor): [n_batches, n_rois]. Ground truth class IDs.
        pred_class_logits (tf.Tensor): [n_batches, n_rois, n_classes]. Predicted class logits.

    Returns:
        tf.Tensor: RCNN class loss.
    """
    # Cast target class ids to int64 for compatibility with tf operations
    target_class_ids = tf.cast(target_class_ids, tf.int64)
    # Compute the sparse softmax cross entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class_ids, logits=rcnn_class_logits)
    # If there are no target class IDs, return 0
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

def rcnn_bbox_loss_graph(target_rcnn_deltas, target_class_ids, rcnn_deltas):
    """
    Compute the RCNN bounding box loss graph.

    Args:
        target_rcnn_deltas (tf.Tensor): [n_batches, n_rois, (dy, dx, log(dh), log(dw))]. 
            Ground truth bounding box deltas.
        target_class_ids (tf.Tensor): [n_batches, n_rois]. Ground truth class IDs.
        rcnn_deltas (tf.Tensor): [n_batches, n_rois, n_classes, (dy, dx, log(dh), log(dw))]. 
            Predicted bounding box deltas.

    Returns:
        tf.Tensor: RCNN bounding box loss.
    """
    # Reshape inputs
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_rcnn_deltas = K.reshape(target_rcnn_deltas, (-1, 4))
    rcnn_deltas = K.reshape(rcnn_deltas, (-1, K.int_shape(rcnn_deltas)[2], 4))
    # Get positive ROI indices and corresponding class ids
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)
    # Get corresponding target and predicted bounding boxes
    target_rcnn_deltas = tf.gather(target_rcnn_deltas, positive_roi_ix)
    rcnn_deltas = tf.gather_nd(rcnn_deltas, indices)
    # Compute the Smooth L1 loss
    loss = K.switch(tf.size(target_rcnn_deltas) > 0, smooth_l1_loss(y_true=target_rcnn_deltas, y_pred=rcnn_deltas), tf.constant(0.0))
    loss = K.mean(loss)
    return loss

############################################################
#  Detection Layer
############################################################

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois(tf.Tensor): [N, (y1, x1, y2, x2)] in normalized coordinates
        probs(tf.Tensor): [N, n_classes]. Class probabilities.
        deltas(tf.Tensor): [N, n_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [n_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [n_bboxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = utils.apply_box_deltas_graph(
        rois, deltas_specific * config.bbox_std_dev)
    # Clip bounding boxes to image window
    refined_rois = utils.clip_boxes_graph(refined_rois, window)

    # TODO: Filter out bounding boxes with zero area

    # Filter out background bounding boxes
    keep = tf.where(class_ids > 0)[:, 0] #in global indexing
    # Filter out low confidence bounding boxes
    if config.detection_min_confidence:
        conf_keep = tf.where(class_scores >= config.detection_min_confidence)[:, 0] #in global indexing
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    # objects below are indexed using pre_nms indexing
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0] 

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0] # indices pre_nms indexing
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.detection_max_instances,
                iou_threshold=config.detection_nms_threshold) # local indexing for each class_id
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep)) # climb up the indexing convention twice: local indexing for each class_id -> pre_nms indexing -> global indexing
        # Pad with -1 so returned tensors have the same shape
        gap = config.detection_max_instances - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.detection_max_instances])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64) #output use global indexing
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.detection_max_instances
    class_scores_keep = tf.gather(class_scores, keep)
    n_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=n_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)
    attention_mask = utils.make_attention_mask(config.image_shape, detections[:,:4])

    # Pad with zeros if detections < detection_max_instances
    gap = config.detection_max_instances - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections, attention_mask


class DetectionLayer(KL.Layer):
    """Takes classified proposal bounding boxes and their bounding box deltas and
    returns the final detection bounding boxes.

    Returns:
    [n_batches, n_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        rcnn_class_probs = inputs[1]
        rcnn_deltas = inputs[2]

        # Get windows of images in normalized coordinates. 
        window = tf.constant([0, 0, 1, 1], dtype=tf.float32)

        # Run detection refinement graph on each item in the batch
        detections_batch, attention_mask_batch = utils.batch_slice(
            [rois, rcnn_class_probs, rcnn_deltas],
            lambda x, y, w: refine_detections_graph(x, y, w, window, self.config),
            self.config.batch_size)

        # Reshape output
        # [n_batches, n_detections, (y1, x1, y2, x2, class_id, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.batch_size, self.config.detection_max_instances, 6]), attention_mask_batch
    
############################################################
#  Region based Convolutional Neural Network(RCNN) Graph
############################################################

class rcnn(keras.Model):
    """
    Region-based Convolutional Neural Network (RCNN) for object detection.
    """
    def __init__(self, mode, config, architecture="resnet50"):
        """
        Initialize the RCNN model.

        Args:
            mode (str): Either 'training' or 'inference'.
            config (Config): Configuration object with model hyperparameters.
            architecture (str, optional): Backbone architecture. Defaults to "resnet50".
        """
        super(rcnn, self).__init__()  
        self.mode = mode
        self.config = config

        if architecture == "resnet50" : 
            input_tensor = KL.Input(shape=(config.image_shape[0], config.image_shape[1], config.image_shape[3]))
            resnet = ResNet50(weights='imagenet', input_tensor=input_tensor)
            stage_outputs = [
                resnet.get_layer(name).output for name in [
                    'conv1_relu',     # Stage 1
                    'conv2_block3_out',  # Stage 2
                    'conv3_block4_out',  # Stage 3
                    'conv4_block6_out',  # Stage 4
                    'conv5_block3_out'   # Stage 5
                ]
            ]
        else : 
            input_tensor = KL.Input(shape=(config.image_shape[0], config.image_shape[1], config.image_shape[3]))
            resnet = ResNet101(weights='imagenet', input_tensor=input_tensor)
            stage_outputs = [
                resnet.get_layer(name).output for name in [
                    'conv1_relu',     # Stage 1
                    'conv2_block3_out',  # Stage 2
                    'conv3_block4_out',  # Stage 3
                    'conv4_block6_out',  # Stage 4
                    'conv5_block3_out'   # Stage 5
                ]
            ]
        self.backbone = KM.Model(inputs=resnet.input, outputs=stage_outputs)
        self.fpn_graph = fpn_graph(self.backbone)
        self.rpn_graph = rpn_graph(len(config.rpn_anchor_ratios), config.rpn_anchor_stride)
        self.proposal_layer = ProposalLayer(
            proposal_count=config.post_nms_rois_training if mode == "training" else config.post_nms_rois_inference,
            nms_threshold=config.rpn_nms_threshold,
            config=config
        )
        self.detection_target_layer = DetectionTargetLayer(config)
        self.fpn_classifier_graph = fpn_classifier_graph(
            config, config.pool_size, config.n_classes, fc_layers_size=config.fpn_classif_fc_layers_size
        )
        self.detection_layer = DetectionLayer(config)

    def call(self, images, normalized_anchors=None, gt_class_ids=None, gt_bboxes=None):
        """
        Call the RCNN model for either training or inference.

        Args:
            images (tf.Tensor): Input images tensor.
            normalized_anchors (tf.Tensor, optional): Precomputed normalized anchors. Defaults to None. 
              Batch dimension should be included.
            gt_class_ids (tf.Tensor, optional): Ground truth class IDs. Defaults to None.
            gt_bboxes (tf.Tensor, optional): Ground truth bounding boxes. Defaults to None.

        Returns:
            dict: Output dictionary containing various model predictions and ground truths.
        """
        if self.mode == "training":
            normalized_gt_bboxes = KL.Lambda(lambda x: utils.norm_tensor_boxes(x, K.shape(images)[1:3]))(gt_bboxes)

        rpn_feature_maps, rcnn_feature_maps = self.fpn_graph(images)

        layer_outputs = [self.rpn_graph(p) for p in rpn_feature_maps]

        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1)(list(o)) for o in outputs]
        rpn_class_logits, rpn_class_probs, normalized_rpn_deltas = outputs

        normalized_proposal_bboxes = self.proposal_layer([rpn_class_probs, normalized_rpn_deltas, normalized_anchors])

        if self.mode == "training":
            normalized_rois, target_class_ids, normalized_target_deltas = self.detection_target_layer(
                [normalized_proposal_bboxes, gt_class_ids, normalized_gt_bboxes]
            )

            rcnn_class_logits, rcnn_class_probs, normalized_rcnn_deltas = self.fpn_classifier_graph(
                normalized_rois, rcnn_feature_maps, train_bn=self.config.train_bn
            )

            output_dict = {
                "rpn_class_logits": rpn_class_logits,
                "rpn_class_probs": rpn_class_probs,
                "normalized_rpn_deltas": normalized_rpn_deltas,
                "rcnn_class_logits": rcnn_class_logits,
                "rcnn_class_probs": rcnn_class_probs,
                "normalized_rcnn_deltas": normalized_rcnn_deltas,
                "target_class_ids": target_class_ids,
                "normalized_target_deltas": normalized_target_deltas
            }

            return output_dict

        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            rcnn_class_logits, rcnn_class_probs, normalized_rcnn_deltas = self.fpn_classifier_graph(
                normalized_proposal_bboxes, rcnn_feature_maps, train_bn=self.config.train_bn     
            )

            # Detections
            # output is [batch, n_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections, attention_masks = self.detection_layer(
                [normalized_proposal_bboxes, rcnn_class_probs, normalized_rcnn_deltas])
            
            
            gt_attention_masks = utils.batch_slice(
            [gt_bboxes],
            lambda x: utils.make_attention_mask(self.config.image_shape, x),
            self.config.batch_size)


            output_dict = {
                "detections" : detections,
                "rcnn_class_probs" : rcnn_class_probs,
                "normalized_rcnn_deltas" : normalized_rcnn_deltas,
                "normalized_proposal_bboxes" : normalized_proposal_bboxes,
                "rpn_class_probs" : rpn_class_probs,
                "normalized_rpn_deltas" : normalized_rpn_deltas,
                "attention_masks" : attention_masks,
                "gt_attention_masks" : gt_attention_masks
                }

            return output_dict
        
class wnet(keras.Model) :
  def __init__(self, unet, rcnn = None, binary_classifier = None) :
    super(wnet, self).__init__()
    self.binary_classifier = binary_classifier
    self.rcnn = rcnn
    self.unet = unet

  def call(self, images, normalized_anchors, gt_bboxes) :
    self.rcnn.mode = "inference"
    masks = self.unet(images)[...,-1:]

    if self.rcnn : 
      attention_masks = self.rcnn(images, normalized_anchors, gt_bboxes = gt_bboxes)["attention_masks"]
      attention_masks = tf.expand_dims(attention_masks, axis = -1)
    else :
      attention_masks = tf.ones((images.shape[0], 1, 1, 1))

    if self.binary_classifier :
      contrail_exists = self.binary_classifier(images)[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    else :
      contrail_exists = tf.ones((images.shape[0], 1, 1, 1))

    return masks, attention_masks, contrail_exists
