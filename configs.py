import numpy as np


class binary_classifier_config:
    image_shape = np.arry([256, 256, 8, 3])

    initial_learning_rate = 0.001
    
    batch_size = 16
    buffer_size = 100
    n_times_before = 4
    n_times_after = 3
    n_epochs = 60


class unet_baseline_config:


    image_shape= np.array([256, 256, 8, 3])

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    initial_learning_rate = 0.01


    # Train or freeze batch normalization layers
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    train_bn = False  # Defaulting to False since batch size is often small

    batch_size = 20
    buffer_size = 10
    n_times_before = 4
    n_times_after = 3
    n_epochs = 60
    decay_steps = 60
    alpha = 0.0

class wnet_unet_config:


    image_shape= np.array([256, 256, 8, 3])

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    initial_learning_rate = 0.01


    # Train or freeze batch normalization layers
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    train_bn = False  # Defaulting to False since batch size is often small

    batch_size = 20
    buffer_size = 10
    n_times_before = 4
    n_times_after = 3
    n_epochs = 60
    decay_steps = 60
    alpha = 0.0

class alt_wnet_unet_config:


    image_shape= np.array([256, 256, 8, 3])

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    initial_learning_rate = 0.01


    # Train or freeze batch normalization layers
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    train_bn = False  # Defaulting to False since batch size is often small

    batch_size = 20
    buffer_size = 10
    n_times_before = 4
    n_times_after = 3
    n_epochs = 60
    decay_steps = 60
    alpha = 0.0

class wnet_rcnn_config:
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    backbone = "resnet50"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    backbone_strides = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    fpn_classif_fc_layers_size = 1024

    # Number of classification classes (including background)
    n_classes = 2  # Override in sub-classes

    # Length of square anchor side in pixels
    rpn_anchor_scales = (8, 16, 32, 64, 128)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    rpn_anchor_ratios = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    rpn_anchor_stride = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    rpn_nms_threshold = 0.7

    # How many anchors per image to use for RPN training
    rpn_train_anchors_per_image = 256

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    pre_nms_limit = 150

    # ROIs kept after non-maximum suppression (training and inference)
    post_nms_rois_training = 100
    post_nms_rois_inference = 100

    image_shape= np.array([256, 256, 8, 3])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    train_rois_per_image = 200

    # Percent of positive ROIs used to train classifier/mask heads
    roi_positive_ratio = 0.33

    # Pooled ROIs
    pool_size = 16

    # Maximum number of ground truth instances to use in one image
    max_gt_instances = 30

    # Bounding box refinement standard deviation for RPN and final detections.
    rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
    bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    detection_max_instances = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    detection_min_confidence = 0.7

    # Non-maximum suppression threshold for detection
    detection_nms_threshold = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    initial_learning_rate = 0.00001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    loss_weights = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }


    # Train or freeze batch normalization layers
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    train_bn = False  # Defaulting to False since batch size is often small

    decay_steps = 60
    alpha = 0

    # Gradient norm clipping
    gradient_clipnorm = 5.0

    batch_size = 20
    buffer_size = 10
    n_times_before = 4
    n_times_after = 3
    n_epochs = 60

class alt_wnet_rcnn_config:
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    backbone = "resnet50"

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    backbone_strides = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    fpn_classif_fc_layers_size = 1024

    # Number of classification classes (including background)
    n_classes = 2  # Override in sub-classes

    # Length of square anchor side in pixels
    rpn_anchor_scales = (8, 16, 32, 64, 128)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    rpn_anchor_ratios = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    rpn_anchor_stride = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    rpn_nms_threshold = 0.7

    # How many anchors per image to use for RPN training
    rpn_train_anchors_per_image = 256

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    pre_nms_limit = 150

    # ROIs kept after non-maximum suppression (training and inference)
    post_nms_rois_training = 100
    post_nms_rois_inference = 100

    image_shape= np.array([256, 256, 8, 3])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    train_rois_per_image = 200

    # Percent of positive ROIs used to train classifier/mask heads
    roi_positive_ratio = 0.33

    # Pooled ROIs
    pool_size = 16

    # Maximum number of ground truth instances to use in one image
    max_gt_instances = 30

    # Bounding box refinement standard deviation for RPN and final detections.
    rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
    bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    detection_max_instances = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    detection_min_confidence = 0.7

    # Non-maximum suppression threshold for detection
    detection_nms_threshold = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    initial_learning_rate = 0.00001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    loss_weights = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }


    # Train or freeze batch normalization layers
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    train_bn = False  # Defaulting to False since batch size is often small

    decay_steps = 60
    alpha = 0

    # Gradient norm clipping
    gradient_clipnorm = 5.0

    batch_size = 20
    buffer_size = 10
    n_times_before = 4
    n_times_after = 3
    n_epochs = 60

class Path :
    train = "/content/drive/MyDrive/kaggle/contrail_detector/data/train/"
    valid = "/content/drive/MyDrive/kaggle/contrail_detector/data/valid/"
    test = "/content/drive/MyDrive/kaggle/contrail_detector/data/test/"