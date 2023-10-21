import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from collections import deque
from tqdm.auto import tqdm
from timeit import default_timer as timer


def norm_tensor_boxes(boxes, shape):
    """
    Normalize the box coordinates by the given image shape.

    Args:
        boxes (tf.Tensor): Box coordinates to be normalized.
        shape (tf.Tensor): Shape of the image. TensorFlow tensor of length 2.

    Returns:
        tf.Tensor: Normalized box coordinates.
    """
    h, w = tf.cast(shape[0], dtype = tf.float32), tf.cast(shape[1], dtype = tf.float32)
    
    # Creating the scale tensor using tf operations
    h_w_tensors = tf.stack([h, w, h, w])
    ones = tf.ones_like(h_w_tensors, dtype=tf.float32)
    scale = h_w_tensors - ones
    
    shift = tf.constant([0, 0, 1, 1], dtype=tf.float32)
    
    return tf.divide(tf.subtract(boxes, shift), scale)

def norm_boxes(boxes, shape):
    """
    Normalize the box coordinates by the given image shape.

    Args:
        boxes (np.ndarray): Box coordinates to be normalized.
        shape (np.array): Shape of the image. Numpy array of length 2.

    Returns:
        np.ndarray: Normalized box coordinates.
    """
    h, w = shape
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    Generate anchor boxes for the given configurations.

    Args:
        scales (list): List of scales for anchors.
        ratios (list): List of aspect ratios for anchors.
        shape (np.array): Shape of the feature map.
        feature_stride (int): Stride of the feature map.
        anchor_stride (int): Stride of anchors on the feature map.

    Returns:
        np.ndarray: Generated anchor boxes.
    """
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)

    anchor_widths, anchor_centers_x = np.meshgrid(widths, shifts_x)
    anchor_heights, anchor_centers_y = np.meshgrid(heights, shifts_y)

    anchor_centers = np.stack([anchor_centers_y, anchor_centers_x], axis=2).reshape([-1, 2])
    anchor_sizes = np.stack([anchor_heights, anchor_widths], axis=2).reshape([-1, 2])

    anchors = np.concatenate([anchor_centers - 0.5 * anchor_sizes, anchor_centers + 0.5 * anchor_sizes], axis=1)
    return anchors

def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides, anchor_stride):
    """
    Generate pyramid of anchor boxes for different scales.

    Args:
        scales (list): List of scales for each feature map.
        ratios (list): List of aspect ratios for anchors.
        feature_shapes (list): List of feature map shapes.
        feature_strides (list): List of feature map strides.
        anchor_stride (int): Stride of anchors on the feature map.

    Returns:
        np.ndarray: Pyramid of anchor boxes.
    """
    anchors = []
    for i in range(len(scales)):
        anchors.append(generate_anchors(scales[i], ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    return np.concatenate(anchors, axis=0)

def batch_slice(inputs, graph_fn, batch_size):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.
    
    Args:
        inputs (list): List of tensors. 
        graph_fn (function): A function to apply on each slice from inputs.
        batch_size (int): Number of slices.
        
    Returns:
        tf.Tensor or list of tf.Tensor
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    outputs = list(zip(*outputs))


    result = [tf.stack(o, axis=0) for o in outputs]
    if len(result) == 1:
        result = result[0]

    return result

def apply_box_deltas_graph(boxes, deltas):
    """Applies deltas to boxes to get refined boxes.
    
    Args:
        boxes (tf.Tensor): [N, 4] where each row is [y1, x1, y2, x2].
        deltas (tf.Tensor): [N, 4] where each row represents dx, dy, log(dh), log(dw).
        
    Returns:
        tf.Tensor: Refined boxes.
    """
    heights = boxes[:, 2] - boxes[:, 0]
    widths = boxes[:, 3] - boxes[:, 1]
    centers_y = boxes[:, 0] + 0.5 * heights
    centers_x = boxes[:, 1] + 0.5 * widths

    centers_y += deltas[:, 0] * heights
    centers_x += deltas[:, 1] * widths
    heights *= tf.exp(deltas[:, 2])
    widths *= tf.exp(deltas[:, 3])
    
    y1 = centers_y - 0.5 * heights
    x1 = centers_x - 0.5 * widths
    y2 = y1 + heights
    x2 = x1 + widths
    result = tf.stack([y1, x1, y2, x2], axis=1)
    return result

def clip_boxes_graph(boxes, window):
    """Clips boxes to a window.
    
    Args:
        boxes (tf.Tensor): [N, 4] each row is [y1, x1, y2, x2].
        window (tf.Tensor): [4] in the form [y1, x1, y2, x2].
        
    Returns:
        tf.Tensor: Clipped boxes.
    """
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1)
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


def box_refinements_graph(boxes, gt_boxes):
  """
    Compute refinement needed to transform `boxes` to `gt_box`.

    Args:
        boxes(tf.Tensor): [N, 4] where each row is y1, x1, y2, x2
        gt_boxes(tf.Tensor): [N, 4] Ground truth boxes.

    Returns:
        result(tf.Tensor): [N, 4] where each row is [dy, dx, log(dh), log(dw)]
  """
  boxes = tf.cast(boxes, tf.float32)
  gt_boxes = tf.cast(gt_boxes, tf.float32)

  heights = boxes[:, 2] - boxes[:, 0]
  widths = boxes[:, 3] - boxes[:, 1]
  centers_y = boxes[:, 0] + 0.5 * heights
  centers_x = boxes[:, 1] + 0.5 * widths

  gt_heights = gt_boxes[:, 2] - gt_boxes[:, 0]
  gt_widths = gt_boxes[:, 3] - gt_boxes[:, 1]
  gt_centers_y = gt_boxes[:, 0] + 0.5 * gt_heights
  gt_centers_x = gt_boxes[:, 1] + 0.5 * gt_widths

  dy = (gt_centers_y - centers_y) / heights
  dx = (gt_centers_x - centers_x) / widths
  dh = tf.math.log(gt_heights / heights)
  dw = tf.math.log(gt_widths / widths)

  result = tf.stack([dy, dx, dh, dw], axis=1)
  return result

def trim_zeros_graph(boxes):
  """
    Removes boxes that have all zeros.

    Args:
        boxes(tf.Tensor): [N, 4] matrix of boxes.

    Returns:
        boxes(tf.Tensor): Trimmed set of boxes
        non_zeros(tf.Tensor): Boolean matrix where True indicates the box is valid
  """
  non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
  boxes = tf.boolean_mask(boxes, non_zeros)
  return boxes, non_zeros

def overlaps_graph(boxes1, boxes2):
  """
    Computes IoU overlaps between two sets of boxes.

    Args:
        boxes1(tf.Tensor), boxes2(tf.Tensor): [N, 4] range of boxes where each box is y1, x1, y2, x2

    Returns:
        overlaps(tf.Tensor): [boxes1, boxes2] IoU overlaps
  """
  # Tile boxes2 and repeat boxes1. This prepares them for elementwise
  # operations to compute the intersection areas for all boxes pairs
  
  b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                          [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
  b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
  # Calculate intersection
  b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
  b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
  y1 = tf.maximum(b1_y1, b2_y1)
  x1 = tf.maximum(b1_x1, b2_x1)
  y2 = tf.minimum(b1_y2, b2_y2)
  x2 = tf.minimum(b1_x2, b2_x2)
  intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
  # Calculate union
  b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
  b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
  union = b1_area + b2_area - intersection
  # Compute IoU
  iou = intersection / union
  overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
  return overlaps

def log2_graph(x):
    """
    Implementation of Log2. 
    TensorFlow doesn't have a native implementation.
    
    Args:
    x (tf.Tensor): The tensor for which log base 2 needs to be computed.
    
    Returns:
    tf.Tensor: Log base 2 of x.
    """
    return tf.math.log(x) / tf.math.log(2.0)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, areas2[i], areas1)
    return overlaps

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

def make_attention_mask(image_shape, boxes):
    boxes, _ = trim_zeros_graph(boxes)
    height = image_shape[0]
    width = image_shape[1]
    # Calculate box coordinates
    y1, x1, y2, x2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    y1, y2 = y1 * height, y2 * height
    x1, x2 = x1 * width, x2 * width
    y1, x1 = tf.cast(tf.floor(y1), tf.int32), tf.cast(tf.floor(x1), tf.int32)
    y2, x2 = tf.cast(tf.math.ceil(y2), tf.int32), tf.cast(tf.math.ceil(x2), tf.int32)

    # Create coordinate matrices
    yy = tf.cast(tf.range(height)[:, tf.newaxis, tf.newaxis], tf.int32)
    xx = tf.cast(tf.range(width)[tf.newaxis, :, tf.newaxis], tf.int32)

    # Broadcast and compare against box coordinates
    mask_y = tf.logical_and(yy >= y1, yy < y2)
    mask_x = tf.logical_and(xx >= x1, xx < x2)
    mask = tf.logical_and(mask_y, mask_x)
    
    # Combine the results
    mask = tf.reduce_any(mask, axis=-1)   # Merge masks from different boxes

    return tf.cast(mask, tf.float32)

#######################################################################################################
#######################################################################################################
#######################################################################################################


def find_boundary(mask, threshold=15):
    m, n = len(mask), len(mask[0])
    visited = set()
    boundary = set()

    directions = [(1,0),(-1,0),(0,1),(0,-1)]

    def bfs(i, j):
        nonlocal m, n, mask, boundary, visited

        component_size = 0
        q = deque()
        q.append((i, j))
        in_queue = set()  # To keep track of what's already in the queue
        in_queue.add((i, j))
        component_boundary = set()

        while q:
            k, l = q.popleft()

            if (k, l) not in visited:
                visited.add((k, l))
                component_size += 1

                for a, b in directions:
                    x, y = k+a, l+b
                    if 0 <= x < m and 0 <= y < n:
                        if mask[x][y] == 0:
                            component_boundary.add((k, l))
                        elif (x, y) not in visited and (x, y) not in in_queue:
                            q.append((x, y))
                            in_queue.add((x, y))

        if component_size < threshold:
            boundary.update(component_boundary)

    non_zero_indices = {(i, j) for i in range(m) for j in range(n) if mask[i][j] != 0}

    for i, j in non_zero_indices:
        if (i, j) not in visited:
            bfs(i, j)

    return boundary

def thicken(mask, boundary, thickness = 5) :
  m = len(mask)
  n = len(mask[0])
  curr_thickness = 0
  thickened_mask = mask.copy()
  in_queue = boundary.copy()
  q = deque(boundary)

  while q and curr_thickness < thickness :
    q_length = len(q)
    for _ in range(q_length) :
      k,l = q.popleft()
      thickened_mask[k][l] = 1
      for a,b in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)] :
          if 0<= k+a<m and 0<=l+b<n :
            if thickened_mask[k+a][l+b] == 0 and (k+a,l+b) not in in_queue:
              q.append((k+a,l+b))
              in_queue.add((k+a,l+b))
    curr_thickness +=1
  return thickened_mask




def generate_component_info_list(mask):
    n_components = 0
    m, n = len(mask), len(mask[0])
    
    mask_by_components = np.full((m, n), -1)
    bbox_list = []
    component_size_list = []
    visited = set()
    
    def bfs(i, j):
        nonlocal n_components, m, n
        nonlocal mask, mask_by_components, visited
        y1, x1, y2, x2 = m- 0.5, n - 0.5, -0.5, -0.5
        component_size = 0
        q = deque([(i, j)])
        queued = set([(i, j)])
        
        while q:
            k, l = q.popleft()
            visited.add((k, l))
            component_size += 1
            mask_by_components[k][l] = n_components
            
            # Update bbox
            y1, x1, y2, x2 = min(y1, k-0.5), min(x1, l-0.5), max(y2, k+0.5), max(x2, l+0.5)
            
            for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                if 0 <= k + a < m and 0 <= l + b < n and (k + a, l + b) not in visited and mask[k + a][l + b] == 1:
                    if (k + a, l + b) not in queued:
                        q.append((k + a, l + b))
                        queued.add((k + a, l + b))

        n_components += 1
        return component_size, np.array([y1, x1, y2, x2])
    
    for i in range(m):
        for j in range(n):
            if mask[i][j] == 1 and (i, j) not in visited:
                component_size, bbox = bfs(i, j)
                component_size_list.append(component_size)
                bbox_list.append(bbox)

    return [[[i], component_size_list[i], bbox_list[i], (bbox_list[i][2]-bbox_list[i][0])*(bbox_list[i][3]-bbox_list[i][1])] for i in range(n_components)], mask_by_components

def correct_component_sizes(thickened_component_info_list, original_component_info_list) :
  used = [False for _ in range(len(original_component_info_list))]
  for thickened_component_info in thickened_component_info_list :
    thickened_component_info[1] = 0
    for j, original_component_info in enumerate(original_component_info_list) :
      if not used[j] :
        if (original_component_info[2][0]>=thickened_component_info[2][0] and 
            original_component_info[2][1]>=thickened_component_info[2][1] and 
            original_component_info[2][2]<=thickened_component_info[2][2] and 
            original_component_info[2][3]<=thickened_component_info[2][3]) :
          used[j] = True
          thickened_component_info[1] += original_component_info[1]
  return thickened_component_info_list



def shrink_bboxes(mask, bbox_info_list) :
  for i in range(len(bbox_info_list)) :
    bbox = bbox_info_list[i][2]
    y1, x1, y2, x2 = int(bbox[0]+0.5), int(bbox[1]+0.5), int(bbox[2]-0.5), int(bbox[3]-0.5)
    while sum(mask[y1:y2+1, x1]) == 0 :
      x1 +=1
    while sum(mask[y1:y2+1, x2]) == 0 :
      x2 -=1
    while sum(mask[y1, x1:x2+1]) == 0 :
      y1 +=1
    while sum(mask[y2, x1:x2+1]) == 0 :
      y2 -=1
    bbox[0], bbox[1], bbox[2], bbox[3] = y1-0.5, x1-0.5, y2+0.5, x2+0.5
    bbox_info_list[i][3] = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
  
  return bbox_info_list




def density_based_merge(bbox_info_list, threshold = 0.9) :
  n_components = len(bbox_info_list)
  deleted = [False for _ in range(n_components)]
  bbox_info_list.sort(key = lambda x : x[1])

  for i in range(n_components) :
    density_decrease = [2 for _ in range(n_components)] 
    for j in range(i+1,n_components) :
      y1 = min(bbox_info_list[i][2][0], bbox_info_list[j][2][0])
      x1 = min(bbox_info_list[i][2][1], bbox_info_list[j][2][1])
      y2 = max(bbox_info_list[i][2][2], bbox_info_list[j][2][2])
      x2 = max(bbox_info_list[i][2][3], bbox_info_list[j][2][3])

      original_density = bbox_info_list[j][1]/bbox_info_list[j][3]
      new_density = (bbox_info_list[i][1]+bbox_info_list[j][1])/((x2-x1)*(y2-y1))
      
      if new_density > threshold * original_density :
        density_decrease[j] = original_density - new_density

    min_index, min_value = min(enumerate(density_decrease), key=lambda pair: pair[1])
    
    if min_value < 2: 
      deleted[i] = True
      bbox_info_list[min_index][0] += bbox_info_list[i][0]
      bbox_info_list[min_index][1] += bbox_info_list[i][1]
      bbox_info_list[min_index][2][0] = min(bbox_info_list[i][2][0], bbox_info_list[min_index][2][0])
      bbox_info_list[min_index][2][1] = min(bbox_info_list[i][2][1], bbox_info_list[min_index][2][1])
      bbox_info_list[min_index][2][2] = max(bbox_info_list[i][2][2], bbox_info_list[min_index][2][2])
      bbox_info_list[min_index][2][3] = max(bbox_info_list[i][2][3], bbox_info_list[min_index][2][3])
      bbox_info_list[min_index][3] = ((bbox_info_list[min_index][2][2]-bbox_info_list[min_index][2][0])*
                                      (bbox_info_list[min_index][2][3]-bbox_info_list[min_index][2][1])) 
  
  return [bbox_info for bbox_info, flag in zip(bbox_info_list, deleted) if not flag]



def intersection_based_merge(bbox_info_list, threshold = 0.6) :
  n_components = len(bbox_info_list)
  deleted = [False for _ in range(n_components)]
  bbox_info_list.sort(key = lambda x : x[3])

  for i in range(n_components) :
    intersection_ratio = [0 for _ in range(n_components)] 
    for j in range(i+1,n_components) :
      y1 = max(bbox_info_list[i][2][0], bbox_info_list[j][2][0])
      x1 = max(bbox_info_list[i][2][1], bbox_info_list[j][2][1])
      y2 = min(bbox_info_list[i][2][2], bbox_info_list[j][2][2])
      x2 = min(bbox_info_list[i][2][3], bbox_info_list[j][2][3])

      if x1<x2 and y1<y2 :
        intersection_ratio[j] = ((x2-x1)*(y2-y1))/bbox_info_list[i][3]

    max_index, max_value = max(enumerate(intersection_ratio), key=lambda pair: (pair[1], -pair[0]))
    
    if max_value > threshold : 
      deleted[i] = True
      bbox_info_list[max_index][0] += bbox_info_list[i][0]
      bbox_info_list[max_index][1] += bbox_info_list[i][1]
      bbox_info_list[max_index][2][0] = min(bbox_info_list[i][2][0], bbox_info_list[max_index][2][0])
      bbox_info_list[max_index][2][1] = min(bbox_info_list[i][2][1], bbox_info_list[max_index][2][1])
      bbox_info_list[max_index][2][2] = max(bbox_info_list[i][2][2], bbox_info_list[max_index][2][2])
      bbox_info_list[max_index][2][3] = max(bbox_info_list[i][2][3], bbox_info_list[max_index][2][3])
      bbox_info_list[max_index][3] = ((bbox_info_list[max_index][2][2]-bbox_info_list[max_index][2][0])*
                                      (bbox_info_list[max_index][2][3]-bbox_info_list[max_index][2][1])) 

  return [bbox_info for bbox_info, flag in zip(bbox_info_list, deleted) if not flag]


def add_padding(bbox_info_list, padding_ratio = 0.2) :
  if padding_ratio > 0 : 
    for i, bbox_info in enumerate(bbox_info_list) :
      bbox = bbox_info[2]
      y_range = bbox[2]-bbox[0]
      x_range = bbox[3]-bbox[1]
      bbox[0] = max(-0.5, bbox[0] - (y_range * padding_ratio)/2)
      bbox[1] = max(-0.5, bbox[1] - (x_range * padding_ratio)/2)
      bbox[2] = min(255.5, bbox[2] + (y_range * padding_ratio)/2)
      bbox[3] = min(255.5, bbox[3] + (x_range * padding_ratio)/2)
      bbox_info_list[i][2] = bbox
  return bbox_info_list
  


def color_mask(bbox_info_list, mask, mask_by_components) :
  # squeezing the list by deleting the boxes that have been merged
  postmerge_idx = [bbox_info[0] for bbox_info in bbox_info_list]
  n_components = len(postmerge_idx)
  postmerge_idx_dict = {-1:-1}

  for new_idx in range(n_components) :
    for old_idx in postmerge_idx[new_idx] : 
      postmerge_idx_dict[old_idx] = new_idx

  for i in range(mask_by_components.shape[0]) :
    for j in range(mask_by_components.shape[1]) :
      if mask[i][j] == 0 :
        mask_by_components[i][j] = -1
      else : 
        mask_by_components[i][j] = postmerge_idx_dict[mask_by_components[i][j].item()] 
  
  return mask_by_components


def generate_bboxes(mask, merge_fns=['i'], thicken_thr=15, thickness=5, d_thr=0.9, i_thr=0.6, pad_ratio=0.2) :
  #
  boundary = find_boundary(mask, thicken_thr)
  #
  thickened_mask = thicken(mask, boundary, thickness)
  #
  original_component_info_list, _ = generate_component_info_list(mask)
  thickened_component_info_list, mask_by_components = generate_component_info_list(thickened_mask)

  bbox_info_list = correct_component_sizes(thickened_component_info_list, original_component_info_list)

  bbox_info_list = shrink_bboxes(mask, bbox_info_list)

  # edge case : there is no masking in the original image
  if len(bbox_info_list) == 0 :
    return [], mask, thickened_mask, mask_by_components
  #
  for s in merge_fns :
    if s == 'd' :
      bbox_info_list = density_based_merge(bbox_info_list, d_thr)
    elif s == 'i' :
      bbox_info_list = intersection_based_merge(bbox_info_list, i_thr)
  
  if merge_fns and merge_fns[-1] == 'i' :
    bbox_info_list.sort(key = lambda x : x[3])
  else : 
    bbox_info_list.sort(key = lambda x : x[1])

  #
  bbox_info_list = add_padding(bbox_info_list, pad_ratio)
  bboxes = np.stack([bbox_info[2] for bbox_info in bbox_info_list])
  mask_by_components = color_mask(bbox_info_list, mask, mask_by_components)
  
  return bboxes, mask, thickened_mask, mask_by_components

def show_bboxes(title, mask, bboxes) :
  plt.figure(figsize=(10, 10))
  if title: 
    plt.title(title)
  plt.imshow(mask)
  plt.axis(False)
  for i in range(bboxes.shape[0]) :
    y1, x1, y2, x2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]
    # Create a rectangle
    rectangle = patches.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='r', facecolor='none')
    # Add the rectangle to the plot
    plt.gca().add_patch(rectangle)
  plt.show()


#######################################################################################################
#######################################################################################################
#######################################################################################################



def load_data(record_id, image_path, mask_path):
    # make path objects
    record_id = record_id.numpy().decode('utf-8')
    image_path = image_path.numpy().decode('utf-8') + record_id + ".npy"
    mask_path = mask_path.numpy().decode('utf-8') + record_id + ".npy"
    # load the images and masks for the correspoding paths
    image = np.load(image_path)[..., 4, :].astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)
    return image, mask

def augment_data(image, mask):
    # random flip
    image_mask_concat = tf.concat([image, mask], axis = -1)
    image_mask_concat = tf.image.random_flip_left_right(image_mask_concat)
    image_mask_concat = tf.image.random_flip_up_down(image_mask_concat)
    # random crop
    cropped = tf.image.random_crop(image_mask_concat, size=[(256*3)//4,(256*3)//4, 4])
    image_mask_concat = tf.image.resize(cropped, [256, 256])

    image = image_mask_concat[..., :-1]
    mask = image_mask_concat[..., -1]
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.cast(mask > 0.5, tf.float32)
    return image, mask


def dice_score(y_true, y_pred, binarize = True, thr=0.5, epsilon=1e-6):
    # Thresholding predictions
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if binarize :
      y_pred = tf.cast(y_pred > thr, tf.float32)

    # Flattening tensors
    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    # Computing intersections and unions
    intersections = tf.reduce_sum(y_true * y_pred, axis=1)
    unions = tf.reduce_sum(y_true + y_pred, axis=1)

    return tf.reduce_mean((2. * intersections + epsilon) / (unions + epsilon))


# Binary Cross Entropy Loss
bce = tf.keras.losses.BinaryCrossentropy(from_logits = False)

# Combined Loss
def combined_loss(y_true, y_pred, alpha=0.5, epsilon=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    return alpha * bce(y_true, y_pred) + (1 - alpha) * (1 - dice_score(y_true, y_pred, binarize = False, epsilon = epsilon))

#######################################################################################################
#######################################################################################################
#######################################################################################################
def create_restore_ckpt(ckpt_dir, model, optimizer, max_to_keep = 3) :
  # Create a checkpoint object and checkpoint manager
  ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                             optimizer=optimizer,
                             model=model)
  manager = tf.train.CheckpointManager(ckpt,
                                       ckpt_dir,
                                       max_to_keep)

  # restore the last checkpoint if there is one
  ckpt.restore(manager.latest_checkpoint)

  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  return ckpt, manager



def log_summary(log_dir, ckpt, train_epoch_loss, valid_epoch_loss, valid_epoch_metric) :
  # Log training metrics for the epoch
  train_summary_writer = tf.summary.create_file_writer(log_dir)
  valid_summary_writer = tf.summary.create_file_writer(log_dir)

  with train_summary_writer.as_default():
      tf.summary.scalar('train_loss', train_epoch_loss, step=ckpt.step.numpy())

  # Log validation metrics for the epoch
  with valid_summary_writer.as_default():
      tf.summary.scalar('valid_loss', valid_epoch_loss, step=ckpt.step.numpy())
      tf.summary.scalar('valid_metric', valid_epoch_metric, step=ckpt.step.numpy())


 
@tf.function
def train_step(X, y_true, model, loss_fn, optimizer):
  with tf.GradientTape() as tape:
    y_pred = model(X, training = True)
    loss = loss_fn(y_true, y_pred)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss

@tf.function
def valid_step(X, y_true, model, loss_fn, metric_fn):
  y_pred = model(X, training = False)
  loss = loss_fn(y_true, y_pred)
  metric = metric_fn(y_true, y_pred)
  return loss, metric

def print_results(train_epoch_loss, valid_epoch_loss, valid_epoch_metric, start_time, end_time) :
  # print out results
  print("\n  (Results)")
  print(f"  Train Loss: {train_epoch_loss:.4f}")
  print(f"  Valid Loss: {valid_epoch_loss:.4f}")
  print(f"  Valid Metric: {valid_epoch_metric:.4f}")
  print("\n  (Time)")
  print("  Start Time : ", start_time)
  print("  End Time : ", end_time)
  print("------------------------------------------")



def train_model(train_dataset, valid_dataset, model, loss_fn, metric_fn, optimizer, n_epochs, ckpt_dir, log_dir, max_to_keep=3):

  # printout which device the model is on
  device_info = model.layers[0].weights[0].device
  device_name = device_info.split(":")[-2]
  print("Device : ", device_name)

  # Create a checkpoint object and checkpoint manager
  ckpt, manager = create_restore_ckpt(ckpt_dir, model, optimizer, max_to_keep)


  # training loop starts
  for epoch in range(n_epochs):
    ckpt.step.assign_add(1)
    print(f"\n Epoch {ckpt.step.numpy()} : \n")
    start_time = timer()

    # train loop
    train_n_batches = 0
    train_total_loss = 0

    for X, y_true in tqdm(train_dataset, desc='Training', position=0):
      loss_value = train_step(X, y_true, model, loss_fn, optimizer)
      train_total_loss += loss_value
      train_n_batches +=1

      if train_n_batches % 1 == 0: # print every 1 batches
        print(f"Batch {train_n_batches}, Train Loss: {loss_value:.4f}")

    train_epoch_loss = train_total_loss / train_n_batches

    # valid loop
    valid_n_batches = 0
    valid_total_loss = 0
    valid_total_metric = 0

    for X, y_true in tqdm(valid_dataset, desc='Validation', position=0):
      loss_value, metric_value = valid_step(X, y_true, model, loss_fn, metric_fn)
      valid_total_loss += loss_value
      valid_total_metric += metric_value
      valid_n_batches += 1

    valid_epoch_loss = valid_total_loss / valid_n_batches
    valid_epoch_metric = valid_total_metric / valid_n_batches


    # log loss and metric/ save checkpoint for the epoch
    log_summary(log_dir, ckpt, train_epoch_loss, valid_epoch_loss, valid_epoch_metric)
    manager.save()

    end_time = timer()

    print_results(train_epoch_loss, valid_epoch_loss, valid_epoch_metric, start_time, end_time)

#######################################################################################################
#######################################################################################################
#######################################################################################################

def show_predictions(original_images, y_true, y_pred, thr=0.5, epsilon=0.001, n_images=1):
    # Thresholding predictions
    trues = tf.cast(y_true, tf.float32)
    preds = tf.cast(y_pred, tf.float32)

    preds = tf.cast(preds > thr, tf.float32)

    # Flattening tensors
    trues = tf.reshape(trues, [tf.shape(trues)[0], -1])
    preds = tf.reshape(preds, [tf.shape(preds)[0], -1])

    # Computing intersections and unions
    intersections = tf.reduce_sum(trues * preds, axis=1)
    unions = tf.reduce_sum(trues, axis=1) + tf.reduce_sum(preds, axis=1)

    # Computing Dice scores
    dices = (2. * intersections + epsilon) / (unions + epsilon)

    best_indices = tf.argsort(dices, direction='ASCENDING')[-n_images:]
    worst_indices = tf.argsort(dices, direction='ASCENDING')[:n_images]

    print("Best Predictions :\n")
    for i,idx in enumerate(best_indices[::-1]):
      print(f"best {i} : ")
      plt.figure(figsize=(20, 10))
      original_image, true, pred =original_images[idx], y_true[idx].numpy(), y_pred[idx].numpy()

      # original image
      plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st image
      plt.imshow(original_image)  # adjust colormap if necessary
      plt.title('original_image')
      plt.axis('off')

      # y_true image
      plt.subplot(1, 3, 2)  # 1 row, 2 columns, 1st image
      plt.imshow(true)  # adjust colormap if necessary
      plt.title('ground truth')
      plt.axis('off')

       # y_pred image
      plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd image
      plt.imshow(pred)  # adjust colormap if necessary
      plt.title('prediction')
      plt.axis('off')

      plt.tight_layout()
      plt.show()

    print("worst predictions :\n")
    for i,idx in enumerate(worst_indices):
      print(f"worst {i} : ")
      plt.figure(figsize=(20, 10))
      original_image, true, pred =original_images[idx], y_true[idx].numpy(), y_pred[idx].numpy()

      # original image
      plt.subplot(1, 3, 1)  # 1 row, 2 columns, 1st image
      plt.imshow(original_image)  # adjust colormap if necessary
      plt.title('original_image')
      plt.axis('off')

      # y_true image
      plt.subplot(1, 3, 2)  # 1 row, 2 columns, 1st image
      plt.imshow(true)  # adjust colormap if necessary
      plt.title('ground truth')
      plt.axis('off')

       # y_pred image
      plt.subplot(1, 3, 3)  # 1 row, 2 columns, 2nd image
      plt.imshow(pred)  # adjust colormap if necessary
      plt.title('prediction')
      plt.axis('off')

      plt.tight_layout()
      plt.show()