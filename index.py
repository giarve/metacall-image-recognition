#!/usr/bin/python3.6

"""MetaCall Examples - Image recognition.

[MetaCall](https://metacall.io)
[MetaCall Examples](https://github.com/metacall/examples)

This module shows the ability to run an open source machine learning library for research and production on Metacall.

""" 

import tensorflow as tf
import numpy as np

from distutils.version import StrictVersion
from PIL import Image
import io
import os

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Object detection
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
import visualization_utils as vis_util

# Ease of use
import helpers

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

# if graph doesnt exist, download it
if not os.path.exists(helpers.PATH_TO_FROZEN_GRAPH):
  helpers.download_model()

def test_check_if_output_exists():
  if os.path.exists(os.path.join('OUTPUT.jpeg')):
    return True
  else:
    return False

def index():
  """Read index.html from file and return it.

  Returns:
    Return the index.html content.
  """

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  image = Image.open(os.path.join('image_input.jpg'))
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = helpers.load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np_expanded, helpers.load_model_into_memory())
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=9)

  im = Image.fromarray(image_np)
  im.save("OUTPUT.jpeg")

  with io.BytesIO() as output:
    im.save(output, format="JPEG")
    contents = output.getvalue()
    return contents # returns as *bytes* as METACALL handles this type

  return "ERROR"