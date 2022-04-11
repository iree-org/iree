/* Copyright 2022 The MLPerf Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MLPERF_DETECTION_DETECTION_POST_PROCESS_H_
#define MLPERF_DETECTION_DETECTION_POST_PROCESS_H_

#include <cstdint>
#include <vector>

#include "cpp_projects/dynamic_images/anchor_layers.pb.h"

// Options for the QuantizedPostProcess class.
struct PostProcessOptions {
  // The height of the input image.
  int input_height = 0;
  // The width of the input image.
  int input_width = 0;
  // The number of classes detected by the model, including the background
  // class.
  int num_classes = 91;
  // The offset of the most meaningful class. Set this to one if there is a
  // background class.
  int class_offset = 1;
  // The maximum number of detections to output per query.
  int max_detections = 50;
  // The minimum class score used when performing NMS.
  float nms_score_threshold = 0.4f;
  // The minimum IoU overlap used when performing NMS.
  float nms_iou_threshold = 0.6f;
  // The anchor configuration.
  iree::detection::AnchorLayers anchor_layers;
};

struct Box {
  int width;
  int height;
};

// The top left (x1, y1) and bottom right (x2, y2) coordinates of a bounding
// box.
struct BoxCoords {
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
};

// Represents a detection.
struct Detection {
  BoxCoords box_coords;
  int class_index = 0;
  float class_score = 0.0f;
};

// Runs the post-processing step for object detection models where outputs
// require additional decoding and a non-max-suppression step. If the input
// size changes, call `ResetInputSize()` to trigger recalculation of anchor
// boxes.
class DetectionPostProcess {
 public:
  DetectionPostProcess(const PostProcessOptions& options);

  // Updates anchor configuration to new `width` and `height`.
  void ResetInputSize(int width, int height);

  // Post-processes the location and score tensors. Expects a model to have
  // output tensors representing location and score values, in alternating
  // order.
  //
  // The `box_encodings` parameter should hold pointers to the tensor data at:
  //     `BoxPredictor_0/BoxEncodingPredictor/BiasAdd`,
  //     `BoxPredictor_1/BoxEncodingPredictor/BiasAdd`,
  //     `BoxPredictor_2/BoxEncodingPredictor/BiasAdd`.
  //
  // Similarly, the `class_predictions` parameter should hold pointers to:
  //     `BoxPredictor_0/ClassPredictor/BiasAdd`,
  //     `BoxPredictor_1/ClassPredictor/BiasAdd`,
  //     `BoxPredictor_2/ClassPredictor/BiasAdd`.
  std::vector<Detection> Run(const std::vector<float*>& box_encodings,
                             const std::vector<float*>& class_predictions);

 protected:
  // Recalculates values for `anchor_maps_` and `base_anchors_` based on the
  // new input `width` and `height`.
  void RecalculateAnchorSizes(int width, int height);

  PostProcessOptions options_;

  // Pre-calculated anchor variables.
  //
  // Each anchor layer consists of an anchor map of size WxH.
  std::vector<Box> anchor_maps_;
  // Each cell in an anchor map consists of N anchors of size WxH. The number
  // of anchors per cell depends on the base size and aspect ratio specified in
  // the anchor configuration.
  std::vector<Box> base_anchors_;
};

#endif  // MLPERF_DETECTION_DETECTION_POST_PROCESS_H_
