#include "cpp_projects/dynamic_images/detection_post_process.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cpp_projects/dynamic_images/anchor_layers.pb.h"

namespace {
// The default scale for anchor y-coordinate.
const float kBoxCoderYScale = 10.0;
// The default scale for anchor x-coordinate.
const float kBoxCoderXScale = 10.0;
// The default scale for anchor width.
const float kBoxCoderWidthScale = 5.0;
// The default scale for anchor height.
const float kBoxCoderHeightScale = 5.0;

inline BoxCoords DecodeBox(float box_y_center, float box_x_center,
                           float box_height, float box_width, float anchor_x,
                           float anchor_y, float anchor_width,
                           float anchor_height) {
  float y_center = box_y_center / kBoxCoderYScale * anchor_height + anchor_y;
  float x_center = box_x_center / kBoxCoderXScale * anchor_width + anchor_x;
  float h = std::exp(box_height / kBoxCoderHeightScale) * anchor_height;
  float w = std::exp(box_width / kBoxCoderWidthScale) * anchor_width;
  return {
      .x1 = x_center - w / 2,
      .y1 = y_center - h / 2,
      .x2 = x_center + w / 2,
      .y2 = y_center + h / 2,
  };
}

// Expects |bbox| to be in the format [xmin, ymin, xmax, ymax]
inline float ComputeArea(const BoxCoords& box) {
  const float width = box.x2 - box.x1;
  const float height = box.y2 - box.y1;
  if (width <= 0 || height < 0) {
    return 0;
  }
  return width * height;
}

float ComputeIou(const BoxCoords& box1, const BoxCoords& box2) {
  const float area1 = ComputeArea(box1);
  const float area2 = ComputeArea(box2);
  if (area1 <= 0 || area2 <= 0) return 0.0;
  BoxCoords intersection_box = {
      .x1 = std::max(box1.x1, box2.x1),
      .y1 = std::max(box1.y1, box2.y1),
      .x2 = std::min(box1.x2, box2.x2),
      .y2 = std::min(box1.y2, box2.y2),
  };
  const float intersection_area = ComputeArea(intersection_box);
  return intersection_area / (area1 + area2 - intersection_area);
}

}  // namespace

DetectionPostProcess::DetectionPostProcess(const PostProcessOptions& options)
    : options_(options) {
  ResetInputSize(options_.input_width, options_.input_height);
}

void DetectionPostProcess::ResetInputSize(int width, int height) {
  anchor_maps_.clear();
  base_anchors_.clear();
  options_.input_width = width;
  options_.input_height = height;
  for (const auto& layer : options_.anchor_layers.anchor_layer()) {
    assert(layer.base_size_size() == layer.aspect_ratio_size());

    Box anchor_map;
    anchor_map.width =
        (width + layer.width_stride() - 1) / layer.width_stride();
    anchor_map.height =
        (height + layer.height_stride() - 1) / layer.height_stride();
    anchor_maps_.push_back(anchor_map);

    for (int i = 0; i < layer.base_size_size(); ++i) {
      Box base_anchor;
      base_anchor.width = layer.base_size(i) * std::sqrt(layer.aspect_ratio(i));
      base_anchor.height =
          layer.base_size(i) / std::sqrt(layer.aspect_ratio(i));
      base_anchors_.push_back(base_anchor);
    }
  }
}

std::vector<Detection> DetectionPostProcess::Run(
    const std::vector<float*>& box_encodings,
    const std::vector<float*>& class_predictions) {
  int effective_num_classes = options_.num_classes - options_.class_offset;
  std::vector<Detection> detections;

  // At each box location, take the highest confidence detection.
  int base_anchor_index = 0;
  for (int layer_index = 0;
       layer_index < options_.anchor_layers.anchor_layer_size();
       ++layer_index) {
    const auto& layer = options_.anchor_layers.anchor_layer(layer_index);
    const float* box_ptr = box_encodings[layer_index];
    const float* scores_ptr = class_predictions[layer_index];

    for (int anchor_id_y = 0; anchor_id_y < anchor_maps_[layer_index].height;
         ++anchor_id_y) {
      float anchor_y =
          layer.height_offset() + layer.height_stride() * anchor_id_y;
      for (int anchor_idx_x = 0; anchor_idx_x < anchor_maps_[layer_index].width;
           ++anchor_idx_x) {
        float anchor_x =
            layer.width_offset() + layer.width_stride() * anchor_idx_x;
        int current_anchor_index = base_anchor_index;
        for (int j = 0; j < layer.base_size_size(); ++j) {
          scores_ptr += options_.class_offset;
          float max_score = 0;
          int max_class_index = -1;
          for (int k = 0; k < effective_num_classes; ++k) {
            if (*scores_ptr > max_score) {
              max_score = *scores_ptr;
              max_class_index = k;
            }
            scores_ptr++;
          }
          if (max_score >= options_.nms_score_threshold &&
              max_class_index != -1) {
            Detection detection;
            // Decode box.
            detection.box_coords = DecodeBox(
                box_ptr[0], box_ptr[1], box_ptr[2], box_ptr[3], anchor_y,
                anchor_x, base_anchors_[current_anchor_index].height,
                base_anchors_[current_anchor_index].width);
            // Decode score.
            detection.class_score = max_score;
            detection.class_index = max_class_index;
            detections.push_back(detection);
          }
          current_anchor_index++;
          box_ptr += 4;
        }
      }
    }
    base_anchor_index += layer.base_size_size();
  }

  // Perform nms on detections.
  std::vector<Detection> nms_detections;
  std::vector<int> indices(detections.size());
  for (int i = 0; i < detections.size(); ++i) {
    indices[i] = i;
  }
  std::sort(indices.begin(), indices.end(),
            [&detections](const int i, const int j) {
              return detections[i].class_score > detections[j].class_score;
            });
  for (int i : indices) {
    bool keep_index = true;
    for (const Detection& nms_detection : nms_detections) {
      if (ComputeIou(detections[i].box_coords, nms_detection.box_coords) >=
          options_.nms_iou_threshold) {
        keep_index = false;
        break;
      }
    }
    if (keep_index) {
      nms_detections.push_back(detections[i]);
    }
    if (nms_detections.size() >= options_.max_detections) {
      break;
    }
  }
  return nms_detections;
}
