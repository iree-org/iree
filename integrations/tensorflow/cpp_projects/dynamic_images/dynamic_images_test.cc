#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "cpp_projects/dynamic_images/detection_post_process.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/evaluation/stages/image_preprocessing_stage.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"
#include "tools/cpp/runfiles/runfiles.h"

// Much of this code is inspired from
// https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/examples/task/vision/desktop/object_detector_demo.cc

namespace {
#ifdef __ANDROID__
constexpr absl::string_view kTestDataDir("/data/local/tmp/test_data");
#else
constexpr absl::string_view kTestDataDir(
    "cpp_projects/dynamic_images/test_data");
#endif

#ifdef __ANDROID__
constexpr absl::string_view kTestOutputDir("/data/local/tmp/");
#else
constexpr absl::string_view kTestOutputDir("/tmp/");
#endif

// The line thickness (in pixels) for drawing the detection results.
constexpr int kLineThickness = 3;

// The number of colors used for drawing the detection results.
constexpr int kColorMapSize = 10;

// The names of the colors used for drawing the detection results.
constexpr std::array<absl::string_view, 10> kColorMapNames = {
    "red",      "green",      "blue",      "yellow", "fuschia",
    "dark red", "dark green", "dark blue", "gray",   "black"};

// The colors used for drawing the detection results as a flattened array of
// {R,G,B} components.
constexpr uint8_t kColorMapComponents[30] = {
    255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0,   255, 0, 255,
    128, 0, 0, 0, 128, 0, 0, 0, 128, 128, 128, 128, 0,   0, 0};

bool ReadAnchorProtoFile(absl::string_view anchor_path,
                         std::vector<char>* proto_bytes) {
  std::ifstream infile;
  infile.open(anchor_path.data(), std::ios::binary | std::ios::ate);
  if (!infile.is_open()) {
    return false;
  }
  proto_bytes->resize(infile.tellg());
  if (proto_bytes->empty()) {
    return false;
  }
  infile.seekg(0);
  if (!infile.read(&(*proto_bytes)[0], proto_bytes->size())) {
    infile.close();
    return false;
  }
  infile.close();
  return true;
}

absl::Status EncodeResultToPngFile(const std::vector<Detection>& detections,
                                   const tflite::task::vision::ImageData& image,
                                   absl::string_view save_path) {
  for (int i = 0; i < detections.size(); ++i) {
    // Get bounding box as left, top, right, bottom.
    const int left = detections[i].box_coords.x1;
    const int top = detections[i].box_coords.y1;
    const int right = detections[i].box_coords.x2;
    const int bottom = detections[i].box_coords.y2;
    // Get color components.
    const uint8_t r = kColorMapComponents[3 * (i % kColorMapSize)];
    const uint8_t g = kColorMapComponents[3 * (i % kColorMapSize) + 1];
    const uint8_t b = kColorMapComponents[3 * (i % kColorMapSize) + 2];
    // Draw. Boxes might have coordinates outside of [0, w( x [0, h( so clamping
    // is applied.
    for (int y = std::max(0, top); y < std::min(image.height, bottom); ++y) {
      for (int x = std::max(0, left); x < std::min(image.width, right); ++x) {
        int pixel_index = image.channels * (image.width * y + x);
        if (x < left + kLineThickness || x > right - kLineThickness ||
            y < top + kLineThickness || y > bottom - kLineThickness) {
          image.pixel_data[pixel_index] = r;
          image.pixel_data[pixel_index + 1] = g;
          image.pixel_data[pixel_index + 2] = b;
        }
      }
    }
  }
  return tflite::task::vision::EncodeImageToPngFile(image,
                                                    std::string(save_path));
}

void SaveImage(int width, int height, int channels, float* input_data,
               const std::vector<Detection>& detections,
               absl::string_view save_path) {
  tflite::task::vision::ImageData image_data;
  image_data.width = width;
  image_data.height = height;
  image_data.channels = channels;
  int buffer_size = width * height * channels;
  // Convert image data from float to uint8.
  std::vector<uint8_t> image_buffer(buffer_size);
  for (int i = 0; i < buffer_size; ++i) {
    image_buffer[i] = *input_data * 127.5f + 127.5f;
    input_data++;
  }
  image_data.pixel_data = image_buffer.data();
  auto status = EncodeResultToPngFile(detections, image_data, save_path);
  EXPECT_TRUE(status.ok());
}

void TfliteResizeAndRun(int width, int height, int channels, float* input_data,
                        tflite::Interpreter* interpreter) {
  auto resize_start = std::chrono::high_resolution_clock::now();
  ASSERT_EQ(interpreter->ResizeInputTensor(interpreter->inputs()[0],
                                           {1, height, width, channels}),
            kTfLiteOk);
  ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);
  auto resize_end = std::chrono::high_resolution_clock::now();

  auto input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  input_tensor->data.raw = reinterpret_cast<char*>(input_data);
  ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);
  auto invoke_end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> resize_latency = resize_end - resize_start;
  std::chrono::duration<double> invoke_latency = invoke_end - resize_end;

  LOG(INFO) << "Resize latency: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   resize_latency)
                   .count()
            << " ms.";
  LOG(INFO) << "Invoke latency "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   invoke_latency)
                   .count()
            << " ms.";
}

class DynamicImagesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup post-processor.
    PostProcessOptions post_process_options;

    // Read anchor file.
    std::vector<char> proto_bytes;
    ASSERT_TRUE(ReadAnchorProtoFile(
        absl::StrCat(kTestDataDir, "/anchors_dynamic.pb"), &proto_bytes));
    ASSERT_TRUE(post_process_options.anchor_layers.ParseFromArray(
        proto_bytes.data(), proto_bytes.size()));

    post_processs_ =
        std::make_unique<DetectionPostProcess>(post_process_options);
  }

  // Runs the pre-processing stage of loading and resizing image input and
  // returns a pointer to the processed input image.
  float* RunPreprocessStage(int width, int height) {
    post_processs_->ResetInputSize(width, height);

    tflite::evaluation::ImagePreprocessingConfigBuilder preprocess_builder(
        "image_preprocessing", kTfLiteFloat32);
    preprocess_builder.AddResizingStep(width, height, true);
    preprocess_builder.AddDefaultNormalizationStep();

    pre_process_ =
        std::make_unique<tflite::evaluation::ImagePreprocessingStage>(
            preprocess_builder.build());
    pre_process_->Init();

    std::string image_path =
        absl::StrCat(kTestDataDir, "/images/coco_bear.jpg");
    pre_process_->SetImagePath(&image_path);
    pre_process_->Run();
    return reinterpret_cast<float*>(pre_process_->GetPreprocessedImageData());
  }

  void TfliteTest(int width, int height, absl::string_view result_filename,
                  tflite::Interpreter* interpreter) {
    int input_channels = 3;

    auto pre_process_start = std::chrono::high_resolution_clock::now();
    float* input_data = RunPreprocessStage(width, height);
    auto pre_process_end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Pre-process latency "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     pre_process_end - pre_process_start)
                     .count()
              << " ms.";

    TfliteResizeAndRun(width, height, input_channels, input_data, interpreter);

    auto post_process_start = std::chrono::high_resolution_clock::now();

    // Get the output.
    std::vector<float*> box_encodings;
    std::vector<float*> class_predictions;
    for (int i = 0; i < interpreter->outputs().size(); i += 2) {
      box_encodings.push_back(interpreter->typed_output_tensor<float>(i));
      class_predictions.push_back(
          interpreter->typed_output_tensor<float>(i + 1));
    }

    std::vector<Detection> detections =
        post_processs_->Run(box_encodings, class_predictions);
    auto post_process_end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Post-process latency "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     post_process_end - post_process_start)
                     .count()
              << " ms.";

    SaveImage(width, height, input_channels, input_data, detections,
              absl::StrCat(kTestOutputDir, result_filename));

    ASSERT_EQ(detections.size(), 1);
    EXPECT_EQ(detections[0].class_index, 22);
    EXPECT_GT(detections[0].class_score, 0.5f);
    EXPECT_LT(detections[0].box_coords.x1, 0.2f * width);
    EXPECT_LT(detections[0].box_coords.y1, 0.2f * height);
    EXPECT_GT(detections[0].box_coords.x2, 0.7f * width);
    EXPECT_GT(detections[0].box_coords.y2, 0.7f * height);
  }

  std::unique_ptr<tflite::evaluation::ImagePreprocessingStage> pre_process_;
  std::unique_ptr<DetectionPostProcess> post_processs_;
};

TEST_F(DynamicImagesTest, TestTfliteWithDynamicInput) {
  // Build the model.
  std::string model_path =
      absl::StrCat(kTestDataDir, "/ssd_mobilenet_v2_coco_dynamic.tflite");
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  // Build the interpreter.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  EXPECT_EQ(tflite::InterpreterBuilder(*model, resolver)(&interpreter),
            kTfLiteOk);

  // Run test for 200x200 input.
  TfliteTest(200, 200, "result_tflite_200x200.png", interpreter.get());

  // Run test for 240x240 input.
  TfliteTest(240, 240, "result_tflite_240x240.png", interpreter.get());

  // Run test for 280x280 input.
  TfliteTest(280, 280, "result_tflite_280x280.png", interpreter.get());

  // Run test for 320x320 input.
  TfliteTest(320, 320, "result_tflite_320x320.png", interpreter.get());

  // Run test for 360x360 input.
  TfliteTest(360, 360, "result_tflite_360x360.png", interpreter.get());
}

}  // namespace
