// NOTE: file is forked from tensorflow/lite/c/c_api_test.cc with tests that
// we don't support in the shim removed. It is as unmodified as possible such
// that if a user of the tflite API does what this test does they should be able
// to use our shim too.

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <stdarg.h>
#include <stdint.h>

#include <array>
#include <fstream>
#include <vector>

#include "iree/testing/gtest.h"

// NOTE: we pull in our own copy here in case the tflite API changes upstream.
#define TFL_COMPILE_LIBRARY 1
#include "bindings/tflite/include/tensorflow/lite/c/c_api.h"

// Test model is available both on the filesystem and here for embedding testing
// embedding the module directly in a binary.
#include "bindings/tflite/testdata/add_dynamic.h"
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_DYNAMIC_EMBEDDED_DATA \
  iree::bindings::tflite::testdata::add_dynamic_create()->data
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_DYNAMIC_EMBEDDED_SIZE \
  iree::bindings::tflite::testdata::add_dynamic_create()->size
#include "bindings/tflite/testdata/add_static.h"
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_DATA \
  iree::bindings::tflite::testdata::add_static_create()->data
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_SIZE \
  iree::bindings::tflite::testdata::add_static_create()->size

// TODO(#3971): currently can't nicely load these due to cmake issues.
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_PATH \
  "tensorflow/lite/testdata/add.bin"

namespace {

TEST(CAPI, Version) { EXPECT_STRNE("", TfLiteVersion()); }

// The original test has been modified here because it uses dynamic shapes
// even though the model has static shapes defined 🤦. IREE does not support
// this misbehavior of compiling with static shapes and treating them as dynamic
// at runtime.
//
// TODO(#3975): need to remove SIP stuff and pass buffer views.
TEST(CApiSimple, DISABLED_DynamicSmoke) {
  // TODO(#3971): currently can't nicely load these due to cmake issues.
  // TfLiteModel* model =
  // TfLiteModelCreateFromFile(IREE_BINDINGS_TFLITE_TESTDATA_ADD_DYNAMIC_PATH);
  TfLiteModel* model = TfLiteModelCreate(
      IREE_BINDINGS_TFLITE_TESTDATA_ADD_DYNAMIC_EMBEDDED_DATA,
      IREE_BINDINGS_TFLITE_TESTDATA_ADD_DYNAMIC_EMBEDDED_SIZE);
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  std::array<float, 2> input = {1.f, 3.f};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(output_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 0), 2);
  EXPECT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 2);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3.f);
  EXPECT_EQ(output[1], 9.f);

  TfLiteInterpreterDelete(interpreter);
}

// As with above but without the static->dynamic override junk.
TEST(CApiSimple, StaticSmoke) {
  // TODO(#3971): currently can't nicely load these due to cmake issues.
  // TfLiteModel* model =
  // TfLiteModelCreateFromFile(IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_PATH);
  TfLiteModel* model =
      TfLiteModelCreate(IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_DATA,
                        IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_SIZE);
  ASSERT_NE(model, nullptr);

  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
  ASSERT_NE(options, nullptr);
  TfLiteInterpreterOptionsSetNumThreads(options, 2);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
  ASSERT_NE(interpreter, nullptr);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterGetInputTensorCount(interpreter), 1);
  ASSERT_EQ(TfLiteInterpreterGetOutputTensorCount(interpreter), 1);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 4);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 1), 8);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 2), 8);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 3), 3);
  EXPECT_EQ(TfLiteTensorByteSize(input_tensor), sizeof(float) * 1 * 8 * 8 * 3);
  EXPECT_NE(TfLiteTensorData(input_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(input_tensor), "input");

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  std::array<float, 1 * 8 * 8 * 3> input = {
      1.f,
      3.f,
  };
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(float)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(output_tensor), kTfLiteFloat32);
  EXPECT_EQ(TfLiteTensorNumDims(output_tensor), 4);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 0), 1);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 1), 8);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 2), 8);
  EXPECT_EQ(TfLiteTensorDim(output_tensor, 3), 3);
  EXPECT_EQ(TfLiteTensorByteSize(output_tensor), sizeof(float) * 1 * 8 * 8 * 3);
  EXPECT_NE(TfLiteTensorData(output_tensor), nullptr);
  EXPECT_STREQ(TfLiteTensorName(output_tensor), "output");

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 1 * 8 * 8 * 3> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(float)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 2.f);
  EXPECT_EQ(output[1], 6.f);

  TfLiteInterpreterDelete(interpreter);
}

// TODO(#3971): fix cmake data deps.
// TODO(#3972): plumb through quantization params.
TEST(CApiSimple, DISABLED_QuantizationParams) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile("tensorflow/lite/testdata/add_quantized.bin");
  ASSERT_NE(model, nullptr);

  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, nullptr);
  ASSERT_NE(interpreter, nullptr);

  TfLiteModelDelete(model);

  const std::array<int, 1> input_dims = {2};
  ASSERT_EQ(TfLiteInterpreterResizeInputTensor(
                interpreter, 0, input_dims.data(), input_dims.size()),
            kTfLiteOk);
  ASSERT_EQ(TfLiteInterpreterAllocateTensors(interpreter), kTfLiteOk);

  TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
  ASSERT_NE(input_tensor, nullptr);
  EXPECT_EQ(TfLiteTensorType(input_tensor), kTfLiteUInt8);
  EXPECT_EQ(TfLiteTensorNumDims(input_tensor), 1);
  EXPECT_EQ(TfLiteTensorDim(input_tensor, 0), 2);

  TfLiteQuantizationParams input_params =
      TfLiteTensorQuantizationParams(input_tensor);
  EXPECT_EQ(input_params.scale, 0.003922f);
  EXPECT_EQ(input_params.zero_point, 0);

  const std::array<uint8_t, 2> input = {1, 3};
  ASSERT_EQ(TfLiteTensorCopyFromBuffer(input_tensor, input.data(),
                                       input.size() * sizeof(uint8_t)),
            kTfLiteOk);

  ASSERT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteOk);

  const TfLiteTensor* output_tensor =
      TfLiteInterpreterGetOutputTensor(interpreter, 0);
  ASSERT_NE(output_tensor, nullptr);

  TfLiteQuantizationParams output_params =
      TfLiteTensorQuantizationParams(output_tensor);
  EXPECT_EQ(output_params.scale, 0.003922f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<uint8_t, 2> output;
  ASSERT_EQ(TfLiteTensorCopyToBuffer(output_tensor, output.data(),
                                     output.size() * sizeof(uint8_t)),
            kTfLiteOk);
  EXPECT_EQ(output[0], 3);
  EXPECT_EQ(output[1], 9);

  const float dequantizedOutput0 =
      output_params.scale * (output[0] - output_params.zero_point);
  const float dequantizedOutput1 =
      output_params.scale * (output[1] - output_params.zero_point);
  EXPECT_EQ(dequantizedOutput0, 0.011766f);
  EXPECT_EQ(dequantizedOutput1, 0.035298f);

  TfLiteInterpreterDelete(interpreter);
}

// TODO(#3972): extract !quant.uniform and plumb through iree.reflection.
#if 0

// TODO(#3971): fix cmake data deps.
TEST(CApiSimple, DISABLED_ErrorReporter) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile(IREE_BINDINGS_TFLITE_TESTDATA_ADD_PATH);
  TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();

  // Install a custom error reporter into the interpreter by way of options.
  tflite::TestErrorReporter reporter;
  TfLiteInterpreterOptionsSetErrorReporter(
      options,
      [](void* user_data, const char* format, va_list args) {
        reinterpret_cast<tflite::TestErrorReporter*>(user_data)->Report(format,
                                                                        args);
      },
      &reporter);
  TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

  // The options/model can be deleted immediately after interpreter creation.
  TfLiteInterpreterOptionsDelete(options);
  TfLiteModelDelete(model);

  // Invoke the interpreter before tensor allocation.
  EXPECT_EQ(TfLiteInterpreterInvoke(interpreter), kTfLiteError);

  // The error should propagate to the custom error reporter.
  EXPECT_EQ(reporter.error_messages(),
            "Invoke called on model that is not ready.");
  EXPECT_EQ(reporter.num_calls(), 1);

  TfLiteInterpreterDelete(interpreter);
}

#endif  // 0

TEST(CApiSimple, ValidModel) {
  TfLiteModel* model =
      TfLiteModelCreate(IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_DATA,
                        IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_SIZE);
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

// TODO(#3971): fix cmake data deps.
TEST(CApiSimple, DISABLED_ValidModelFromFile) {
  TfLiteModel* model =
      TfLiteModelCreateFromFile(IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_PATH);
  ASSERT_NE(model, nullptr);
  TfLiteModelDelete(model);
}

TEST(CApiSimple, InvalidModel) {
  std::vector<char> invalid_model(20, 'c');
  TfLiteModel* model =
      TfLiteModelCreate(invalid_model.data(), invalid_model.size());
  ASSERT_EQ(model, nullptr);
}

// TODO(#3971): fix cmake data deps.
TEST(CApiSimple, DISABLED_InvalidModelFromFile) {
  TfLiteModel* model = TfLiteModelCreateFromFile("invalid/path/foo.vmfb");
  ASSERT_EQ(model, nullptr);
}

}  // namespace
