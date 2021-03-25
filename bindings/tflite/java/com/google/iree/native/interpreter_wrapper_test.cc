// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bindings/tflite/java/com/google/iree/native/interpreter_wrapper.h"

#include <memory>

#include "bindings/tflite/java/com/google/iree/native/model_wrapper.h"
#include "bindings/tflite/java/com/google/iree/native/options_wrapper.h"
#include "bindings/tflite/java/com/google/iree/native/tensor_wrapper.h"
#include "iree/base/api.h"
#include "iree/testing/gtest.h"

// Test model is available both on the filesystem and here for embedding testing
// embedding the module directly in a binary.
#include "bindings/tflite/testdata/add_static.h"
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_DATA \
  iree::bindings::tflite::testdata::add_static_create()->data
#define IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_SIZE \
  iree::bindings::tflite::testdata::add_static_create()->size

#define ASSERT_OK(status) ASSERT_EQ(status, iree_ok_status())

namespace iree {
namespace tflite {
namespace {

class InterpreterWrapperTest : public testing::Test {
 protected:
  void SetUp() override {
    // Code here will be called before *each* test.
  }

  void TearDown() override {
    // Code here will be called after *each* test.
  }
};

TEST_F(InterpreterWrapperTest, AddStaticTest) {
  // std::unique_ptr<ModelWrapper> model_wrapper = absl::MakeUnique();
  ModelWrapper model_wrapper;
  ASSERT_OK(model_wrapper.Create(
      IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_DATA,
      IREE_BINDINGS_TFLITE_TESTDATA_ADD_STATIC_EMBEDDED_SIZE));
  ASSERT_NE(model_wrapper.model(), nullptr);

  OptionsWrapper options_wrapper;
  options_wrapper.SetNumThreads(2);
  ASSERT_NE(options_wrapper.options(), nullptr);

  InterpreterWrapper interpreter_wrapper;
  ASSERT_OK(interpreter_wrapper.Create(model_wrapper, options_wrapper));
  ASSERT_NE(interpreter_wrapper.interpreter(), nullptr);

  ASSERT_OK(interpreter_wrapper.AllocateTensors());
  ASSERT_EQ(interpreter_wrapper.get_input_tensor_count(), 1);
  ASSERT_EQ(interpreter_wrapper.get_output_tensor_count(), 1);

  TensorWrapper input_tensor_wrapper = *(interpreter_wrapper.GetInputTensor(0));
  ASSERT_NE(input_tensor_wrapper.tensor(), nullptr);
  EXPECT_EQ(input_tensor_wrapper.tensor_type(), kTfLiteFloat32);
  EXPECT_EQ(input_tensor_wrapper.num_dims(), 4);
  EXPECT_EQ(input_tensor_wrapper.dim(0), 1);
  EXPECT_EQ(input_tensor_wrapper.dim(1), 8);
  EXPECT_EQ(input_tensor_wrapper.dim(2), 8);
  EXPECT_EQ(input_tensor_wrapper.dim(3), 3);
  EXPECT_EQ(input_tensor_wrapper.byte_size(), sizeof(float) * 1 * 8 * 8 * 3);
  EXPECT_NE(input_tensor_wrapper.tensor_data(), nullptr);
  EXPECT_STREQ(input_tensor_wrapper.tensor_name(), "input");

  TfLiteQuantizationParams input_params =
      input_tensor_wrapper.quantization_params();
  EXPECT_EQ(input_params.scale, 0.f);
  EXPECT_EQ(input_params.zero_point, 0);

  std::array<float, 1 * 8 * 8 * 3> input = {
      1.f,
      3.f,
  };
  ASSERT_OK(input_tensor_wrapper.CopyFromBuffer(input.data(),
                                                input.size() * sizeof(float)));

  ASSERT_OK(interpreter_wrapper.Invoke());

  TensorWrapper output_tensor_wrapper =
      *(interpreter_wrapper.GetOutputTensor(0));
  ASSERT_NE(output_tensor_wrapper.tensor(), nullptr);
  EXPECT_EQ(output_tensor_wrapper.tensor_type(), kTfLiteFloat32);
  EXPECT_EQ(output_tensor_wrapper.num_dims(), 4);
  EXPECT_EQ(output_tensor_wrapper.dim(0), 1);
  EXPECT_EQ(output_tensor_wrapper.dim(1), 8);
  EXPECT_EQ(output_tensor_wrapper.dim(2), 8);
  EXPECT_EQ(output_tensor_wrapper.dim(3), 3);
  EXPECT_EQ(output_tensor_wrapper.byte_size(), sizeof(float) * 1 * 8 * 8 * 3);
  EXPECT_NE(output_tensor_wrapper.tensor_data(), nullptr);
  EXPECT_STREQ(output_tensor_wrapper.tensor_name(), "output");

  TfLiteQuantizationParams output_params =
      output_tensor_wrapper.quantization_params();
  EXPECT_EQ(output_params.scale, 0.f);
  EXPECT_EQ(output_params.zero_point, 0);

  std::array<float, 1 * 8 * 8 * 3> output;
  ASSERT_OK(output_tensor_wrapper.CopyToBuffer(output.data(),
                                               output.size() * sizeof(float)));
  EXPECT_EQ(output[0], 2.f);
  EXPECT_EQ(output[1], 6.f);
}

}  // namespace
}  // namespace tflite
}  // namespace iree
