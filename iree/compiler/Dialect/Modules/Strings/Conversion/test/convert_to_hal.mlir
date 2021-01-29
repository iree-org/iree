// RUN: iree-opt --iree-convert-to-hal %s --split-input-file | IreeFileCheck %s

// CHECK: func @to_string_tensor.f32(%arg0: !hal.buffer)
func @to_string_tensor.f32(%arg0: tensor<5xf32>) -> !strings.string_tensor {
  // CHECK: [[VIEW:%.+]] = hal.buffer_view.create %arg0
  // CHECK: [[V0:%.+]] = "strings.to_string_tensor"([[VIEW]])
  %0 = "strings.to_string_tensor"(%arg0) : (tensor<5xf32>) -> !strings.string_tensor

  // CHECK: return [[V0]]
  return %0 : !strings.string_tensor
}
