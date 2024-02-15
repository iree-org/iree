// RUN: iree-opt --split-input-file %s | FileCheck %s

util.func public @optional_convert_scalar(%arg0 : i32) -> i32 {
  // CHECK: util.numeric.optional_narrow %arg0 : i32 as si8
  %0 = util.numeric.optional_narrow %arg0 : i32 as si8
  util.return %0 : i32
}

util.func public @optional_convert_tensor(%arg0 : tensor<f32>) -> tensor<f32> {
  // CHECK: util.numeric.optional_narrow %arg0 : tensor<f32> as si8
  %0 = util.numeric.optional_narrow %arg0 : tensor<f32> as si8
  util.return %0 : tensor<f32>
}

util.func public @optional_convert_zero(%arg0 : i32) -> i32 {
  // CHECK: util.numeric.optional_narrow %arg0 : i32 as ui0
  %0 = util.numeric.optional_narrow %arg0 : i32 as ui0
  util.return %0 : i32
}
