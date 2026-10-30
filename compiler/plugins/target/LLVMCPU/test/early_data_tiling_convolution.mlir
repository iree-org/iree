// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-global-opt-use-im2col-for-convs=false --iree-dispatch-creation-experimental-set-data-tiling-ops=convolution --iree-opt-experimental-early-data-tiling --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --compile-to=dispatch-creation | FileCheck %s
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-global-opt-use-im2col-for-convs=false --iree-dispatch-creation-experimental-set-data-tiling-ops=convolution --iree-opt-experimental-early-data-tiling --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --compile-to=executable-targets -o %t
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-global-opt-use-im2col-for-convs=false --iree-dispatch-creation-experimental-set-data-tiling-ops=convolution --iree-opt-experimental-early-data-tiling --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf --compile-to=executable-targets -o %t

// Exercise the existing blocked-convolution family with default CPU codegen.
// Named N=1 and batchless forms must retain their packed epilogue; dispatch
// preprocessing must preserve the canonical loop order used by CPU heuristics.

// CHECK-LABEL: util.func public @blocked_conv_relu
// CHECK: %[[CONV:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]
// CHECK-SAME: outs(%{{.*}} : tensor<1x1x14x14x8xf32>)
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: flow.return
// CHECK: %[[RELU:.*]] = linalg.generic
// CHECK-SAME: ins(%[[CONV]] : tensor<1x1x14x14x8xf32>)
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[RELU]]
// CHECK-SAME: tensor<1x1x14x14x8xf32> into tensor<1x14x14x8xf32>
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [2, 0, 1] inner_dims_pos = [2]
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
func.func @blocked_conv_relu(%lhs: tensor<1x16x16x4xf32>, %rhs: tensor<3x3x4x8xf32>) -> tensor<1x14x14x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>
  %mm = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%lhs, %rhs : tensor<1x16x16x4xf32>, tensor<3x3x4x8xf32>) outs(%init : tensor<1x14x14x8xf32>) -> tensor<1x14x14x8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%mm : tensor<1x14x14x8xf32>) outs(%empty : tensor<1x14x14x8xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x14x14x8xf32>
  return %result : tensor<1x14x14x8xf32>
}

// CHECK-LABEL: util.func public @nchw_stride2_relu
// CHECK: %[[CONV:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]
// CHECK-SAME: outs(%{{.*}} : tensor<2x2x8x9x8xf32>)
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: flow.return
// CHECK: %[[RELU:.*]] = linalg.generic
// CHECK-SAME: ins(%[[CONV]] : tensor<2x2x8x9x8xf32>)
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[RELU]] outer_dims_perm = [0, 1, 2, 3] inner_dims_pos = [1]
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
func.func @nchw_stride2_relu(%lhs: tensor<2x7x17x19xf32>, %rhs: tensor<13x7x3x3xf32>) -> tensor<2x13x8x9xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<2x13x8x9xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x13x8x9xf32>) -> tensor<2x13x8x9xf32>
  %mm = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%lhs, %rhs : tensor<2x7x17x19xf32>, tensor<13x7x3x3xf32>) outs(%init : tensor<2x13x8x9xf32>) -> tensor<2x13x8x9xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%mm : tensor<2x13x8x9xf32>) outs(%empty : tensor<2x13x8x9xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<2x13x8x9xf32>
  return %result : tensor<2x13x8x9xf32>
}

// CHECK-LABEL: util.func public @batchless_relu
// CHECK: %[[CONV:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction", "parallel", "reduction"]
// CHECK-SAME: outs(%{{.*}} : tensor<1x1x14x14x8xf32>)
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: flow.return
// CHECK: %[[RELU:.*]] = linalg.generic
// CHECK-SAME: ins(%[[CONV]] : tensor<1x1x14x14x8xf32>)
// CHECK: arith.maximumf
// CHECK-NOT: flow.return
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[RELU]] {{\[\[0, 1\], \[2\], \[3\], \[4\]\]}}
// CHECK-NOT: flow.return
// CHECK: %[[UNPACK:.*]] = linalg.unpack %[[COLLAPSED]] outer_dims_perm = [2, 0, 1] inner_dims_pos = [2]
// CHECK-NEXT: iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
func.func @batchless_relu(%lhs: tensor<16x16x4xf32>, %rhs: tensor<3x3x4x8xf32>) -> tensor<14x14x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<14x14x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<14x14x8xf32>) -> tensor<14x14x8xf32>
  %mm = linalg.generic {
    indexing_maps = [affine_map<(h, w, c, kh, kw, k) -> (h + kh, w + kw, k)>, affine_map<(h, w, c, kh, kw, k) -> (kh, kw, k, c)>, affine_map<(h, w, c, kh, kw, k) -> (h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%lhs, %rhs : tensor<16x16x4xf32>, tensor<3x3x4x8xf32>) outs(%init : tensor<14x14x8xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %prod = arith.mulf %a, %b : f32
    %sum = arith.addf %prod, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<14x14x8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(h, w, c) -> (h, w, c)>, affine_map<(h, w, c) -> (h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%mm : tensor<14x14x8xf32>) outs(%empty : tensor<14x14x8xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<14x14x8xf32>
  return %result : tensor<14x14x8xf32>
}

// CHECK-LABEL: util.func public @no_pack_dilated
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic
// CHECK-SAME: tensor<16x16x4xf32>
// CHECK-NOT: linalg.unpack
// CHECK: util.return
func.func @no_pack_dilated(%lhs: tensor<1x16x16x4xf32>, %rhs: tensor<3x3x4x8xf32>) -> tensor<1x12x12x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x12x12x8xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x12x12x8xf32>) -> tensor<1x12x12x8xf32>
  %mm = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%lhs, %rhs : tensor<1x16x16x4xf32>, tensor<3x3x4x8xf32>) outs(%init : tensor<1x12x12x8xf32>) -> tensor<1x12x12x8xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%mm : tensor<1x12x12x8xf32>) outs(%empty : tensor<1x12x12x8xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x12x12x8xf32>
  return %result : tensor<1x12x12x8xf32>
}

// CHECK-LABEL: util.func public @no_pack_depthwise
// CHECK-NOT: linalg.pack
// CHECK: linalg.generic
// CHECK-SAME: tensor<16x16x4xf32>
// CHECK-NOT: linalg.unpack
// CHECK: util.return
func.func @no_pack_depthwise(%lhs: tensor<1x16x16x4xf32>, %rhs: tensor<3x3x4xf32>) -> tensor<1x14x14x4xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x14x14x4xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x14x14x4xf32>) -> tensor<1x14x14x4xf32>
  %mm = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%lhs, %rhs : tensor<1x16x16x4xf32>, tensor<3x3x4xf32>) outs(%init : tensor<1x14x14x4xf32>) -> tensor<1x14x14x4xf32>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>, affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
    ins(%mm : tensor<1x14x14x4xf32>) outs(%empty : tensor<1x14x14x4xf32>) {
  ^bb0(%value: f32, %unused: f32):
    %max = arith.maximumf %value, %zero : f32
    linalg.yield %max : f32
  } -> tensor<1x14x14x4xf32>
  return %result : tensor<1x14x14x4xf32>
}
