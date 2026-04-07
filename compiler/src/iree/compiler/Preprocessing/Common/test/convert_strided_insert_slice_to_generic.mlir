// RUN: iree-opt --split-input-file --mlir-print-local-scope --iree-preprocessing-convert-strided-insert-slice-to-generic %s | FileCheck %s

// Converted: stride-2 with non-zero offsets, no passthrough dims, no collapse.
// Checks the index arithmetic: sub offset, rem/div for stride check, bounds check, clamp, extract, select.
util.func public @stride2_no_passthrough(%src: tensor<4x4xf16>) -> tensor<9x9xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<9x9xf16>
  %0 = tensor.insert_slice %src into %cst[1, 1] [4, 4] [2, 2] : tensor<4x4xf16> into tensor<9x9xf16>
  util.return %0 : tensor<9x9xf16>
}

// CHECK-LABEL: @stride2_no_passthrough
// CHECK-SAME:      %[[SRC:.*]]: tensor<4x4xf16>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      outs({{.*}} : tensor<9x9xf16>)
// CHECK:         linalg.index 0
// CHECK:         arith.subi
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         linalg.index 1
// CHECK:         arith.subi
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         tensor.extract %[[SRC]]
// CHECK:         arith.select
// CHECK:         linalg.yield
// CHECK:       util.return %[[GENERIC]]

// -----

// Converted: stride-3 with non-zero offsets.
util.func public @stride3_no_passthrough(%src: tensor<3x3xf32>) -> tensor<10x10xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<10x10xf32>
  %0 = tensor.insert_slice %src into %cst[1, 1] [3, 3] [3, 3] : tensor<3x3xf32> into tensor<10x10xf32>
  util.return %0 : tensor<10x10xf32>
}

// CHECK-LABEL: @stride3_no_passthrough
// CHECK-SAME:      %[[SRC:.*]]: tensor<3x3xf32>
// CHECK-NOT:   tensor.insert_slice
// CHECK:       %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:      outs({{.*}} : tensor<10x10xf32>)
// CHECK:         linalg.index 0
// CHECK:         arith.subi
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         linalg.index 1
// CHECK:         arith.subi
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// CHECK:         tensor.extract %[[SRC]]
// CHECK:         arith.select
// CHECK:       util.return %[[GENERIC]]

// -----

// Converted with dim collapse: 5D input with passthrough trailing dims [3,4]
// collapsed to 4D. Checks collapse_shape, 4D generic, and expand_shape.
util.func public @stride2_with_collapse(%src: tensor<1x25x25x4x8xf16>) -> tensor<1x52x52x4x8xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<1x52x52x4x8xf16>
  %0 = tensor.insert_slice %src into %cst[0, 1, 1, 0, 0] [1, 25, 25, 4, 8] [1, 2, 2, 1, 1] : tensor<1x25x25x4x8xf16> into tensor<1x52x52x4x8xf16>
  util.return %0 : tensor<1x52x52x4x8xf16>
}

// CHECK-LABEL: @stride2_with_collapse
// CHECK-SAME:      %[[SRC:.*]]: tensor<1x25x25x4x8xf16>
// CHECK-NOT:   tensor.insert_slice
// Source collapsed: dims [3,4] merged (4*8=32).
// CHECK:       %[[CSRC:.*]] = tensor.collapse_shape %[[SRC]]
// CHECK-SAME:      tensor<1x25x25x4x8xf16> into tensor<1x25x25x32xf16>
// Generic at reduced rank 4.
// CHECK:       %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:      outs({{.*}} : tensor<1x52x52x32xf16>)
// Dim 0 (batch): passthrough.
// CHECK:         linalg.index 0
// Dim 1 (H): strided.
// CHECK:         linalg.index 1
// CHECK:         arith.subi
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// Dim 2 (W): strided.
// CHECK:         linalg.index 2
// CHECK:         arith.subi
// CHECK:         arith.remsi
// CHECK:         arith.divsi
// Dim 3 (collapsed G*C): passthrough.
// CHECK:         linalg.index 3
// CHECK:         tensor.extract %[[CSRC]]
// CHECK:         arith.select
// CHECK:         linalg.yield
// Result expanded back to 5D.
// CHECK:       %[[EXP:.*]] = tensor.expand_shape %[[GENERIC]]
// CHECK-SAME:      tensor<1x52x52x32xf16> into tensor<1x52x52x4x8xf16>
// CHECK:       util.return %[[EXP]]

// -----

// No transformation: all strides are 1.
util.func public @no_transform_unit_strides(%src: tensor<32x25x25x2048xf16>) -> tensor<32x50x50x2048xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x50x50x2048xf16>
  %0 = tensor.insert_slice %src into %cst[0, 1, 1, 0] [32, 25, 25, 2048] [1, 1, 1, 1] : tensor<32x25x25x2048xf16> into tensor<32x50x50x2048xf16>
  util.return %0 : tensor<32x50x50x2048xf16>
}

// CHECK-LABEL: @no_transform_unit_strides
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic

// -----

// No transformation: destination is not a zero constant.
util.func public @no_transform_nonzero_dest(%src: tensor<4x4xf32>, %dest: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = tensor.insert_slice %src into %dest[0, 0] [4, 4] [2, 2] : tensor<4x4xf32> into tensor<8x8xf32>
  util.return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: @no_transform_nonzero_dest
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic

// -----

// No transformation: passthrough product (32*2048=65536) exceeds threshold.
util.func public @no_transform_large_passthrough(%src: tensor<32x25x25x2048xf16>) -> tensor<32x50x50x2048xf16> {
  %cst = arith.constant dense<0.000000e+00> : tensor<32x50x50x2048xf16>
  %0 = tensor.insert_slice %src into %cst[0, 0, 0, 0] [32, 25, 25, 2048] [1, 2, 2, 1] : tensor<32x25x25x2048xf16> into tensor<32x50x50x2048xf16>
  util.return %0 : tensor<32x50x50x2048xf16>
}

// CHECK-LABEL: @no_transform_large_passthrough
// CHECK:       tensor.insert_slice
// CHECK-NOT:   linalg.generic
