// RUN: iree-opt %s --iree-codegen-gemmini-ir-dump -o /dev/null 2>&1 | FileCheck %s

module {
  func.func @matmul(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<4x4xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<4x4xf32>) -> tensor<4x4xf32>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%fill : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  func.func @batch_matmul(%arg0: tensor<2x4x4xf32>, %arg1: tensor<2x4x4xf32>) -> tensor<2x4x4xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<2x4x4xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    %0 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%fill : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    return %0 : tensor<2x4x4xf32>
  }

  func.func @conv(%input: tensor<1x1x3x3xf32>, %filter: tensor<1x1x1x1xf32>) -> tensor<1x1x3x3xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x1x3x3xf32>
    %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    %0 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%input, %filter : tensor<1x1x3x3xf32>, tensor<1x1x1x1xf32>) outs(%fill : tensor<1x1x3x3xf32>) -> tensor<1x1x3x3xf32>
    return %0 : tensor<1x1x3x3xf32>
  }

  func.func @gemmini_ops(%a: memref<1x1xi8>, %b: memref<1x1xi8>, %c: memref<1x1xi32>, %d: memref<1x1xi32>,
                         %input: memref<1x1x1x1xi8>, %weights: memref<1x1xi8>, %bias: memref<1xi32>, %output: memref<1x1xi32>) {
    %out_row = arith.constant 1 : i64
    %out_col = arith.constant 1 : i64
    %kdim = arith.constant 1 : i64
    gemmini.tile_matmul %a %b %c %d : memref<1x1xi8> memref<1x1xi8> memref<1x1xi32> memref<1x1xi32>
    gemmini.tile_conv %input %weights %bias %output %out_row %out_col %kdim : memref<1x1x1x1xi8> memref<1x1xi8> memref<1xi32> memref<1x1xi32> i64 i64 i64
    return
  }
}

// CHECK: GEMMINI_IR_DUMP: linalg.matmul
// CHECK: GEMMINI_IR_DUMP_FUNC: @matmul
// CHECK: GEMMINI_IR_DUMP: linalg.batch_matmul
// CHECK: GEMMINI_IR_DUMP_FUNC: @batch_matmul
// CHECK: GEMMINI_IR_DUMP: linalg.conv_2d_nchw_fchw
// CHECK: GEMMINI_IR_DUMP_FUNC: @conv
// CHECK: GEMMINI_IR_DUMP: gemmini.tile_matmul
// CHECK: GEMMINI_IR_DUMP_FUNC: @gemmini_ops
// CHECK: GEMMINI_IR_DUMP: gemmini.tile_conv
// CHECK: GEMMINI_IR_DUMP_FUNC: @gemmini_ops
