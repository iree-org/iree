// RUN: iree-opt --pass-pipeline='builtin.module(iree-linalg-pad-contraction-to-block-size{rowAlignment=16 columnAlignment=32})' --split-input-file %s | FileCheck %s

// CHECK-LABEL: @pad_matmul_static
// Full verification is done on this case. Others use reduced checks.
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = tensor.pad %arg0 low[0, 0] high[6, 12]  {
// CHECK:           ^bb0(%[[VAL_5:.*]]: index, %[[VAL_6:.*]]: index):
// CHECK:             tensor.yield %[[VAL_3]] : f32
// CHECK:           } : tensor<250x500xf32> to tensor<256x512xf32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = tensor.pad %arg1 low[0, 0] high[12, 4]  {
// CHECK:           ^bb0(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
// CHECK:             tensor.yield %[[VAL_7]] : f32
// CHECK:           } : tensor<500x1020xf32> to tensor<512x1024xf32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_12:.*]] = tensor.pad %arg2 low[0, 0] high[6, 4]  {
// CHECK:           ^bb0(%[[VAL_13:.*]]: index, %[[VAL_14:.*]]: index):
// CHECK:             tensor.yield %[[VAL_11]] : f32
// CHECK:           } : tensor<250x1020xf32> to tensor<256x1024xf32>
// CHECK:           %[[VAL_15:.*]] = linalg.matmul ins(%[[VAL_16:.*]], %[[VAL_17:.*]] : tensor<256x512xf32>, tensor<512x1024xf32>) outs(%[[VAL_18:.*]] : tensor<256x1024xf32>) -> tensor<256x1024xf32>
// CHECK:           %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_15]][0, 0] [250, 1020] [1, 1] : tensor<256x1024xf32> to tensor<250x1020xf32>
// CHECK:           return %[[VAL_19]] : tensor<250x1020xf32>
func.func @pad_matmul_static(%arg0 : tensor<250x500xf32>, %arg1 : tensor<500x1020xf32>,
        %arg2 : tensor<250x1020xf32>) -> tensor<250x1020xf32> {
  %matmul = linalg.matmul
      ins(%arg0, %arg1 : tensor<250x500xf32>, tensor<500x1020xf32>)
      outs(%arg2 : tensor<250x1020xf32>) -> tensor<250x1020xf32>
  return %matmul : tensor<250x1020xf32>
}

// -----
// CHECK-LABEL: @pad_matmul_noop
// CHECK-NOT: pad_tensor
// CHECK-NOT: extract_slice
func.func @pad_matmul_noop(%arg0 : tensor<256x512xf32>, %arg1 : tensor<512x1024xf32>,
        %arg2 : tensor<256x1024xf32>) -> tensor<256x1024xf32> {
  %matmul = linalg.matmul
      ins(%arg0, %arg1 : tensor<256x512xf32>, tensor<512x1024xf32>)
      outs(%arg2 : tensor<256x1024xf32>) -> tensor<256x1024xf32>
  return %matmul : tensor<256x1024xf32>
}

// -----
// CHECK-LABEL: @pad_matmul_dynamic_row
// Should trigger row alignment (16).
// Pad LHS:
// CHECK:           %[[LHS_DIM0:.*]] = arith.constant 0 : index
// CHECK:           %[[LHS_DIM:.*]] = tensor.dim %arg0, %[[LHS_DIM0]] : tensor<?x512xf32>
// CHECK:           %[[LHS_ALIGN:.*]] = arith.constant 16 : index
// CHECK:           %[[LHS_DIM_ALIGNED:.*]] = iree_input.align %[[LHS_DIM]], %[[LHS_ALIGN]] : index
// CHECK:           %[[LHS_ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[LHS_PADDED:.*]] = tensor.pad %arg0 low[0, 0] high{{\[}}%[[LHS_DIM_ALIGNED]], 0]   {
// CHECK:           } : tensor<?x512xf32> to tensor<?x512xf32>
// Pad Output:
// CHECK:           %[[OUTPUT_PADDED:.*]] = tensor.pad %arg2 low[0, 0] high{{\[}}{{.*}}, 0]  {
// CHECK:           } : tensor<?x1024xf32> to tensor<?x1024xf32>
// Matmul:
// CHECK:           %[[PADDED_RESULT:.*]] = linalg.matmul ins(%[[LHS_PADDED]], %arg1 : tensor<?x512xf32>, tensor<512x1024xf32>) outs(%[[OUTPUT_PADDED]] : tensor<?x1024xf32>) -> tensor<?x1024xf32>
// CHECK:           %[[DIM0:.*]] = arith.constant 0 : index
// CHECK:           %[[ORIG_DIM_VALUE:.*]] = tensor.dim %arg2, %[[DIM0]]
// CHECK:           %[[RETURN:.*]] = tensor.extract_slice %[[PADDED_RESULT]][0, 0] {{\[}}%[[ORIG_DIM_VALUE]], 1024] [1, 1] : tensor<?x1024xf32> to tensor<?x1024xf32>
// CHECK:           return %[[RETURN]] : tensor<?x1024xf32>
func.func @pad_matmul_dynamic_row(%arg0 : tensor<?x512xf32>, %arg1 : tensor<512x1024xf32>,
        %arg2 : tensor<?x1024xf32>) -> tensor<?x1024xf32> {
  %matmul = linalg.matmul
      ins(%arg0, %arg1 : tensor<?x512xf32>, tensor<512x1024xf32>)
      outs(%arg2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  return %matmul : tensor<?x1024xf32>
}

// -----
// CHECK-LABEL: @pad_matmul_dynamic_col
// Should trigger column alignment (32).
// Pad RHS:
// CHECK:           %[[RHS_ALIGNMENT:.*]] = arith.constant 32 : index
// CHECK:           %[[RHS_ALIGNED_DIM:.*]] = iree_input.align %{{.*}}, %[[RHS_ALIGNMENT]] : index
// CHECK:           %[[RHS_PADDED:.*]] = tensor.pad %arg1 low[0, 0] high[0, %[[RHS_ALIGNED_DIM]]]  {
// CHECK:           } : tensor<512x?xf32> to tensor<512x?xf32>
// Pad Output:
// CHECK:           %[[OUTPUT_ALIGNMENT:.*]] = arith.constant 32 : index
// CHECK:           %[[OUTPUT_ALIGNED_DIM:.*]] = iree_input.align %{{.*}}, %[[OUTPUT_ALIGNMENT]] : index
// CHECK:           %[[OUTPUT_PADDED:.*]] = tensor.pad %arg2 low[0, 0] high[0, %[[OUTPUT_ALIGNED_DIM]]]  {
// CHECK:           } : tensor<256x?xf32> to tensor<256x?xf32>
// Matmul:
// CHECK:           %{{.*}} = linalg.matmul ins(%arg0, %[[RHS_PADDED]] : tensor<256x512xf32>, tensor<512x?xf32>) outs(%[[OUTPUT_PADDED]] : tensor<256x?xf32>) -> tensor<256x?xf32>
func.func @pad_matmul_dynamic_col(%arg0 : tensor<256x512xf32>, %arg1 : tensor<512x?xf32>,
        %arg2 : tensor<256x?xf32>) -> tensor<256x?xf32> {
  %matmul = linalg.matmul
      ins(%arg0, %arg1 : tensor<256x512xf32>, tensor<512x?xf32>)
      outs(%arg2 : tensor<256x?xf32>) -> tensor<256x?xf32>
  return %matmul : tensor<256x?xf32>
}
