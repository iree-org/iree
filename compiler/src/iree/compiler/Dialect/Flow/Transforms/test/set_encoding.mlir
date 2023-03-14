// RUN: iree-opt --iree-flow-set-encoding --cse --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-flow-set-encoding{default-padding=4}, cse))" --split-input-file %s | FileCheck %s --check-prefix=PADDING

func.func @matmul_no_padding(%arg0 : tensor<128x256xf32>, %arg1 : tensor<256x512xf32>,
    %arg2 : tensor<128x512xf32>) -> tensor<128x512xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x256xf32>, tensor<256x512xf32>)
      outs(%arg2 : tensor<128x512xf32>) -> tensor<128x512xf32>
  return %0 : tensor<128x512xf32>
}
//      CHECK: func @matmul_no_padding(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<128x256xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<256x512xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<128x512xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[ARG0]]
// CHECK-SAME:       tensor<128x256xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[ARG1]]
// CHECK-SAME:       tensor<256x512xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[ARG2]]
// CHECK-SAME:       tensor<128x512xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_padding(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_padding(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[12, 6]
//      CHECK:       tensor<100x250xf32> to tensor<112x256xf32>
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[6, 12]
//      CHECK:       tensor<250x500xf32> to tensor<256x512xf32>
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[12, 12]
//      CHECK:       tensor<100x500xf32> to tensor<112x512xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<112x256xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<256x512xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<112x512xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

//      PADDING: func @matmul_padding(
// PADDING-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// PADDING-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// PADDING-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      PADDING:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[0, 2]
//      PADDING:       tensor<100x250xf32> to tensor<100x252xf32>
//      PADDING:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[2, 0]
//      PADDING:       tensor<250x500xf32> to tensor<252x500xf32>
//      PADDING:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// PADDING-SAME:       tensor<100x252xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
//      PADDING:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// PADDING-SAME:       tensor<252x500xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
//      PADDING:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[ARG2]]
// PADDING-SAME:       tensor<100x500xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
//      PADDING:   %[[MATMUL:.+]] = linalg.matmul
// PADDING-SAME:       ins(%[[LHS]], %[[RHS]] :
// PADDING-SAME:       outs(%[[OUTS]] :
//      PADDING:   %[[RESULT:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      PADDING:   return %[[RESULT]]


// -----

func.func @matmul_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (-s0 + (s0 ceildiv 16) * 16)
//      CHECK: func @matmul_dynamic(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[LHS_D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[LHS_D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[HIGHPAD_LHS_0:.+]] = affine.apply #[[MAP]]()[%[[LHS_D0]]]
//  CHECK-DAG:   %[[HIGHPAD_LHS_1:.+]] = affine.apply #[[MAP]]()[%[[LHS_D1]]]
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[HIGHPAD_LHS_0]], %[[HIGHPAD_LHS_1]]]
//  CHECK-DAG:   %[[RHS_D0:.+]] = tensor.dim %[[ARG1]], %[[C0]]
//  CHECK-DAG:   %[[RHS_D1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:   %[[HIGHPAD_RHS_0:.+]] = affine.apply #[[MAP]]()[%[[RHS_D0]]]
//  CHECK-DAG:   %[[HIGHPAD_RHS_1:.+]] = affine.apply #[[MAP]]()[%[[RHS_D1]]]
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[HIGHPAD_RHS_0]], %[[HIGHPAD_RHS_1]]]
//  CHECK-DAG:   %[[OUTS_D0:.+]] = tensor.dim %[[ARG2]], %[[C0]]
//  CHECK-DAG:   %[[OUTS_D1:.+]] = tensor.dim %[[ARG2]], %[[C1]]
//  CHECK-DAG:   %[[HIGHPAD_OUTS_0:.+]] = affine.apply #[[MAP]]()[%[[OUTS_D0]]]
//  CHECK-DAG:   %[[HIGHPAD_OUTS_1:.+]] = affine.apply #[[MAP]]()[%[[OUTS_D1]]]
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[HIGHPAD_OUTS_0]], %[[HIGHPAD_OUTS_1]]]
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RHS>>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [%[[OUTS_D0]], %[[OUTS_D1]]] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @fold_fill_with_set_encoding(%arg0 : index, %arg1 : index)
  -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = iree_linalg_ext.set_encoding %1 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
  return %2 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
}
//      CHECK: func @fold_fill_with_set_encoding(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_LHS>>)
//      CHECK:   return %[[FILL]]

// -----

func.func @fold_fill_with_tensor_pad(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[0, 0] high[%arg2, %arg3] {
  ^bb0(%b0: index, %b1 : index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %2 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
  return %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
}
//      CHECK: func @fold_fill_with_tensor_pad(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<MATMUL_F32F32F32_RESULT>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   return %[[FILL]]
