// RUN: iree-opt --iree-flow-set-encoding --cse --split-input-file %s | FileCheck %s

func.func @matmul_f32f32f32(%arg0 : tensor<100x250xf32>, %arg1 : tensor<250x500xf32>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf32>, tensor<250x500xf32>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_f32f32f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf32>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf32>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<100x250xf32> <user = MATMUL_F32F32F32, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x250xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS, orig_type = tensor<100x250xf32>>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<250x500xf32> <user = MATMUL_F32F32F32, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<250x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS, orig_type = tensor<250x500xf32>>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<100x500xf32> <user = MATMUL_F32F32F32, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT, orig_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f32f32f32_dynamic(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>,
    %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//      CHECK: func @matmul_f32f32f32_dynamic(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<?x?xf32> <user = MATMUL_F32F32F32, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<?x?xf32> <user = MATMUL_F32F32F32, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RHS>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<?x?xf32> <user = MATMUL_F32F32F32, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<?x?xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [{{.*}}] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_i8i8i32(%arg0 : tensor<100x250xi8>, %arg1 : tensor<250x500xi8>,
    %arg2 : tensor<100x500xi32>) -> tensor<100x500xi32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xi8>, tensor<250x500xi8>)
      outs(%arg2 : tensor<100x500xi32>) -> tensor<100x500xi32>
  return %0 : tensor<100x500xi32>
}
//      CHECK: func @matmul_i8i8i32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xi8>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xi8>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xi32>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<100x250xi8> <user = MATMUL_I8I8I32, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x250xi8> to tensor<?x?xi8>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = LHS, orig_type = tensor<100x250xi8>>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<250x500xi8> <user = MATMUL_I8I8I32, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<250x500xi8> to tensor<?x?xi8>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xi8, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = RHS, orig_type = tensor<250x500xi8>>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<100x500xi32> <user = MATMUL_I8I8I32, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x500xi32> to tensor<?x?xi32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xi32, #iree_linalg_ext.encoding<user = MATMUL_I8I8I32, role = RESULT, orig_type = tensor<100x500xi32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f16f16f32(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_f16f16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<100x250xf16> <user = MATMUL_F16F16F32, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = LHS, orig_type = tensor<100x250xf16>>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<250x500xf16> <user = MATMUL_F16F16F32, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = RHS, orig_type = tensor<250x500xf16>>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<100x500xf32> <user = MATMUL_F16F16F32, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F16F16F32, role = RESULT, orig_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_f16f16f16(%arg0 : tensor<100x250xf16>, %arg1 : tensor<250x500xf16>,
    %arg2 : tensor<100x500xf16>) -> tensor<100x500xf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xf16>, tensor<250x500xf16>)
      outs(%arg2 : tensor<100x500xf16>) -> tensor<100x500xf16>
  return %0 : tensor<100x500xf16>
}
//      CHECK: func @matmul_f16f16f16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf16>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<100x250xf16> <user = MATMUL_F16F16F16, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x250xf16> to tensor<?x?xf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = LHS, orig_type = tensor<100x250xf16>>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<250x500xf16> <user = MATMUL_F16F16F16, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<250x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = RHS, orig_type = tensor<250x500xf16>>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<100x500xf16> <user = MATMUL_F16F16F16, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x500xf16> to tensor<?x?xf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf16, #iree_linalg_ext.encoding<user = MATMUL_F16F16F16, role = RESULT, orig_type = tensor<100x500xf16>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_bf16bf16f32(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xf32>) -> tensor<100x500xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xf32>) -> tensor<100x500xf32>
  return %0 : tensor<100x500xf32>
}
//      CHECK: func @matmul_bf16bf16f32(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xf32>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<100x250xbf16> <user = MATMUL_BF16BF16F32, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = LHS, orig_type = tensor<100x250xbf16>>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<250x500xbf16> <user = MATMUL_BF16BF16F32, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = RHS, orig_type = tensor<250x500xbf16>>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<100x500xf32> <user = MATMUL_BF16BF16F32, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x500xf32> to tensor<?x?xf32>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16F32, role = RESULT, orig_type = tensor<100x500xf32>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @matmul_bf16bf16bf16(%arg0 : tensor<100x250xbf16>, %arg1 : tensor<250x500xbf16>,
    %arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<100x250xbf16>, tensor<250x500xbf16>)
      outs(%arg2 : tensor<100x500xbf16>) -> tensor<100x500xbf16>
  return %0 : tensor<100x500xbf16>
}
//      CHECK: func @matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:.+]]: tensor<100x250xbf16>
// CHECK-SAME:     %[[ARG1:.+]]: tensor<250x500xbf16>
// CHECK-SAME:     %[[ARG2:.+]]: tensor<100x500xbf16>
//      CHECK:   %[[LHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG0]] : tensor<100x250xbf16> <user = MATMUL_BF16BF16BF16, role = LHS> -> index, index
//      CHECK:   %[[LHS_PAD:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[%[[LHS_PADDING_SIZE]]#0, %[[LHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x250xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[LHS:.+]] = iree_linalg_ext.set_encoding %[[LHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = LHS, orig_type = tensor<100x250xbf16>>>
//      CHECK:   %[[RHS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG1]] : tensor<250x500xbf16> <user = MATMUL_BF16BF16BF16, role = RHS> -> index, index
//      CHECK:   %[[RHS_PAD:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[%[[RHS_PADDING_SIZE]]#0, %[[RHS_PADDING_SIZE]]#1]
//      CHECK:       tensor<250x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[RHS:.+]] = iree_linalg_ext.set_encoding %[[RHS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = RHS, orig_type = tensor<250x500xbf16>>>
//      CHECK:   %[[OUTS_PADDING_SIZE:.+]]:2 = iree_linalg_ext.encoding_padding_size %[[ARG2]] : tensor<100x500xbf16> <user = MATMUL_BF16BF16BF16, role = RESULT> -> index, index
//      CHECK:   %[[OUTS_PAD:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[%[[OUTS_PADDING_SIZE]]#0, %[[OUTS_PADDING_SIZE]]#1]
//      CHECK:       tensor<100x500xbf16> to tensor<?x?xbf16>
//      CHECK:   %[[OUTS:.+]] = iree_linalg_ext.set_encoding %[[OUTS_PAD]]
// CHECK-SAME:       tensor<?x?xbf16, #iree_linalg_ext.encoding<user = MATMUL_BF16BF16BF16, role = RESULT, orig_type = tensor<100x500xbf16>>>
//      CHECK:   %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:       ins(%[[LHS]], %[[RHS]] :
// CHECK-SAME:       outs(%[[OUTS]] :
//      CHECK:   %[[RESULT_PADDED:.+]] = iree_linalg_ext.unset_encoding %[[MATMUL]]
//      CHECK:   %[[RESULT:.+]] = tensor.extract_slice %[[RESULT_PADDED]][0, 0] [100, 500] [1, 1]
//      CHECK:   return %[[RESULT]]

// -----

func.func @fold_fill_with_set_encoding(%arg0 : index, %arg1 : index)
  -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = iree_linalg_ext.set_encoding %1 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
  return %2 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
}
//      CHECK: func @fold_fill_with_set_encoding(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(%{{.+}}, %{{.+}}) : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = LHS>>)
//      CHECK:   return %[[FILL]]

// -----

func.func @fold_fill_with_tensor_pad(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index)
    -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>> {
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty(%arg0, %arg1) : tensor<?x?xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = tensor.pad %1 low[0, 0] high[%arg2, %arg3] {
  ^bb0(%b0: index, %b1 : index):
    tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<?x?xf32>
  %3 = iree_linalg_ext.set_encoding %2 : tensor<?x?xf32>
      -> tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
  return %3 : tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
}
//      CHECK: func @fold_fill_with_tensor_pad(
//      CHECK:   %[[EMPTY:.+]] = tensor.empty(
// CHECK-SAME:       tensor<?x?xf32, #iree_linalg_ext.encoding<user = MATMUL_F32F32F32, role = RESULT>>
//      CHECK:   %[[FILL:.+]] = linalg.fill
// CHECK-SAME:       outs(%[[EMPTY]] :
//      CHECK:   return %[[FILL]]
