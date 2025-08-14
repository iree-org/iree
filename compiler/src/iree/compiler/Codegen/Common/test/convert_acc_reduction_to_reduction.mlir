// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-convert-acc-reduction-to-reduction,canonicalize))" %s | FileCheck %s
// The test relies on canonicalizer to fold neutral constants away. It happens
// when the filled value is as the same as the neutral constannt.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @accumulate_gemm(%arg0 : tensor<512x128xi8>, %arg1 : tensor<512x128xi8>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x512xi32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x512xi32>>
  %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x512xi32>> -> tensor<512x512xi32>
  %3 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<512x128xi8>, tensor<512x128xi8>) outs(%2 : tensor<512x512xi32>) {
      ^bb0(%in: i8, %in_0: i8, %out: i32):
        %5 = arith.extsi %in : i8 to i32
        %6 = arith.extsi %in_0 : i8 to i32
        %7 = arith.muli %5, %6 : i32
        %8 = arith.addi %out, %7 : i32
        linalg.yield %8 : i32
      } -> tensor<512x512xi32>
  iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : tensor<512x512xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<512x512xi32>>
  return
}

// CHECK-LABEL: func.func @accumulate_gemm
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<512x512xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<512x512xi32>) -> tensor<512x512xi32>
//       CHECK: %[[GEMM:.+]] = linalg.generic {{.*}} outs(%[[FILL]] : tensor<512x512xi32>) {
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[GEMM]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: iree_tensor_ext.dispatch.tensor.store %[[ADD]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
]>
#contraction_accesses = [
 affine_map<(i, j, k) -> (i, k)>,
 affine_map<(i, j, k) -> (k, j)>,
 affine_map<(i, j, k) -> (i, j)>
]
func.func @accumulate_inner_tiled(%arg0 : tensor<?x?x4xf16>, %arg1 : tensor<?x?x4xf16>, %d0: index, %d1: index) -> tensor<?x?x4xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<?x?x4xf32, #hal.descriptor_type<storage_buffer>>{%d0, %d1}
  %1 = iree_codegen.load_from_buffer %0 : memref<?x?x4xf32, #hal.descriptor_type<storage_buffer>> -> tensor<?x?x4xf32>
  %2 = iree_codegen.inner_tiled ins(%arg0, %arg1) outs(%1) {
    indexing_maps = #contraction_accesses,
    iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
    kind = #iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>
  } : tensor<?x?x4xf16>, tensor<?x?x4xf16> into tensor<?x?x4xf32>
  return %2 : tensor<?x?x4xf32>
}
// CHECK-LABEL: func.func @accumulate_inner_tiled
//       CHECK: %[[FILL:.+]] = linalg.fill
//       CHECK: %[[GEMM:.+]] = iree_codegen.inner_tiled {{.*}} outs(%[[FILL]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
]>
func.func @acc_conv_nchw(%arg0: tensor<1x64x58x58xf32>, %arg1: tensor<64x64x3x3xf32>) -> tensor<1x64x56x56xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x64x56x56xf32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 64, 56, 56], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x64x56x56xf32>> -> tensor<1x64x56x56xf32>
  %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<1x64x58x58xf32>, tensor<64x64x3x3xf32>) outs(%1 : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
  return %2 : tensor<1x64x56x56xf32>
}

// CHECK-LABEL: func.func @acc_conv_nchw
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1x64x56x56xf32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[C0]] : f32) outs(%[[EMPTY]] : tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
//       CHECK: %[[CONV:.+]] = linalg.conv_2d_nchw_fchw {{.*}} outs(%[[FILL]] : tensor<1x64x56x56xf32>)
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[CONV]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: return %[[ADD]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
]>
func.func @acc_fill_gemm(%arg0: tensor<512x128xi8>, %arg1: tensor<512x128xi8>) -> tensor<512x512xi32> {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x512xi32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<512x512xi32>> -> tensor<512x512xi32>
  %2 = linalg.fill ins(%c0_i32 : i32) outs(%1 : tensor<512x512xi32>) -> tensor<512x512xi32>
  %3 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<512x128xi8>, tensor<512x128xi8>) outs(%2 : tensor<512x512xi32>) -> tensor<512x512xi32>
  return %3 : tensor<512x512xi32>
}
// CHECK-LABEL: func.func @acc_fill_gemm
//   CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<512x512xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:   ins(%[[C0]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: %[[GEMM:.+]] = linalg.matmul_transpose_b
//  CHECK-SAME:   outs(%[[FILL]]
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[GEMM]], %[[C0]]
//  CHECK-SAME:   outs(%[[EMPTY]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
]>
func.func @acc_fill_pooling_nchw_max(%arg0: tensor<1x832x9x9xf32>, %arg1: tensor<3x3xf32>) -> tensor<1x832x7x7xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x832x7x7xf32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 832, 7, 7], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x832x7x7xf32>> -> tensor<1x832x7x7xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x832x7x7xf32>) -> tensor<1x832x7x7xf32>
  %3 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<1x832x9x9xf32>, tensor<3x3xf32>) outs(%2 : tensor<1x832x7x7xf32>) -> tensor<1x832x7x7xf32>
  return %3 : tensor<1x832x7x7xf32>
}
// CHECK-LABEL: func.func @acc_fill_pooling_nchw_max
//   CHECK-DAG: %[[C_IDENTITY:.+]] = arith.constant 0xFF800000 : f32
//   CHECK-DAG: %[[C_INIT:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1x832x7x7xf32>
//       CHECK: %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:   ins(%[[C_IDENTITY]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: %[[POOL:.+]] = linalg.pooling_nchw_max
//  CHECK-SAME:   outs(%[[FILL]]
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[POOL]], %[[C_INIT]]
//  CHECK-SAME:   outs(%[[EMPTY]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
]>
func.func @acc_fill_pooling_nhwc_min(%9: tensor<1x4x16x1xf32>, %10: tensor<2x2xf32>) -> tensor<1x2x4x1xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2x4x1xf32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 2, 4, 1], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x2x4x1xf32>> -> tensor<1x2x4x1xf32>
  %cst = arith.constant 0.0 : f32
  %11 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  %12 = linalg.pooling_nhwc_min {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%9, %10 : tensor<1x4x16x1xf32>, tensor<2x2xf32>) outs(%11 : tensor<1x2x4x1xf32>) -> tensor<1x2x4x1xf32>
  return %12 : tensor<1x2x4x1xf32>
}
// CHECK-LABEL: func.func @acc_fill_pooling_nhwc_min
//   CHECK-DAG: %[[C_IDENTITY:.+]] = arith.constant 0x7F800000 : f32
//   CHECK-DAG: %[[C_INIT:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1x2x4x1xf32>
//       CHECK: %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:   ins(%[[C_IDENTITY]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: %[[POOL:.+]] = linalg.pooling_nhwc_min
//  CHECK-SAME:   outs(%[[FILL]]
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[POOL]], %[[C_INIT]]
//  CHECK-SAME:   outs(%[[EMPTY]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
]>
func.func @acc_fill_pooling_nchw_sum(%9: tensor<1x832x9x9xf32>, %10: tensor<3x3xf32>) -> tensor<1x832x7x7xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x832x7x7xf32>>
  %1 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 832, 7, 7], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x832x7x7xf32>> -> tensor<1x832x7x7xf32>
  %cst = arith.constant -1.0 : f32
  %11 = linalg.fill ins(%cst : f32) outs(%1 : tensor<1x832x7x7xf32>) -> tensor<1x832x7x7xf32>
  %12 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%9, %10 : tensor<1x832x9x9xf32>, tensor<3x3xf32>) outs(%11 : tensor<1x832x7x7xf32>) -> tensor<1x832x7x7xf32>
  return %12 : tensor<1x832x7x7xf32>
}
// CHECK-LABEL: func.func @acc_fill_pooling_nchw_sum
//   CHECK-DAG: %[[C_IDENTITY:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-DAG: %[[C_INIT:.+]] = arith.constant -1.000000e+00 : f32
//   CHECK-DAG: %[[EMPTY:.+]] = tensor.empty() : tensor<1x832x7x7xf32>
//       CHECK: %[[FILL:.+]] = linalg.fill
//  CHECK-SAME:   ins(%[[C_IDENTITY]]
//  CHECK-SAME:   outs(%[[EMPTY]]
//       CHECK: %[[POOL:.+]] = linalg.pooling_nchw_sum
//  CHECK-SAME:   outs(%[[FILL]]
//       CHECK: %[[ADD:.+]] = linalg.generic {{.+}} ins(%[[POOL]], %[[C_INIT]]
//  CHECK-SAME:   outs(%[[EMPTY]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
func.func @nonacc_gemm(%1 : tensor<512x128xi8>, %2 : tensor<512x128xi8>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x512xi32>>
  %empty = tensor.empty() : tensor<512x512xi32>
  %fill = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<512x512xi32>) -> tensor<512x512xi32>
  %5 = linalg.matmul_transpose_b
    ins(%1, %2 : tensor<512x128xi8>, tensor<512x128xi8>) outs(%fill : tensor<512x512xi32>) -> tensor<512x512xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [512, 512], strides = [1, 1] : tensor<512x512xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512x512xi32>>
  return
}

// CHECK-LABEL: func.func @nonacc_gemm
//       CHECK: linalg.matmul_transpose_b
//   CHECK-NOT: linalg.generic
