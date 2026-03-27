// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-convolution-to-igemm),canonicalize,cse)" %s | FileCheck %s

#map = affine_map<(d0, d1, d2, d3)->(d0, d1, d2, d3)>
func.func public @conv_with_consumer(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf16> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x14x14x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %0 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %1 = tensor.empty() : tensor<1x14x14x16xf16>
  %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x14x14x16xf32>) outs(%1 : tensor<1x14x14x16xf16>) {
  ^bb0(%in: f32, %out: f16):
    %3 = arith.truncf %in : f32 to f16
    linalg.yield %3 : f16
  } -> tensor<1x14x14x16xf16>
  return %2 : tensor<1x14x14x16xf16>
}
// CHECK:      func.func public @conv_with_consumer
// CHECK-DAG:    %[[IM2COL:.+]] = iree_linalg_ext.im2col {{.*}} -> tensor<1x14x14x36xf32>
// CHECK-DAG:    %[[FILL:.+]] = linalg.fill {{.*}} -> tensor<1x14x14x16xf32>
// CHECK:        %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:     ins(%[[IM2COL]], %{{.*}} : tensor<1x14x14x36xf32>
// CHECK-SAME:     outs(%[[FILL]] : tensor<1x14x14x16xf32>)
// CHECK:        %[[TRUNCF:.+]] = linalg.generic
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[MATMUL]] : tensor<1x14x14x16xf32>)
// CHECK:        return {{.*}} : tensor<1x14x14x16xf16>

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @fold_with_interface_tensor() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x16x4xf32>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4x16xf32>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>>
    %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [1, 16, 16, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x16x4xf32>> -> tensor<1x16x16x4xf32>
    %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 4, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x4x16xf32>> -> tensor<3x3x4x16xf32>
    %5 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 16], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>> -> tensor<1x14x14x16xf32>
    %cst = arith.constant 0.0 : f32
    %fill = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    %6 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%3, %4: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0, 0, 0, 0], sizes = [1, 14, 14, 16], strides = [1, 1, 1, 1] : tensor<1x14x14x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x14x14x16xf32>>
    return
  }
}

// CHECK:      func.func @fold_with_interface_tensor
// CHECK-DAG:  %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load {{.*}} -> tensor<1x16x16x4xf32>
// CHECK-DAG:  %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load {{.*}} -> tensor<36x16xf32>
// CHECK-DAG:  %[[RES:.+]] = iree_tensor_ext.dispatch.tensor.load {{.*}} -> tensor<1x14x14x16xf32>
// CHECK-DAG:  %[[IM2COL:.+]] = iree_linalg_ext.im2col {{.*}} ins(%[[LHS]] : tensor<1x16x16x4xf32>){{.*}}-> tensor<1x14x14x36xf32>
// CHECK-DAG:  %[[FILL:.+]] = linalg.fill {{.*}}outs(%[[RES]] : tensor<1x14x14x16xf32>)
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[RHS]] : tensor<1x14x14x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[FILL]] : tensor<1x14x14x16xf32>) {
// CHECK:      iree_tensor_ext.dispatch.tensor.store %[[MATMUL]]

// -----

func.func @fold_with_buffer_load_store(
    %arg0: memref<1x16x16x4xf32>,
    %arg1: memref<3x3x4x16xf32>,
    %arg2: memref<1x14x14x16xf32>) {
  %0 = iree_codegen.load_from_buffer %arg0 : memref<1x16x16x4xf32> -> tensor<1x16x16x4xf32>
  %1 = iree_codegen.load_from_buffer %arg1 : memref<3x3x4x16xf32> -> tensor<3x3x4x16xf32>
  %2 = iree_codegen.load_from_buffer %arg2 : memref<1x14x14x16xf32> -> tensor<1x14x14x16xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  %3 = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
     ins(%0, %1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
    outs(%fill: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
  iree_codegen.store_to_buffer %3, %arg2 : tensor<1x14x14x16xf32> into memref<1x14x14x16xf32>
  return
}

// CHECK:      func.func @fold_with_buffer_load_store
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: memref<1x16x16x4xf32>
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9]+]]: memref<3x3x4x16xf32>
// CHECK-SAME:   %[[OUTPUT:[a-zA-Z0-9]+]]: memref<1x14x14x16xf32>
// CHECK-DAG:  %[[LHS:.+]] = iree_codegen.load_from_buffer %[[INPUT]] : memref<1x16x16x4xf32> -> tensor<1x16x16x4xf32>
// CHECK-DAG:  %[[RHS:.+]] = iree_codegen.load_from_buffer {{.*}} : memref<36x16xf32> -> tensor<36x16xf32>
// CHECK-DAG:  %[[RES:.+]] = iree_codegen.load_from_buffer {{.*}} : memref<1x14x14x16xf32> -> tensor<1x14x14x16xf32>
// CHECK-DAG:  %[[IM2COL:.+]] = iree_linalg_ext.im2col {{.*}} ins(%[[LHS]] : tensor<1x16x16x4xf32>){{.*}}-> tensor<1x14x14x36xf32>
// CHECK-DAG:  %[[FILL:.+]] = linalg.fill {{.*}}outs(%[[RES]] : tensor<1x14x14x16xf32>)
// CHECK:      %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:   ins(%[[IM2COL]], %[[RHS]] : tensor<1x14x14x36xf32>, tensor<36x16xf32>)
// CHECK-SAME:   outs(%[[FILL]] : tensor<1x14x14x16xf32>) {
// CHECK:      iree_codegen.store_to_buffer %[[MATMUL]]

// -----

func.func @conv_with_lowering_config() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_gpu.pipeline<TileAndFuse> workgroup_size = [256, 1, 1] subgroup_size = 64, {gpu_pipeline_options = #iree_gpu.pipeline_options<prefetch_num_stages = 2, no_reduce_shared_memory_bank_conflicts = false>}>} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x64xf32>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [2, 34, 34, 128], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x34x34x128xf32>> -> tensor<2x34x34x128xf32>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [3, 3, 128, 64], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<3x3x128x64xf32>> -> tensor<3x3x128x64xf32>
  %5 = tensor.empty() : tensor<2x32x32x64xf32>
  %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  %7 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, lowering_config = #iree_gpu.lowering_config<{mma_kind = #iree_gpu.mma_layout<MFMA_F32_16x16x4_F32>, promote_operands = [0, 1], reduction = [0, 0, 0, 0, 8], subgroup = [1, 2, 2, 1, 0], workgroup = [1, 2, 2, 4, 0]}>, strides = dense<1> : tensor<2xi64>} ins(%3, %4 : tensor<2x34x34x128xf32>, tensor<3x3x128x64xf32>) outs(%6 : tensor<2x32x32x64xf32>) -> tensor<2x32x32x64xf32>
  iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0, 0, 0], sizes = [2, 32, 32, 64], strides = [1, 1, 1, 1] : tensor<2x32x32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x32x32x64xf32>>
  return
}

// CHECK:      func.func @conv_with_lowering_config
// CHECK-DAG:    %[[IM2COL:.+]] = iree_linalg_ext.im2col
// CHECK-DAG:    %[[FILL:.+]] = linalg.fill
// CHECK:        %[[MATMUL:.+]] = linalg.generic {{.*}} ins(%[[IM2COL]], {{.*}}) outs(%[[FILL]] : {{.*}}) {{.*}}lowering_config = {{.*}}
// CHECK:        iree_tensor_ext.dispatch.tensor.store %[[MATMUL]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func public @no_conv_contraction(%arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>) -> tensor<128x128xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<128x128xf32>) -> tensor<128x128xf32>
  %matmul = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%fill : tensor<128x128xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %0 = arith.mulf %in, %in_0 : f32
    %1 = arith.addf %0, %out : f32
    linalg.yield %1 : f32
  } -> tensor<128x128xf32>
  return %matmul : tensor<128x128xf32>
}
// CHECK:       func.func public @no_conv_contraction
// CHECK-NOT:     iree_linalg_ext.im2col
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]

// -----

// Test that without a conv, the pass decomposes and re-fuses the broadcast
// so the generic is unchanged.

#map_id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_bcast = affine_map<(d0, d1, d2) -> (d2)>
func.func @elementwise_broadcast_roundtrip(
    %arg0: tensor<1x196x16xf32>,
    %bias: tensor<16xf32>) -> tensor<1x196x16xf32> {
  %empty = tensor.empty() : tensor<1x196x16xf32>
  %result = linalg.generic {
    indexing_maps = [#map_id, #map_bcast, #map_id],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0, %bias : tensor<1x196x16xf32>, tensor<16xf32>)
    outs(%empty : tensor<1x196x16xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %add = arith.addf %in, %b : f32
    linalg.yield %add : f32
  } -> tensor<1x196x16xf32>
  return %result : tensor<1x196x16xf32>
}
// CHECK-LABEL: func.func @elementwise_broadcast_roundtrip
//       CHECK:   %[[RES:.+]] = linalg.generic
//  CHECK-SAME:     ins(%{{.*}}, %{{.*}} : tensor<1x196x16xf32>, tensor<16xf32>)
//   CHECK-NOT:   linalg.broadcast
//       CHECK:   return %[[RES]]

// -----

// Test that an expand_shape before a broadcasted element-wise consumer
// propagates through the generic and into the store_to_buffer. The broadcast
// decomposition turns the non-identity map into identity maps so the
// expand_shape can fold through, then the broadcast is fused back.

#map_id_4d = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_bcast_4d = affine_map<(d0, d1, d2, d3) -> (d2)>
func.func @expand_shape_propagation_with_broadcast(
    %arg0: tensor<1x196x16xf32>,
    %bias: tensor<14xf32>,
    %arg2: memref<1x14x14x16xf32>) {
  %expanded = tensor.expand_shape %arg0 [[0], [1, 2], [3]]
    output_shape [1, 14, 14, 16] : tensor<1x196x16xf32> into tensor<1x14x14x16xf32>
  %empty = tensor.empty() : tensor<1x14x14x16xf32>
  %add = linalg.generic {
    indexing_maps = [#map_id_4d, #map_bcast_4d, #map_id_4d],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%expanded, %bias : tensor<1x14x14x16xf32>, tensor<14xf32>)
    outs(%empty : tensor<1x14x14x16xf32>) {
  ^bb0(%in: f32, %b: f32, %out: f32):
    %sum = arith.addf %in, %b : f32
    linalg.yield %sum : f32
  } -> tensor<1x14x14x16xf32>
  iree_codegen.store_to_buffer %add, %arg2
    : tensor<1x14x14x16xf32> into memref<1x14x14x16xf32>
  return
}
// The expand_shape propagates through the generic to the store boundary.
// The output buffer gets a memref.collapse_shape and the store operates in
// the collapsed 3D shape.
// CHECK-LABEL: func.func @expand_shape_propagation_with_broadcast
//  CHECK-SAME:   %[[INPUT:[a-zA-Z0-9]+]]: tensor<1x196x16xf32>
//  CHECK-SAME:   %[[OUTPUT_BUF:[a-zA-Z0-9]+]]: memref<1x14x14x16xf32>
//       CHECK:   %[[COLLAPSED_OUT:.+]] = memref.collapse_shape %[[OUTPUT_BUF]]
//  CHECK-SAME:     memref<1x14x14x16xf32> into memref<1x196x16xf32>
//   CHECK-NOT:   tensor.expand_shape
//       CHECK:   %[[BCAST:.+]] = linalg.generic
//  CHECK-SAME:     ins(%{{.*}} : tensor<14xf32>)
//  CHECK-SAME:     outs(%{{.*}} : tensor<1x14x14x16xf32>)
//       CHECK:   %[[COLLAPSED_BCAST:.+]] = tensor.collapse_shape %[[BCAST]]
//  CHECK-SAME:     tensor<1x14x14x16xf32> into tensor<1x196x16xf32>
//       CHECK:   %[[ADD:.+]] = linalg.generic
//  CHECK-SAME:     ins(%[[INPUT]], %[[COLLAPSED_BCAST]] : tensor<1x196x16xf32>, tensor<1x196x16xf32>)
//       CHECK:   iree_codegen.store_to_buffer %[[ADD]], %[[COLLAPSED_OUT]]
//  CHECK-SAME:     tensor<1x196x16xf32> into memref<1x196x16xf32>
