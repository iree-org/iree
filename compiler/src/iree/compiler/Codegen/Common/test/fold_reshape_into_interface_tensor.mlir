// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-fold-reshape-into-interface-tensor,canonicalize))" \
// RUN:   --split-input-file %s --mlir-print-local-scope | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_collapse_into_loads_dynamic() -> tensor<?x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = tensor.collapse_shape %2 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  return %3 : tensor<?x32xf32>
}
// CHECK-LABEL: func @fold_collapse_into_loads_dynamic()
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0], sizes = [%[[SHAPE]], 32], strides = [1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">], flags = Indirect>
func.func @fold_expand_into_loads_dynamic() -> tensor<2x?x16x32xf32> {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0}
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, %0, 32], strides = [1, 1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x32xf32>>{%0} -> tensor<2x?x32xf32>
  %3 = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%0]
  %4 = tensor.expand_shape %2 [[0], [1, 2], [3]] output_shape [2, %3, 16, 32] : tensor<2x?x32xf32> into tensor<2x?x16x32xf32>
  return %4 : tensor<2x?x16x32xf32>
}
// CHECK-LABEL: func @fold_expand_into_loads_dynamic()
//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//   CHECK-DAG:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C16]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}
//       CHECK:   %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [2, %[[SHAPE]], 16, 32], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x?x16x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic(
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_rank_reduced_outer(%arg0 : tensor<2x?xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x?xf32> into tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 1], strides = [1, 1]
      : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_rank_reduced_outer(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 1], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_rank_reduced_inner(%arg0 : tensor<2x?xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x?xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1]] : tensor<2x?xf32> into tensor<?xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [1, %0], strides = [1, 1]
      : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x?xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_rank_reduced_inner(
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = arith.divsi %[[CONST]], %[[C2]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x2x?xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [1, 2, %[[SHAPE]]], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x2x?xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_partial_size(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_partial_size(
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_partial_with_offset(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 8], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x40xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_partial_with_offset(
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 8], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x40xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_same(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_same(
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//   CHECK-DAG:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_slice_1d(%arg0 : tensor<3x?x16xf32>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<3x?x16xf32> into tensor<?x16xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, %arg1], sizes = [%0, 16], strides = [1, 1]
      : tensor<?x16xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_slice_1d(
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9]+]]: index
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//   CHECK-DAG:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 3)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, %[[ARG1]]], sizes = [3, %[[SHAPE]], 16], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<3x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_slice_3d(%arg0 : tensor<4x8x4x128xf32>) {
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4608xf32>>
  %2 = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<4x8x4x128xf32> into tensor<4x4096xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 1024], sizes = [4, 4096], strides = [1, 1]
      : tensor<4x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4608xf32>>
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_slice_3d(
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x9x4x128xf32>>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 2, 0, 0], sizes = [4, 8, 4, 128], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x9x4x128xf32>>

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_collapse_into_stores_dynamic_diff(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %3 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [0, 0], sizes = [%1, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @fold_collapse_into_stores_dynamic_diff
//       CHECK:   %[[CONST:.+]] = hal.interface.constant.load
//       CHECK:   %[[SHAPE:.+]] = affine.apply affine_map<()[s0] -> (s0 ceildiv 2)>()[%[[CONST]]]
//       CHECK:   %[[SUBSPAN:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN]]
//  CHECK-SAME:       offsets = [0, 0, 0], sizes = [2, %[[SHAPE]], 32], strides = [1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x?x32xf32>>{%[[SHAPE]]}

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @unsupported_multiple_dynamic_dims_in_group(%arg0 : tensor<?x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : index
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %3 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<?x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %3, %2, offsets = [0, 0], sizes = [%1, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @unsupported_multiple_dynamic_dims_in_group
//       CHECK:   tensor.collapse_shape

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @unsupported_dynamic_store_into_static_subspan(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32xf32>>
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16x32xf32>>
  return
}
// CHECK-LABEL: func @unsupported_dynamic_store_into_static_subspan
//       CHECK:   tensor.collapse_shape

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @unsupported_offset_in_dynamic_dim(%arg0 : tensor<2x?x32xf32>) {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  %2 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x?x32xf32> into tensor<?x32xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [8, 0], sizes = [%0, 32], strides = [1, 1]
      : tensor<?x32xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x32xf32>>{%0}
  return
}
// CHECK-LABEL: func @unsupported_offset_in_dynamic_dim
//       CHECK:   tensor.collapse_shape

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @unsupported_offset_in_static_dims(%arg0 : tensor<4x8x4x128xf32>) {
  %c0 = arith.constant 0 : index
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0)
      flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4608xf32>>
  %2 = tensor.collapse_shape %arg0 [[0], [1, 2, 3]] : tensor<4x8x4x128xf32> into tensor<4x4096xf32>
  iree_tensor_ext.dispatch.tensor.store %2, %1, offsets = [0, 128], sizes = [4, 4096], strides = [1, 1]
      : tensor<4x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x4608xf32>>
  return
}
// CHECK-LABEL: func @unsupported_offset_in_static_dims
//       CHECK:   tensor.collapse_shape

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
    #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @fold_expand_and_collapse(%arg0 : tensor<1x?x1x8xi32>) {
  %c128 = arith.constant 128 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : index
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c128) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x8xi32>>{%0}
  %2 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [%0, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x8xi32>>{%0} -> tensor<?x8xi32>
  %3 = tensor.expand_shape %2 [[0, 1], [2, 3]] output_shape [1, %0, 1, 8] : tensor<?x8xi32> into tensor<1x?x1x8xi32>
  %4 = linalg.copy ins(%arg0 : tensor<1x?x1x8xi32>)
        outs(%3 : tensor<1x?x1x8xi32>) -> tensor<1x?x1x8xi32>
  %5 = tensor.collapse_shape %4 [[0, 1], [2, 3]] : tensor<1x?x1x8xi32> into tensor<?x8xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [%0, 8], strides = [1, 1] : tensor<?x8xi32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x8xi32>>{%0}
  return
}
// CHECK-LABEL: func @fold_expand_and_collapse
//       CHECK:   %[[SHAPE:.+]] = hal.interface.constant.load
//       CHECK:   %[[SUBSPAN1:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x?x1x8xi32>>{%[[SHAPE]]}
//       CHECK:   %[[SUBSPAN2:.+]] = hal.interface.binding.subspan
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x?x1x8xi32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.load %[[SUBSPAN1]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, %[[SHAPE]], 1, 8], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x?x1x8xi32>>{%[[SHAPE]]}
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %{{.+}}, %[[SUBSPAN2]]
//  CHECK-SAME:       offsets = [0, 0, 0, 0], sizes = [1, %[[SHAPE]], 1, 8], strides = [1, 1, 1, 1]
//  CHECK-SAME:       !iree_tensor_ext.dispatch.tensor<readwrite:tensor<1x?x1x8xi32>>{%[[SHAPE]]}
