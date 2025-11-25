// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-boundary-pack-unpack-ops))" --split-input-file %s | FileCheck %s -check-prefixes=CHECK

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @pack_at_source() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32>> -> tensor<16x16xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %pack = linalg.pack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<16x16xf32> -> tensor<4x4x4x4xf32>
  %barrier = util.optimization_barrier %pack : tensor<4x4x4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %barrier, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// CHECK-LABEL: func.func @pack_at_source
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SRC]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 4, 4, 4] : tensor<16x16xf32> into tensor<4x4x4x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXPANDED]] : tensor<4x4x4x4xf32>) outs(%[[DEST]] : tensor<4x4x4x4xf32>) permutation = [0, 2, 1, 3]
//       CHECK:   util.optimization_barrier %[[TRANSPOSED]]
//   CHECK-NOT:   linalg.pack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unpack_at_source() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  %barrier = util.optimization_barrier %unpack : tensor<16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %barrier, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// Unpack is not decomposed because the barrier blocks the source side transformation.
// CHECK-LABEL: func.func @unpack_at_source
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[SRC]] {{.+}} into %[[DEST]]
//       CHECK:   util.optimization_barrier %[[UNPACK]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @pack_at_dest() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x16xf32>> -> tensor<16x16xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %barrier = util.optimization_barrier %src : tensor<16x16xf32>
  %pack = linalg.pack %barrier inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<16x16xf32> -> tensor<4x4x4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// Pack is not decomposed because the barrier blocks the destination side transformation.
// CHECK-LABEL: func.func @pack_at_dest
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[BARRIER:.+]] = util.optimization_barrier %[[SRC]]
//       CHECK:   linalg.pack %[[BARRIER]] {{.+}} into %[[DEST]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unpack_at_dest() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %barrier = util.optimization_barrier %src : tensor<4x4x4x4xf32>
  %unpack = linalg.unpack %barrier inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// Unpack is decomposed to transpose + collapse_shape + copy.
// CHECK-LABEL: func.func @unpack_at_dest
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[BARRIER:.+]] = util.optimization_barrier %[[SRC]]
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x4x4x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[BARRIER]] : tensor<4x4x4x4xf32>) outs(%[[EMPTY]] : tensor<4x4x4x4xf32>) permutation = [0, 2, 1, 3]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSPOSED]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<4x4x4x4xf32> into tensor<16x16xf32>
//       CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[COLLAPSED]] : tensor<16x16xf32>) outs(%[[DEST]] : tensor<16x16xf32>)
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[COPY]]
//   CHECK-NOT:   linalg.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @padded_pack() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<15x15xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [15, 15], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<15x15xf32>> -> tensor<15x15xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %pack = linalg.pack %src padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<15x15xf32> -> tensor<4x4x4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// Pack is not decomposed because it has padding.
// CHECK-LABEL: func.func @padded_pack
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<15x15xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   linalg.pack %[[SRC]] padding_value({{.+}}) {{.+}} into %[[DEST]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @padded_unpack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<15x15xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [15, 15], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<15x15xf32>> -> tensor<15x15xf32>
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<15x15xf32>
  iree_tensor_ext.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [15, 15], strides = [1, 1] : tensor<15x15xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<15x15xf32>>
  return
}
// Unpack is not decomposed because it unpacks to a smaller tensor (15x15 from 4x4x4x4).
// CHECK-LABEL: func.func @padded_unpack
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<15x15xf32>
//       CHECK:   linalg.unpack %[[SRC]] {{.+}} into %[[DEST]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_pack() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_castui %0 : i32 to index
  %3 = arith.index_castui %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %2}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x4xf32>>{%3, %3}
  %src = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %2], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %2} -> tensor<?x?xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%3, %3, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x4xf32>>{%3, %3} -> tensor<?x?x4x4xf32>
  %pack = linalg.pack %src padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<?x?xf32> -> tensor<?x?x4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %5, offsets = [0, 0, 0, 0], sizes = [%3, %3, 4, 4], strides = [1, 1, 1, 1] : tensor<?x?x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?x4x4xf32>>{%3, %3}
  return
}
// Pack is not decomposed because it has dynamic shapes.
// CHECK-LABEL: func.func @dynamic_pack
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<?x?xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<?x?x4x4xf32>
//       CHECK:   linalg.pack %[[SRC]] padding_value({{.+}}) {{.+}} into %[[DEST]]

// -----

#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @dynamic_unpack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#pipeline_layout) ordinal(1) : i32
  %2 = arith.index_castui %0 : i32 to index
  %3 = arith.index_castui %1 : i32 to index
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x4xf32>>{%2, %2}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%3, %3}
  %src = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%2, %2, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?x4x4xf32>>{%2, %2} -> tensor<?x?x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%3, %3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%3, %3} -> tensor<?x?xf32>
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<?x?x4x4xf32> -> tensor<?x?xf32>
  iree_tensor_ext.dispatch.tensor.store %unpack, %5, offsets = [0, 0], sizes = [%3, %3], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x?xf32>>{%3, %3}
  return
}
// Unpack is not decomposed because it has dynamic shapes.
// CHECK-LABEL: func.func @dynamic_unpack
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<?x?x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<?x?xf32>
//       CHECK:   linalg.unpack %[[SRC]] {{.+}} into %[[DEST]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @load_non_full_slice() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<17x17xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<17x17xf32>> -> tensor<16x16xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %pack = linalg.pack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<16x16xf32> -> tensor<4x4x4x4xf32>
  iree_tensor_ext.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// Pack is decomposed because source is a non-full slice (16x16 from 17x17).
// CHECK-LABEL: func.func @load_non_full_slice
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[SRC]] {{\[}}[0, 1], [2, 3]{{\]}} output_shape [4, 4, 4, 4] : tensor<16x16xf32> into tensor<4x4x4x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[EXPANDED]] : tensor<4x4x4x4xf32>) outs(%[[DEST]] : tensor<4x4x4x4xf32>) permutation = [0, 2, 1, 3]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[TRANSPOSED]]
//   CHECK-NOT:   linalg.pack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @store_non_full_slice() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<17x17xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<17x17xf32>> -> tensor<16x16xf32>
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<17x17xf32>>
  return
}
// Unpack is decomposed because destination is a non-full slice (16x16 to 17x17).
// CHECK-LABEL: func.func @store_non_full_slice
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x4x4x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[SRC]] : tensor<4x4x4x4xf32>) outs(%[[EMPTY]] : tensor<4x4x4x4xf32>) permutation = [0, 2, 1, 3]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSPOSED]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<4x4x4x4xf32> into tensor<16x16xf32>
//       CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[COLLAPSED]] : tensor<16x16xf32>) outs(%[[DEST]] : tensor<16x16xf32>)
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[COPY]]
//   CHECK-NOT:   linalg.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @multi_use_unpack_fold() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  iree_tensor_ext.dispatch.tensor.store %unpack, %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// Unpack is decomposed because all uses are stores (multiple stores to same value is fine).
// CHECK-LABEL: func.func @multi_use_unpack_fold
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<4x4x4x4xf32>
//       CHECK:   %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[SRC]] : tensor<4x4x4x4xf32>) outs(%[[EMPTY]] : tensor<4x4x4x4xf32>) permutation = [0, 2, 1, 3]
//       CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[TRANSPOSED]] {{\[}}[0, 1], [2, 3]{{\]}} : tensor<4x4x4x4xf32> into tensor<16x16xf32>
//       CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[COLLAPSED]] : tensor<16x16xf32>) outs(%[[DEST]] : tensor<16x16xf32>)
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[COPY]], {{.+}}, {{.+}} : tensor<16x16xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[COPY]], {{.+}}, {{.+}} : tensor<16x16xf32>
//   CHECK-NOT:   linalg.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @multi_use_unpack_no_fold() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %dest2 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %unpack = linalg.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %copy = linalg.copy ins(%unpack : tensor<16x16xf32>) outs(%dest2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  iree_tensor_ext.dispatch.tensor.store %copy, %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// Unpack is not decomposed because it has a non-store use (linalg.copy).
// CHECK-LABEL: func.func @multi_use_unpack_no_fold
//       CHECK:   %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<4x4x4x4xf32>
//       CHECK:   %[[DEST:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[DEST2:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<16x16xf32>
//       CHECK:   %[[UNPACK:.+]] = linalg.unpack %[[SRC]] {{.+}} into %[[DEST]]
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[UNPACK]]
//       CHECK:   %[[COPY:.+]] = linalg.copy ins(%[[UNPACK]] : tensor<16x16xf32>) outs(%[[DEST2]] : tensor<16x16xf32>)
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[COPY]]
