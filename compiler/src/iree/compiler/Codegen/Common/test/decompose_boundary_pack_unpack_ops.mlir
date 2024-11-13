// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-decompose-boundary-pack-unpack-ops))" --split-input-file %s | FileCheck %s -check-prefixes=CHECK

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @pack_at_source() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x16xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16xf32>> -> tensor<16x16xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %pack = tensor.pack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<16x16xf32> -> tensor<4x4x4x4xf32>
  %barrier = util.optimization_barrier %pack : tensor<4x4x4x4xf32>
  flow.dispatch.tensor.store %barrier, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// CHECK-LABEL: func.func @pack_at_source
// CHECK-NOT:     tensor.pack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unpack_at_source() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %unpack = tensor.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  %barrier = util.optimization_barrier %unpack : tensor<16x16xf32>
  flow.dispatch.tensor.store %barrier, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// CHECK-LABEL: func.func @unpack_at_source
// CHECK:         tensor.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @pack_at_dest() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<16x16xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x16xf32>> -> tensor<16x16xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %barrier = util.optimization_barrier %src : tensor<16x16xf32>
  %pack = tensor.pack %barrier inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<16x16xf32> -> tensor<4x4x4x4xf32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// CHECK-LABEL: func.func @pack_at_dest
// CHECK:         tensor.pack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @unpack_at_dest() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %barrier = util.optimization_barrier %src : tensor<4x4x4x4xf32>
  %unpack = tensor.unpack %barrier inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// CHECK-LABEL: func.func @unpack_at_dest
// CHECK-NOT:     tensor.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @padded_pack() {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<15x15xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [15, 15], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<15x15xf32>> -> tensor<15x15xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %pack = tensor.pack %src padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<15x15xf32> -> tensor<4x4x4x4xf32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// CHECK-LABEL: func.func @padded_pack
// CHECK:         tensor.pack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @padded_unpack() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<15x15xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [15, 15], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<15x15xf32>> -> tensor<15x15xf32>
  %unpack = tensor.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<15x15xf32>
  flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [15, 15], strides = [1, 1] : tensor<15x15xf32> -> !flow.dispatch.tensor<readwrite:tensor<15x15xf32>>
  return
}
// CHECK-LABEL: func.func @padded_unpack
// CHECK:         tensor.unpack

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
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %2}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?x?x4x4xf32>>{%3, %3}
  %src = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %2], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%2, %2} -> tensor<?x?xf32>
  %dest = flow.dispatch.tensor.load %5, offsets = [0, 0, 0, 0], sizes = [%3, %3, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?x4x4xf32>>{%3, %3} -> tensor<?x?x4x4xf32>
  %pack = tensor.pack %src padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<?x?xf32> -> tensor<?x?x4x4xf32>
  flow.dispatch.tensor.store %pack, %5, offsets = [0, 0, 0, 0], sizes = [%3, %3, 4, 4], strides = [1, 1, 1, 1] : tensor<?x?x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?x4x4xf32>>{%3, %3}
  return
}
// CHECK-LABEL: func.func @dynamic_pack
// CHECK:         tensor.pack

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
  %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<?x?x4x4xf32>>{%2, %2}
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%3, %3}
  %src = flow.dispatch.tensor.load %4, offsets = [0, 0, 0, 0], sizes = [%2, %2, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?x4x4xf32>>{%2, %2} -> tensor<?x?x4x4xf32>
  %dest = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%3, %3], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%3, %3} -> tensor<?x?xf32>
  %unpack = tensor.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<?x?x4x4xf32> -> tensor<?x?xf32>
  flow.dispatch.tensor.store %unpack, %5, offsets = [0, 0], sizes = [%3, %3], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%3, %3}
  return
}
// CHECK-LABEL: func.func @dynamic_unpack
// CHECK:         tensor.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @load_non_full_slice() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<17x17xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<17x17xf32>> -> tensor<16x16xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %pack = tensor.pack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<16x16xf32> -> tensor<4x4x4x4xf32>
  flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : tensor<4x4x4x4xf32> -> !flow.dispatch.tensor<readwrite:tensor<4x4x4x4xf32>>
  return
}
// CHECK-LABEL: func.func @load_non_full_slice
// CHECK-NOT:     tensor.pack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @store_non_full_slice() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<17x17xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<17x17xf32>> -> tensor<16x16xf32>
  %unpack = tensor.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<17x17xf32>>
  return
}
// CHECK-LABEL: func.func @store_non_full_slice
// CHECK-NOT:     tensor.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @multi_use_unpack_fold() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %unpack = tensor.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  flow.dispatch.tensor.store %unpack, %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// CHECK-LABEL: func.func @multi_use_unpack_fold
// CHECK-NOT:     tensor.unpack

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
func.func @multi_use_unpack_no_fold() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %src = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 4, 4, 4], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4x4x4xf32>> -> tensor<4x4x4x4xf32>
  %dest = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %dest2 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : !flow.dispatch.tensor<readwrite:tensor<16x16xf32>> -> tensor<16x16xf32>
  %unpack = tensor.unpack %src inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %dest : tensor<4x4x4x4xf32> -> tensor<16x16xf32>
  flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  %copy = linalg.copy ins(%unpack : tensor<16x16xf32>) outs(%dest2 : tensor<16x16xf32>) -> tensor<16x16xf32>
  flow.dispatch.tensor.store %copy, %2, offsets = [0, 0], sizes = [16, 16], strides = [1, 1] : tensor<16x16xf32> -> !flow.dispatch.tensor<readwrite:tensor<16x16xf32>>
  return
}
// CHECK-LABEL: func.func @multi_use_unpack_no_fold
// CHECK:         tensor.unpack
