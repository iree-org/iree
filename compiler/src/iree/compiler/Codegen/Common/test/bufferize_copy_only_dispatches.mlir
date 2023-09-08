// RUN: iree-opt --iree-codegen-bufferize-copy-only-dispatches --split-input-file %s | FileCheck %s

builtin.module {
  func.func @tensor_insert_slice() {
    %slice_size = hal.interface.constant.load[0] : index
    %dest_offset_y = hal.interface.constant.load[1] : index
    %dest_offset_x = hal.interface.constant.load[2] : index
    %dest_stride_y = hal.interface.constant.load[3] : index
    %dest_stride_x = hal.interface.constant.load[4] : index
    %source_offset_y = hal.interface.constant.load[5] : index
    %source_offset_x = hal.interface.constant.load[6] : index
    %source_stride_y = hal.interface.constant.load[7] : index
    %source_stride_x = hal.interface.constant.load[8] : index
    %dest_binding_size_y = hal.interface.constant.load[9] : index
    %dest_binding_size_x = hal.interface.constant.load[10] : index
    %source_binding_size_y = hal.interface.constant.load[11] : index
    %source_binding_size_x = hal.interface.constant.load[12] : index
    %source = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
        : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%source_binding_size_y, %source_binding_size_x}
    %dest = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
        : !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%dest_binding_size_y, %dest_binding_size_x}
    %source_load = flow.dispatch.tensor.load %source, offsets = [%source_offset_y, %source_offset_x],
        sizes = [1, %slice_size], strides = [%source_stride_y, %source_stride_x]
        : !flow.dispatch.tensor<readonly:tensor<?x?xi32>>{%source_binding_size_y, %source_binding_size_x} -> tensor<?xi32>
    flow.dispatch.tensor.store %source_load, %dest, offsets = [%dest_offset_y, %dest_offset_x],
        sizes = [%slice_size, 1], strides = [%dest_stride_y, %dest_stride_x]
        : tensor<?xi32> -> !flow.dispatch.tensor<readwrite:tensor<?x?xi32>>{%dest_binding_size_y, %dest_binding_size_x}
    return
  }
}
//      CHECK: func.func @tensor_insert_slice()
//  CHECK-DAG:   %[[SLICE_SIZE:.+]] = hal.interface.constant.load[0]
//  CHECK-DAG:   %[[DEST_OFFSET_Y:.+]] = hal.interface.constant.load[1]
//  CHECK-DAG:   %[[DEST_OFFSET_X:.+]] = hal.interface.constant.load[2]
//  CHECK-DAG:   %[[DEST_STRIDE_Y:.+]] = hal.interface.constant.load[3]
//  CHECK-DAG:   %[[DEST_STRIDE_X:.+]] = hal.interface.constant.load[4]
//  CHECK-DAG:   %[[SOURCE_OFFSET_Y:.+]] = hal.interface.constant.load[5]
//  CHECK-DAG:   %[[SOURCE_OFFSET_X:.+]] = hal.interface.constant.load[6]
//  CHECK-DAG:   %[[SOURCE_STRIDE_Y:.+]] = hal.interface.constant.load[7]
//  CHECK-DAG:   %[[SOURCE_STRIDE_X:.+]] = hal.interface.constant.load[8]
//  CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[SOURCE_OFFSET_Y]], %[[SOURCE_OFFSET_X]]] [1, %[[SLICE_SIZE]]] [%[[SOURCE_STRIDE_Y]], %[[SOURCE_STRIDE_X]]]
//  CHECK-DAG:   %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]]] [%[[SLICE_SIZE]], 1] [%[[DEST_STRIDE_Y]], %[[DEST_STRIDE_X]]]
//      CHECK:   linalg.generic
// CHECK-SAME:       ins(%[[SOURCE_SUBVIEW]] :
// CHECK-SAME:       outs(%[[DEST_SUBVIEW]] :

// -----

builtin.module {
  func.func @UpSampling1D() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x16x3xf32>>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(uniform_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2x8x3xf32>>
    %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, 1, 3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<2x8x3xf32>> -> tensor<2x3xf32>
    flow.dispatch.tensor.store %2, %0, offsets = [0, 0, 0], sizes = [2, 1, 3], strides = [1, 1, 1] : tensor<2x3xf32> -> !flow.dispatch.tensor<readwrite:tensor<2x16x3xf32>>
    return
  }
}
// CHECK-LABEL: func.func @UpSampling1D()
//   CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][0, 0, 0] [2, 1, 3]
//   CHECK-DAG:   %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][0, 0, 0] [2, 1, 3]
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[SOURCE_SUBVIEW]] : memref<2x3xf32, strided<[24, 1]>, #hal.descriptor_type<uniform_buffer>>)
//  CHECK-SAME:       outs(%[[DEST_SUBVIEW]] : memref<2x3xf32, strided<[48, 1]>, #hal.descriptor_type<storage_buffer>>)

// -----

builtin.module {
  func.func @concatenate_cst() {
    %cst = arith.constant dense<0> : tensor<2x3xi32>
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readwrite:tensor<2x5xi32>>
    flow.dispatch.tensor.store %cst, %0, offsets = [0, 2], sizes = [2, 3], strides = [1, 1] : tensor<2x3xi32> -> !flow.dispatch.tensor<readwrite:tensor<2x5xi32>>
    return
  }
}
// CHECK-LABEL: func.func @concatenate_cst()
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<0> : tensor<2x3xi32>
//   CHECK-DAG:   %[[ZERO:.+]] = bufferization.to_memref %[[CST]] : memref<2x3xi32
//   CHECK-DAG:   %[[DEST_BINDING:.+]] = hal.interface.binding.subspan
//   CHECK-DAG:   %[[SUBVIEW:.+]] = memref.subview %[[DEST_BINDING]][0, 2] [2, 3]
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[ZERO]] :
//  CHECK-SAME:       outs(%[[SUBVIEW]] :

// -----

module {
  func.func @already_bufferized() {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<1001xf32, #hal.descriptor_type<storage_buffer>>
    memref.assume_alignment %0, 64 : memref<1001xf32, #hal.descriptor_type<storage_buffer>>
    %alloc = memref.alloc() : memref<1001xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<1001xf32>)
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["reduction"]} ins(%alloc : memref<1001xf32>) outs(%0 : memref<1001xf32, #hal.descriptor_type<storage_buffer>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    memref.dealloc %alloc : memref<1001xf32>
    return
  }
}

// CHECK-LABEL: func.func @already_bufferized
//       CHECK: memref.alloc
