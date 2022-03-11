// RUN: iree-opt -iree-codegen-bufferize-copy-only-dispatches -split-input-file %s | FileCheck %s

builtin.module {
  func @tensor_insert_slice() {
    %source_size_y = hal.interface.constant.load[0] : index
    %source_size_x = hal.interface.constant.load[1] : index
    %dest_size_y = hal.interface.constant.load[2] : index
    %dest_size_x = hal.interface.constant.load[3] : index
    %dest_offset_y = hal.interface.constant.load[4] : index
    %dest_offset_x = hal.interface.constant.load[5] : index
    %insert_offset_y = hal.interface.constant.load[6] : index
    %insert_offset_x = hal.interface.constant.load[7] : index
    %source = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
        : !flow.dispatch.tensor<readonly:?x?xi32>{%source_size_y, %source_size_x}
    %dest = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
        : !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x}
    %source_load = flow.dispatch.tensor.load %source, offsets = [0, 0], sizes = [%source_size_y, %source_size_x], strides = [1, 1]
        : !flow.dispatch.tensor<readonly:?x?xi32>{%source_size_y, %source_size_x} -> tensor<?x?xi32>
    %dest_load = flow.dispatch.tensor.load %dest, offsets = [0, 0], sizes = [%dest_size_y, %dest_size_x], strides = [1, 1]
        : !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x} -> tensor<?x?xi32>
    %insert = tensor.insert_slice %source_load into %dest_load[%insert_offset_y, %insert_offset_x] [%source_size_y, %source_size_x] [1, 1] : tensor<?x?xi32> into tensor<?x?xi32>
    flow.dispatch.tensor.store %insert, %dest, offsets = [%dest_offset_y, %dest_offset_x], sizes = [%source_size_y, %source_size_x], strides = [1, 1]
        : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x}
    return
  }
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//      CHECK: func @tensor_insert_slice()
//  CHECK-DAG:   %[[DEST_OFFSET_Y:.+]] = hal.interface.constant.load[4]
//  CHECK-DAG:   %[[DEST_OFFSET_X:.+]] = hal.interface.constant.load[5]
//  CHECK-DAG:   %[[INSERT_OFFSET_Y:.+]] = hal.interface.constant.load[6]
//  CHECK-DAG:   %[[INSERT_OFFSET_X:.+]] = hal.interface.constant.load[7]
//  CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[OFFSET_Y:.+]] = affine.apply #[[MAP0]]()[%[[INSERT_OFFSET_Y]], %[[DEST_OFFSET_Y]]]
//  CHECK-DAG:   %[[OFFSET_X:.+]] = affine.apply #[[MAP0]]()[%[[INSERT_OFFSET_X]], %[[DEST_OFFSET_X]]]
//  CHECK-DAG:   %[[SUBVIEW:.+]] = memref.subview %[[DEST]][%[[OFFSET_Y]], %[[OFFSET_X]]]
//      CHECK:   linalg.generic
// CHECK-SAME:       ins(%[[SOURCE]] :
// CHECK-SAME:       outs(%[[SUBVIEW]] :

// -----

builtin.module {
  func @tensor_extract_slice() {
    %source_size_y = hal.interface.constant.load[0] : index
    %source_size_x = hal.interface.constant.load[1] : index
    %dest_size_y = hal.interface.constant.load[2] : index
    %dest_size_x = hal.interface.constant.load[3] : index
    %dest_offset_y = hal.interface.constant.load[4] : index
    %dest_offset_x = hal.interface.constant.load[5] : index
    %extract_offset_y = hal.interface.constant.load[6] : index
    %extract_offset_x = hal.interface.constant.load[7] : index
    %slice_size_y = hal.interface.constant.load[8] : index
    %slice_size_x = hal.interface.constant.load[9] : index
    %source = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
        : !flow.dispatch.tensor<readonly:?x?xi32>{%source_size_y, %source_size_x}
    %dest = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
        : !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x}
    %source_load = flow.dispatch.tensor.load %source, offsets = [0, 0], sizes = [%source_size_y, %source_size_x], strides = [1, 1]
        : !flow.dispatch.tensor<readonly:?x?xi32>{%source_size_y, %source_size_x} -> tensor<?x?xi32>
    %extract = tensor.extract_slice %source_load[%extract_offset_y, %extract_offset_x] [%slice_size_y, %slice_size_x] [1, 1] : tensor<?x?xi32> to tensor<?x?xi32>
    flow.dispatch.tensor.store %extract, %dest, offsets = [%dest_offset_y, %dest_offset_x], sizes = [%slice_size_y, %slice_size_x], strides = [1, 1]
        : tensor<?x?xi32> -> !flow.dispatch.tensor<readwrite:?x?xi32>{%dest_size_y, %dest_size_x}
    return
  }
}
//      CHECK: func @tensor_extract_slice()
//  CHECK-DAG:   %[[DEST_OFFSET_Y:.+]] = hal.interface.constant.load[4]
//  CHECK-DAG:   %[[DEST_OFFSET_X:.+]] = hal.interface.constant.load[5]
//  CHECK-DAG:   %[[EXTRACT_OFFSET_Y:.+]] = hal.interface.constant.load[6]
//  CHECK-DAG:   %[[EXTRACT_OFFSET_X:.+]] = hal.interface.constant.load[7]
//  CHECK-DAG:   %[[SLICE_SIZE_Y:.+]] = hal.interface.constant.load[8]
//  CHECK-DAG:   %[[SLICE_SIZE_X:.+]] = hal.interface.constant.load[9]
//  CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(0)
//  CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
//  CHECK-DAG:   %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][%[[EXTRACT_OFFSET_Y]], %[[EXTRACT_OFFSET_X]]] [%[[SLICE_SIZE_Y:.+]], %[[SLICE_SIZE_X]]]
//  CHECK-DAG:   %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][%[[DEST_OFFSET_Y]], %[[DEST_OFFSET_X]]] [%[[SLICE_SIZE_Y]], %[[SLICE_SIZE_X]]]
//      CHECK:   linalg.generic
// CHECK-SAME:       ins(%[[SOURCE_SUBVIEW]] :
// CHECK-SAME:       outs(%[[DEST_SUBVIEW]] :

// -----

builtin.module {
  func @UpSampling1D() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readwrite:2x16x3xf32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset(%c0) alignment(64) : !flow.dispatch.tensor<readonly:2x8x3xf32>
    %2 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [2, 8, 3], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:2x8x3xf32> -> tensor<2x8x3xf32>
    %3 = tensor.extract_slice %2[0, 0, 0] [2, 1, 3] [1, 1, 1] : tensor<2x8x3xf32> to tensor<2x3xf32>
    %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [2, 16, 3], strides = [1, 1, 1] : !flow.dispatch.tensor<readwrite:2x16x3xf32> -> tensor<2x16x3xf32>
    %5 = tensor.insert_slice %3 into %4[0, 0, 0] [2, 1, 3] [1, 1, 1] : tensor<2x3xf32> into tensor<2x16x3xf32>
    flow.dispatch.tensor.store %5, %0, offsets = [0, 0, 0], sizes = [2, 16, 3], strides = [1, 1, 1] : tensor<2x16x3xf32> -> !flow.dispatch.tensor<readwrite:2x16x3xf32>
    return
  }
}
// CHECK-LABEL: func @UpSampling1D()
//   CHECK-DAG:   %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(0)
//   CHECK-DAG:   %[[SOURCE:.+]] = hal.interface.binding.subspan set(0) binding(1)
//   CHECK-DAG:   %[[SOURCE_SUBVIEW:.+]] = memref.subview %[[SOURCE]][0, 0, 0] [2, 1, 3]
//   CHECK-DAG:   %[[DEST_SUBVIEW:.+]] = memref.subview %[[DEST]][0, 0, 0] [2, 1, 3]
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%[[SOURCE_SUBVIEW]] : memref<2x3xf32, #{{[a-zA-Z0-9]+}}>)
//  CHECK-SAME:       outs(%[[DEST_SUBVIEW]] : memref<2x3xf32, #{{[a-zA-Z0-9]+}}>)
