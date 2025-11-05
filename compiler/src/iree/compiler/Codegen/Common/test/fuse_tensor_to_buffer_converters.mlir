// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-codegen-fuse-tensor-to-buffer-converters)" \
// RUN:  --split-input-file | FileCheck %s

func.func @fuse_store_to_buffer_loop(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %n: index) -> (tensor<64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %tensor0 = tensor.empty() : tensor<64xf32>
  %tensor1 = tensor.empty() : tensor<64xf32>

  %result:2 = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%r0 = %tensor0, %r1 = %tensor1)[%id: index]
        : (!pcf.sref<64xf32, sync(#pcf.sequential)>,
           !pcf.sref<64xf32, sync(#pcf.sequential)>)
        -> (tensor<64xf32>, tensor<64xf32>) {
    %slice0 = arith.constant dense<1.0> : tensor<16xf32>
    pcf.write_slice %slice0 into %r0[%c0][16][1] : tensor<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    %slice1_tensor = arith.constant dense<2.0> : tensor<16xf32>
    pcf.write_slice %slice1_tensor into %r1[%c0][16][1] : tensor<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    %slice1_vector = arith.constant dense<3.0> : vector<16xf32>
    pcf.write_slice %slice1_vector into %r1[16][16][1] : vector<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    %alloc = memref.alloc() : memref<16xf32>
    %slice1_memref = memref.cast %alloc : memref<16xf32> to memref<16xf32>
    pcf.write_slice %slice1_memref into %r1[32][16][1] : memref<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    pcf.return
  }

  iree_codegen.store_to_buffer %result#0, %arg0 : tensor<64xf32> into memref<64xf32>
  iree_codegen.store_to_buffer %result#1, %arg1 : tensor<64xf32> into memref<64xf32>
  return %result#0 : tensor<64xf32>
}

// CHECK-LABEL: @fuse_store_to_buffer_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: memref<64xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: memref<64xf32>
//   CHECK-DAG:   %[[CST_VEC:.+]] = arith.constant dense<3.0{{.*}}> : vector<16xf32>
//   CHECK-DAG:   %[[CST_TENSOR1:.+]] = arith.constant dense<2.0{{.*}}> : tensor<16xf32>
//   CHECK-DAG:   %[[CST_TENSOR0:.+]] = arith.constant dense<1.0{{.*}}> : tensor<16xf32>
//       CHECK:   pcf.loop
//       CHECK:     %[[SUBVIEW0:.+]] = memref.subview %[[ARG0]][%c0] [16] [1]
//       CHECK:     iree_codegen.store_to_buffer %[[CST_TENSOR0]], %[[SUBVIEW0]]
//       CHECK:     pcf.write_slice %[[CST_TENSOR0]] into
//       CHECK:     %[[SUBVIEW1:.+]] = memref.subview %[[ARG1]][%c0] [16] [1]
//       CHECK:     iree_codegen.store_to_buffer %[[CST_TENSOR1]], %[[SUBVIEW1]]
//       CHECK:     %[[SUBVIEW2:.+]] = memref.subview %[[ARG1]][16] [16] [1]
//       CHECK:     vector.transfer_write %[[CST_VEC]], %[[SUBVIEW2]][%c0]
//       CHECK:     %[[ALLOC:.+]] = memref.alloc()
//       CHECK:     %[[SUBVIEW3:.+]] = memref.subview %[[ARG1]][32] [16] [1]
//       CHECK:     memref.copy %[[ALLOC]], %[[SUBVIEW3]]
//   CHECK-NOT:   iree_codegen.store_to_buffer

// -----

func.func @fuse_store_to_buffer_generic(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %n: index) -> (tensor<64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %tensor0 = tensor.empty() : tensor<64xf32>
  %tensor1 = tensor.empty() : tensor<64xf32>

  %result:2 = pcf.generic scope(#pcf.sequential)
    execute(%r0 = %tensor0, %r1 = %tensor1)[%id: index, %count: index]
        : (!pcf.sref<64xf32, sync(#pcf.sequential)>,
           !pcf.sref<64xf32, sync(#pcf.sequential)>)
        -> (tensor<64xf32>, tensor<64xf32>) {
    %slice0 = arith.constant dense<1.0> : tensor<16xf32>
    pcf.write_slice %slice0 into %r0[%c0][16][1] : tensor<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    %slice1_tensor = arith.constant dense<2.0> : tensor<16xf32>
    pcf.write_slice %slice1_tensor into %r1[%c0][16][1] : tensor<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    %slice1_vector = arith.constant dense<3.0> : vector<16xf32>
    pcf.write_slice %slice1_vector into %r1[16][16][1] : vector<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    %alloc = memref.alloc() : memref<16xf32>
    %slice1_memref = memref.cast %alloc : memref<16xf32> to memref<16xf32>
    pcf.write_slice %slice1_memref into %r1[32][16][1] : memref<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>

    pcf.return
  }

  iree_codegen.store_to_buffer %result#0, %arg0 : tensor<64xf32> into memref<64xf32>
  iree_codegen.store_to_buffer %result#1, %arg1 : tensor<64xf32> into memref<64xf32>
  return %result#0 : tensor<64xf32>
}

// CHECK-LABEL: @fuse_store_to_buffer_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: memref<64xf32>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: memref<64xf32>
//   CHECK-DAG:   %[[CST_VEC:.+]] = arith.constant dense<3.0{{.*}}> : vector<16xf32>
//   CHECK-DAG:   %[[CST_TENSOR1:.+]] = arith.constant dense<2.0{{.*}}> : tensor<16xf32>
//   CHECK-DAG:   %[[CST_TENSOR0:.+]] = arith.constant dense<1.0{{.*}}> : tensor<16xf32>
//       CHECK:   pcf.generic
//       CHECK:     %[[SUBVIEW0:.+]] = memref.subview %[[ARG0]][%c0] [16] [1]
//       CHECK:     iree_codegen.store_to_buffer %[[CST_TENSOR0]], %[[SUBVIEW0]]
//       CHECK:     pcf.write_slice %[[CST_TENSOR0]] into
//       CHECK:     %[[SUBVIEW1:.+]] = memref.subview %[[ARG1]][%c0] [16] [1]
//       CHECK:     iree_codegen.store_to_buffer %[[CST_TENSOR1]], %[[SUBVIEW1]]
//       CHECK:     %[[SUBVIEW2:.+]] = memref.subview %[[ARG1]][16] [16] [1]
//       CHECK:     vector.transfer_write %[[CST_VEC]], %[[SUBVIEW2]][%c0]
//       CHECK:     %[[ALLOC:.+]] = memref.alloc()
//       CHECK:     %[[SUBVIEW3:.+]] = memref.subview %[[ARG1]][32] [16] [1]
//       CHECK:     memref.copy %[[ALLOC]], %[[SUBVIEW3]]
//   CHECK-NOT:   iree_codegen.store_to_buffer

// -----

// Test pcf.loop with one tensor result stored with iree_tensor_ext.dispatch.tensor.store.
// Single pcf.write_slice with tensor operand type.
func.func @fuse_dispatch_tensor_store_loop(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>, %n: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %tensor = tensor.empty() : tensor<64xf32>

  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%r = %tensor)[%id: index]
        : (!pcf.sref<64xf32, sync(#pcf.sequential)>)
       -> (tensor<64xf32>) {
    %slice = arith.constant dense<1.0> : tensor<16xf32>
    pcf.write_slice %slice into %r[%c0][16][1] : tensor<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>
    pcf.return
  }

  iree_tensor_ext.dispatch.tensor.store %result, %arg0, offsets = [0], sizes = [64], strides = [1]
    : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
  return
}

// CHECK-LABEL: @fuse_dispatch_tensor_store_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<1.0{{.*}}> : tensor<16xf32>
//       CHECK:   pcf.loop
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[CST]], %[[ARG0]], offsets = [0], sizes = [16], strides = [1]
//   CHECK-NOT:   iree_tensor_ext.dispatch.tensor.store

// -----

func.func @fuse_dispatch_tensor_store_generic(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>, %n: index) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %tensor = tensor.empty() : tensor<64xf32>

  %result = pcf.generic scope(#pcf.sequential)
    execute(%r = %tensor)[%id: index, %count: index]
        : (!pcf.sref<64xf32, sync(#pcf.sequential)>)
       -> (tensor<64xf32>) {
    %slice = arith.constant dense<1.0> : tensor<16xf32>
    pcf.write_slice %slice into %r[%c0][16][1] : tensor<16xf32> into !pcf.sref<64xf32, sync(#pcf.sequential)>
    pcf.return
  }

  iree_tensor_ext.dispatch.tensor.store %result, %arg0, offsets = [0], sizes = [64], strides = [1]
    : tensor<64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
  return
}

// CHECK-LABEL: @fuse_dispatch_tensor_store_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64xf32>>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<1.0{{.*}}> : tensor<16xf32>
//       CHECK:   pcf.generic
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[CST]], %[[ARG0]], offsets = [0], sizes = [16], strides = [1]
//   CHECK-NOT:   iree_tensor_ext.dispatch.tensor.store
