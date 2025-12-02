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

// -----

// Test folding tensor.expand_shape into iree_codegen.store_to_buffer.
func.func @fuse_expand_shape_into_buffer_store(%arg0: memref<1x64xf32>) {
  %cst = arith.constant dense<1.0> : tensor<64xf32>
  %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
  iree_codegen.store_to_buffer %expanded, %arg0 : tensor<1x64xf32> into memref<1x64xf32>
  return
}

// CHECK-LABEL: @fuse_expand_shape_into_buffer_store
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: memref<1x64xf32>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<1.0{{.*}}> : tensor<64xf32>
//       CHECK:   %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG0]] {{\[\[}}0, 1{{\]\]}} : memref<1x64xf32> into memref<64xf32>
//       CHECK:   iree_codegen.store_to_buffer %[[CST]], %[[COLLAPSED]] : tensor<64xf32> into memref<64xf32>
//   CHECK-NOT:   tensor.expand_shape

// -----

// Test folding tensor.expand_shape into iree_codegen.store_to_buffer with strided layout.
func.func @fuse_expand_shape_into_buffer_store_strided(%arg0: memref<1x64xf32, strided<[128, 1], offset: 64>>) {
  %cst = arith.constant dense<1.0> : tensor<64xf32>
  %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
  iree_codegen.store_to_buffer %expanded, %arg0 : tensor<1x64xf32> into memref<1x64xf32, strided<[128, 1], offset: 64>>
  return
}

// CHECK-LABEL: @fuse_expand_shape_into_buffer_store_strided
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: memref<1x64xf32, strided<[128, 1], offset: 64>>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<1.0{{.*}}> : tensor<64xf32>
//       CHECK:   %[[COLLAPSED:.+]] = memref.collapse_shape %[[ARG0]] {{\[\[}}0, 1{{\]\]}} : memref<1x64xf32, strided<[128, 1], offset: 64>> into memref<64xf32, strided<[1], offset: 64>>
//       CHECK:   iree_codegen.store_to_buffer %[[CST]], %[[COLLAPSED]] : tensor<64xf32> into memref<64xf32, strided<[1], offset: 64>>
//   CHECK-NOT:   tensor.expand_shape

// -----

// Test folding tensor.expand_shape into iree_tensor_ext.dispatch.tensor.store.
func.func @fuse_expand_shape_into_dispatch_tensor_store(%arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xf32>>) {
  %cst = arith.constant dense<1.0> : tensor<64xf32>
  %expanded = tensor.expand_shape %cst [[0, 1]] output_shape [1, 64] : tensor<64xf32> into tensor<1x64xf32>
  iree_tensor_ext.dispatch.tensor.store %expanded, %arg0, offsets = [0, 0], sizes = [1, 64], strides = [1, 1]
    : tensor<1x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xf32>>
  return
}

// CHECK-LABEL: @fuse_expand_shape_into_dispatch_tensor_store
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x64xf32>>
//   CHECK-DAG:   %[[CST:.+]] = arith.constant dense<1.0{{.*}}> : tensor<64xf32>
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[CST]], %[[ARG0]], offsets = [0, 0], sizes = [1, 64], strides = [1, 1]
//   CHECK-NOT:   tensor.expand_shape

// -----

// Test folding tensor.insert_slice of rank-0 tensor into unit-dim tensor
// into iree_tensor_ext.dispatch.tensor.store.
func.func @fuse_unit_insert_slice_into_dispatch_tensor_store(
    %arg0: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>,
    %arg1: tensor<f32>,
    %idx: index) {
  %empty = tensor.empty() : tensor<1xf32>
  %inserted = tensor.insert_slice %arg1 into %empty[0] [1] [1]
    : tensor<f32> into tensor<1xf32>
  iree_tensor_ext.dispatch.tensor.store %inserted, %arg0, offsets = [%idx], sizes = [1], strides = [1]
    : tensor<1xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
  return
}

// CHECK-LABEL: @fuse_unit_insert_slice_into_dispatch_tensor_store
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<512xf32>>
//  CHECK-SAME:   %[[ARG1:[A-Za-z0-9_]+]]: tensor<f32>
//  CHECK-SAME:   %[[IDX:[A-Za-z0-9_]+]]: index
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[ARG1]], %[[ARG0]], offsets = [%[[IDX]]], sizes = [1], strides = [1]
//   CHECK-NOT:   tensor.insert_slice
//   CHECK-NOT:   tensor.empty

// -----

// Test composing nested tensor.extract_slice ops inside a PCF loop.
func.func @compose_extract_slice_inside_pcf_loop(%arg0: tensor<64x64xf32>, %n: index) -> tensor<64x64xf32> {
  %tensor = tensor.empty() : tensor<64x64xf32>

  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%r = %tensor)[%id: index]
        : (!pcf.sref<64x64xf32, sync(#pcf.sequential)>)
       -> (tensor<64x64xf32>) {
    // First extract_slice: extract [id, 0] [16, 64] from input
    %slice0 = tensor.extract_slice %arg0[%id, 0] [16, 64] [1, 1]
      : tensor<64x64xf32> to tensor<16x64xf32>
    // Second extract_slice: extract [0, 0] [16, 16] from first slice
    %slice1 = tensor.extract_slice %slice0[0, 0] [16, 16] [1, 1]
      : tensor<16x64xf32> to tensor<16x16xf32>
    pcf.write_slice %slice1 into %r[0, 0][16, 16][1, 1]
      : tensor<16x16xf32> into !pcf.sref<64x64xf32, sync(#pcf.sequential)>
    pcf.return
  }

  return %result : tensor<64x64xf32>
}

// CHECK-LABEL: @compose_extract_slice_inside_pcf_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<64x64xf32>
//       CHECK:   pcf.loop
//       CHECK:     %[[SLICE:.+]] = tensor.extract_slice %[[ARG0]][%{{.*}}, 0] [16, 16] [1, 1]
//       CHECK:     pcf.write_slice %[[SLICE]]
//   CHECK-NOT:   tensor.extract_slice

// -----

// Test that nested extract_slice ops are NOT composed outside a PCF loop.
func.func @no_compose_extract_slice_outside_pcf_loop(%arg0: tensor<64x64xf32>, %idx: index) -> tensor<16x16xf32> {
  // First extract_slice
  %slice0 = tensor.extract_slice %arg0[%idx, 0] [16, 64] [1, 1]
    : tensor<64x64xf32> to tensor<16x64xf32>
  // Second extract_slice
  %slice1 = tensor.extract_slice %slice0[0, 0] [16, 16] [1, 1]
    : tensor<16x64xf32> to tensor<16x16xf32>
  return %slice1 : tensor<16x16xf32>
}

// CHECK-LABEL: @no_compose_extract_slice_outside_pcf_loop
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice

// -----

// Test composing tensor.extract_slice with dispatch.tensor.load inside a PCF loop.
func.func @compose_extract_slice_of_dispatch_load_inside_pcf_loop(
    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>>,
    %arg1: memref<64x64xf32>,
    %n: index) {
  %tensor = tensor.empty() : tensor<64x64xf32>

  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%r = %tensor)[%id: index]
        : (!pcf.sref<64x64xf32, sync(#pcf.sequential)>)
       -> (tensor<64x64xf32>) {
    // Load full tensor from dispatch tensor
    %loaded = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [%id, 0], sizes = [16, 64], strides = [1, 1]
      : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<16x64xf32>
    // Extract smaller slice from loaded tensor
    %slice = tensor.extract_slice %loaded[0, 0] [16, 16] [1, 1]
      : tensor<16x64xf32> to tensor<16x16xf32>
    pcf.write_slice %slice into %r[0, 0][16, 16][1, 1]
      : tensor<16x16xf32> into !pcf.sref<64x64xf32, sync(#pcf.sequential)>
    pcf.return
  }

  iree_codegen.store_to_buffer %result, %arg1 : tensor<64x64xf32> into memref<64x64xf32>
  return
}

// CHECK-LABEL: @compose_extract_slice_of_dispatch_load_inside_pcf_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>>
//       CHECK:   pcf.loop
//       CHECK:     %[[LOAD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[ARG0]], offsets = [%{{.*}}, 0], sizes = [16, 16], strides = [1, 1]
//  CHECK-SAME:       -> tensor<16x16xf32>
//       CHECK:     iree_codegen.store_to_buffer %[[LOAD]]
//   CHECK-NOT:   tensor.extract_slice

// -----

// Test that extract_slice of dispatch.tensor.load is NOT composed outside a PCF loop.
func.func @no_compose_extract_slice_of_dispatch_load_outside_pcf_loop(
    %arg0: !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>>,
    %idx: index) -> tensor<16x16xf32> {
  // Load from dispatch tensor
  %loaded = iree_tensor_ext.dispatch.tensor.load %arg0, offsets = [%idx, 0], sizes = [16, 64], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<64x64xf32>> -> tensor<16x64xf32>
  // Extract slice from loaded tensor
  %slice = tensor.extract_slice %loaded[0, 0] [16, 16] [1, 1]
    : tensor<16x64xf32> to tensor<16x16xf32>
  return %slice : tensor<16x16xf32>
}

// CHECK-LABEL: @no_compose_extract_slice_of_dispatch_load_outside_pcf_loop
//       CHECK:   iree_tensor_ext.dispatch.tensor.load
//       CHECK:   tensor.extract_slice
