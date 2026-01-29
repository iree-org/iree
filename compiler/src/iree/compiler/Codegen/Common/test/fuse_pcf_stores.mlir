// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-codegen-fuse-pcf-stores)" --split-input-file | FileCheck %s

// Test fusing store_to_buffer with pcf.loop producing tensor result.
// After fusion, the loop no longer produces a result (ref arg dropped).
func.func @fuse_store_to_buffer_tensor(%init: tensor<32x64xf32>, %dest: memref<32x64xf32>, %n: index) {
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_codegen.store_to_buffer %result, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @fuse_store_to_buffer_tensor(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: memref<32x64xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9_]+]]: index

//       CHECK:   pcf.loop scope(#pcf.sequential) count(%[[N]])
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[DEST]][%[[I]], 0] [8, 8] [1, 1]
//       CHECK:     iree_codegen.store_to_buffer %[[TILE]], %[[SUBVIEW]]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_codegen.store_to_buffer
//       CHECK:   return

// -----

// Test fusing store_to_buffer with vector source in write_slice.
// The vector constant may be hoisted outside the loop.
func.func @fuse_store_to_buffer_vector(%init: tensor<32x64xf32>, %dest: memref<32x64xf32>, %n: index) {
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant dense<0.0> : vector<8x8xf32>
    pcf.write_slice %c0 into %ref[%i, 0] [8, 8] [1, 1] : vector<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_codegen.store_to_buffer %result, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @fuse_store_to_buffer_vector(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: memref<32x64xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9_]+]]: index

//   CHECK-DAG:   %[[VEC:.+]] = arith.constant dense<0.000000e+00> : vector<8x8xf32>
//       CHECK:   pcf.loop scope(#pcf.sequential) count(%[[N]])
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[DEST]][%[[I]], 0] [8, 8] [1, 1]
//       CHECK:     vector.transfer_write %[[VEC]], %[[SUBVIEW]]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_codegen.store_to_buffer
//       CHECK:   return

// -----

// Test fusing with pcf.generic instead of pcf.loop.
func.func @fuse_store_to_buffer_generic(%init: tensor<32x64xf32>, %dest: memref<32x64xf32>) {
  %result = pcf.generic scope(#pcf.sequential)
    execute(%ref = %init)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%id0, %id1] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_codegen.store_to_buffer %result, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @fuse_store_to_buffer_generic(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: memref<32x64xf32>

//       CHECK:   pcf.generic scope(#pcf.sequential)
//       CHECK:     execute[
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[DEST]]
//       CHECK:     iree_codegen.store_to_buffer %[[TILE]], %[[SUBVIEW]]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_codegen.store_to_buffer
//       CHECK:   return

// -----

// Test with memref source in write_slice - should use memref.copy.
func.func @fuse_store_to_buffer_memref_source(%init: tensor<32x64xf32>, %dest: memref<32x64xf32>, %src: memref<8x8xf32>, %n: index) {
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    pcf.write_slice %src into %ref[%i, 0] [8, 8] [1, 1] : memref<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_codegen.store_to_buffer %result, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @fuse_store_to_buffer_memref_source(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: memref<32x64xf32>
//  CHECK-SAME:   %[[SRC:[A-Za-z0-9_]+]]: memref<8x8xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9_]+]]: index

//       CHECK:   pcf.loop scope(#pcf.sequential) count(%[[N]])
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
//       CHECK:     %[[SUBVIEW:.+]] = memref.subview %[[DEST]][%[[I]], 0] [8, 8] [1, 1]
//       CHECK:     memref.copy %[[SRC]], %[[SUBVIEW]]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_codegen.store_to_buffer
//       CHECK:   return

// -----

// Negative test: non return-only-sync scope should not fuse.
func.func @no_fuse_non_return_only_sync(%init: tensor<32x64xf32>, %dest: memref<32x64xf32>, %n: index) {
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, #pcf.sequential>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, #pcf.sequential>
    pcf.return
  }
  iree_codegen.store_to_buffer %result, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @no_fuse_non_return_only_sync(
//       CHECK:   %[[RESULT:.+]] = pcf.loop
//       CHECK:   iree_codegen.store_to_buffer %[[RESULT]]
//       CHECK:   return

// -----

// Negative test: buffer does not dominate the producer loop.
func.func @no_fuse_buffer_not_dominating(%init: tensor<32x64xf32>, %n: index) {
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  // Buffer is allocated after the loop, so it cannot dominate.
  %alloc = memref.alloc() : memref<32x64xf32>
  iree_codegen.store_to_buffer %result, %alloc : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @no_fuse_buffer_not_dominating(
//       CHECK:   %[[RESULT:.+]] = pcf.loop
//       CHECK:   %[[ALLOC:.+]] = memref.alloc
//       CHECK:   iree_codegen.store_to_buffer %[[RESULT]], %[[ALLOC]]
//       CHECK:   return

// -----

// Negative test: tensor not from pcf.loop or pcf.generic.
func.func @no_fuse_tensor_not_from_pcf(%tensor: tensor<32x64xf32>, %dest: memref<32x64xf32>) {
  iree_codegen.store_to_buffer %tensor, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @no_fuse_tensor_not_from_pcf(
//       CHECK:   iree_codegen.store_to_buffer
//       CHECK:   return

// -----

// Test fusing with multiple write_slices in the same loop body.
func.func @fuse_store_to_buffer_multiple_writes(%init: tensor<32x64xf32>, %dest: memref<32x64xf32>, %n: index) {
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile1 = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile1 into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    %c1 = arith.constant 1.0 : f32
    %tile2 = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c1 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile2 into %ref[%i, 8] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_codegen.store_to_buffer %result, %dest : tensor<32x64xf32> into memref<32x64xf32>
  return
}

// CHECK-LABEL: @fuse_store_to_buffer_multiple_writes(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: memref<32x64xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9_]+]]: index

//       CHECK:   pcf.loop scope(#pcf.sequential) count(%[[N]])
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
//       CHECK:     tensor.generate
//       CHECK:     memref.subview %[[DEST]][%[[I]], 0] [8, 8] [1, 1]
//       CHECK:     iree_codegen.store_to_buffer
//       CHECK:     tensor.generate
//       CHECK:     memref.subview %[[DEST]][%[[I]], 8] [8, 8] [1, 1]
//       CHECK:     iree_codegen.store_to_buffer
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_codegen.store_to_buffer
//       CHECK:   return

// -----

// =============================================================================
// FuseDispatchTensorStore pattern tests
// =============================================================================

// Test fusing dispatch_tensor_store with pcf.loop producing tensor result.
// After fusion, the loop no longer produces a result and stores directly to
// the dispatch tensor binding.

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @fuse_dispatch_tensor_store_basic(%init: tensor<32x64xf32>, %n: index) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0, 0], sizes = [32, 64], strides = [1, 1]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  return
}

// CHECK-LABEL: @fuse_dispatch_tensor_store_basic(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9_]+]]: index

//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   pcf.loop scope(#pcf.sequential) count(%[[N]])
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[BINDING]]
//  CHECK-SAME:       offsets = [%[[I]], 0], sizes = [8, 8], strides = [1, 1]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_tensor_ext.dispatch.tensor.store
//       CHECK:   return

// -----

// Test that offsets from write_slice are added to dispatch_tensor_store offsets.
// The original store has offsets [4, 8], and write_slice has offsets [%i, 0].
// The fused store should have offsets [%i + 4, 8].

#pipeline_layout2 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @fuse_dispatch_tensor_store_offset_composition(%init: tensor<32x64xf32>, %n: index) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout2) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  // Store at offset [4, 8] into a larger tensor.
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [4, 8], sizes = [32, 64], strides = [1, 1]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
  return
}

// CHECK-LABEL: @fuse_dispatch_tensor_store_offset_composition(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[N:[A-Za-z0-9_]+]]: index

//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   pcf.loop scope(#pcf.sequential) count(%[[N]])
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     %[[OFFSET:.+]] = affine.apply
//  CHECK-SAME:       %[[I]]
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[BINDING]]
//  CHECK-SAME:       offsets = [%[[OFFSET]], 8], sizes = [8, 8], strides = [1, 1]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_tensor_ext.dispatch.tensor.store
//       CHECK:   return

// -----

// Test fusing dispatch_tensor_store with pcf.generic.

#pipeline_layout3 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @fuse_dispatch_tensor_store_generic(%init: tensor<32x64xf32>) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout3) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  %result = pcf.generic scope(#pcf.sequential)
    execute(%ref = %init)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%id0, %id1] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0, 0], sizes = [32, 64], strides = [1, 1]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  return
}

// CHECK-LABEL: @fuse_dispatch_tensor_store_generic(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>

//       CHECK:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   pcf.generic scope(#pcf.sequential)
//       CHECK:     execute[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[TILE]], %[[BINDING]]
//  CHECK-SAME:       offsets = [%[[ID0]], %[[ID1]]], sizes = [8, 8], strides = [1, 1]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_tensor_ext.dispatch.tensor.store
//       CHECK:   return

// -----

// Test rank-reducing insert_slice optimization: when write_slice source is a
// rank-reducing insert_slice into tensor.empty, we use the insert_slice source
// directly instead of the padded tensor.

#pipeline_layout_rank_reduce = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @fuse_dispatch_tensor_store_rank_reducing_insert(%init: tensor<32x64xf32>, %n: index) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout_rank_reduce) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    // Create a 1D vector (rank 1)
    %vec = arith.constant dense<0.0> : tensor<8xf32>
    // Create a 2D empty tensor (rank 2)
    %empty = tensor.empty() : tensor<8x1xf32>
    // Rank-reducing insert: 1D tensor<8xf32> into 2D tensor<8x1xf32>
    %padded = tensor.insert_slice %vec into %empty[0, 0] [8, 1] [1, 1]
        : tensor<8xf32> into tensor<8x1xf32>
    // Write the padded 2D tensor
    pcf.write_slice %padded into %ref[%i, 0] [8, 1] [1, 1]
        : tensor<8x1xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0, 0], sizes = [32, 64], strides = [1, 1]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  return
}

// CHECK-LABEL: @fuse_dispatch_tensor_store_rank_reducing_insert(
// The constant gets hoisted outside the loop:
//   CHECK-DAG:   %[[VEC:.+]] = arith.constant dense<0.000000e+00> : tensor<8xf32>
//   CHECK-DAG:   %[[BINDING:.+]] = hal.interface.binding.subspan
//       CHECK:   pcf.loop
//       CHECK:     execute[%[[I:[A-Za-z0-9_]+]]: index] {
// The optimization should use the 1D tensor directly, not the 2D padded version:
//       CHECK:     iree_tensor_ext.dispatch.tensor.store %[[VEC]], %[[BINDING]]
//  CHECK-SAME:       sizes = [8, 1]
//       CHECK:     pcf.return
//   CHECK-NOT:   iree_tensor_ext.dispatch.tensor.store
//       CHECK:   return

// -----

// Negative test: non-unit stride in dispatch_tensor_store should not fuse.

#pipeline_layout4 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @no_fuse_dispatch_tensor_store_non_unit_stride(%init: tensor<32x64xf32>, %n: index) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout4) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    pcf.write_slice %tile into %ref[%i, 0] [8, 8] [1, 1] : tensor<8x8xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  // Non-unit stride - should not fuse.
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0, 0], sizes = [32, 64], strides = [2, 2]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<64x128xf32>>
  return
}

// CHECK-LABEL: @no_fuse_dispatch_tensor_store_non_unit_stride(
//       CHECK:   %[[RESULT:.+]] = pcf.loop
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[RESULT]]
//  CHECK-SAME:     strides = [2, 2]
//       CHECK:   return

// -----

// Negative test: tensor not from pcf.loop or pcf.generic should not fuse.

#pipeline_layout5 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @no_fuse_dispatch_tensor_store_non_pcf_source(%tensor: tensor<32x64xf32>) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout5) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  iree_tensor_ext.dispatch.tensor.store %tensor, %binding, offsets = [0, 0], sizes = [32, 64], strides = [1, 1]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  return
}

// CHECK-LABEL: @no_fuse_dispatch_tensor_store_non_pcf_source(
//       CHECK:   iree_tensor_ext.dispatch.tensor.store
//       CHECK:   return

// -----

// Negative test: non-unit stride in write_slice should not fuse.

#pipeline_layout6 = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

func.func @no_fuse_dispatch_tensor_store_write_slice_non_unit_stride(%init: tensor<32x64xf32>, %n: index) {
  %binding = hal.interface.binding.subspan layout(#pipeline_layout6) binding(0)
      : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  %result = pcf.loop scope(#pcf.sequential) count(%n)
    execute(%ref = %init)[%i: index]
         : (!pcf.sref<32x64xf32, sync(#pcf.sequential)>)
        -> (tensor<32x64xf32>) {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<4x4xf32>
    // Non-unit stride in write_slice - should not fuse.
    pcf.write_slice %tile into %ref[%i, 0] [4, 4] [2, 2]
        : tensor<4x4xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
    pcf.return
  }
  iree_tensor_ext.dispatch.tensor.store %result, %binding, offsets = [0, 0], sizes = [32, 64], strides = [1, 1]
      : tensor<32x64xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x64xf32>>
  return
}

// CHECK-LABEL: @no_fuse_dispatch_tensor_store_write_slice_non_unit_stride(
//       CHECK:   %[[RESULT:.+]] = pcf.loop
//       CHECK:   iree_tensor_ext.dispatch.tensor.store %[[RESULT]]
//       CHECK:   return
