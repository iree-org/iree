// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-pcf-writes)" --split-input-file | FileCheck %s

func.func @fuse_write_slice_with_parallel_insert(%init: tensor<32x64xf32>, %dest: !pcf.sref<32x64xf32, sync(#pcf.sequential)>) {
  %result = scf.forall (%i, %j) in (4, 8) shared_outs(%iter = %init) -> tensor<32x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %iter[%i, %j] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x64xf32>
    }
  }
  pcf.write_slice %result into %dest[0, 0] [32, 64] [1, 1] : tensor<32x64xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
  return
}

// CHECK-LABEL: @fuse_write_slice_with_parallel_insert(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: !pcf.sref<32x64xf32, sync(#pcf.sequential)>

//       CHECK:   scf.forall (%[[I:.+]], %[[J:.+]]) in (4, 8) {
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     pcf.write_slice %[[TILE]] into %[[DEST]][%[[I]], %[[J]]] [8, 8] [1, 1]
//       CHECK:   }
//       CHECK-NOT:   pcf.write_slice
//       CHECK:   return

// -----

func.func @fuse_with_offset(%init: tensor<32x64xf32>, %dest: !pcf.sref<32x64xf32, sync(#pcf.sequential)>) {
  %result = scf.forall (%i, %j) in (4, 8) shared_outs(%iter = %init) -> tensor<32x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %iter[%i, %j] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x64xf32>
    }
  }
  %c16 = arith.constant 16 : index
  pcf.write_slice %result into %dest[%c16, 0] [32, 64] [1, 1] : tensor<32x64xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
  return
}

// CHECK-LABEL: @fuse_with_offset(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: !pcf.sref<32x64xf32, sync(#pcf.sequential)>

//   CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
//       CHECK:   scf.forall (%[[I:.+]], %[[J:.+]]) in (4, 8) {
//       CHECK:     %[[TILE:.+]] = tensor.generate
//       CHECK:     %[[COMPOSED_OFFSET:.+]] = arith.addi %[[I]], %[[C16]]
//       CHECK:     pcf.write_slice %[[TILE]] into %[[DEST]][%[[COMPOSED_OFFSET]], %[[J]]] [8, 8] [1, 1]
//       CHECK:   }
//       CHECK-NOT:   pcf.write_slice
//       CHECK:   return

// -----

func.func @fuse_with_stride(%init: tensor<32x64xf32>, %dest: !pcf.sref<64x128xf32, sync(#pcf.sequential)>) {
  %result = scf.forall (%i, %j) in (4, 8) shared_outs(%iter = %init) -> tensor<32x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %iter[%i, %j] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x64xf32>
    }
  }
  pcf.write_slice %result into %dest[0, 0] [32, 64] [2, 2] : tensor<32x64xf32> into !pcf.sref<64x128xf32, sync(#pcf.sequential)>
  return
}

// CHECK-LABEL: @fuse_with_stride(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: !pcf.sref<64x128xf32, sync(#pcf.sequential)>

//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK:   scf.forall (%[[I:.+]], %[[J:.+]]) in (4, 8) {
//       CHECK:     %[[TILE:.+]] = tensor.generate
//   CHECK-DAG:     %[[OFFSET_0:.+]] = arith.muli %[[I]], %[[C2]]
//   CHECK-DAG:     %[[OFFSET_1:.+]] = arith.muli %[[J]], %[[C2]]
//       CHECK:     pcf.write_slice %[[TILE]] into %[[DEST]][%[[OFFSET_0]], %[[OFFSET_1]]] [8, 8] [2, 2]
//       CHECK:   }
//       CHECK-NOT:   pcf.write_slice
//       CHECK:   return

// -----

func.func @no_fusion_different_source(%init: tensor<32x64xf32>, %other: tensor<32x64xf32>, %dest: !pcf.sref<32x64xf32, sync(#pcf.sequential)>) {
  %result = scf.forall (%i, %j) in (4, 8) shared_outs(%iter = %init) -> tensor<32x64xf32> {
    %c0 = arith.constant 0.0 : f32
    %tile = tensor.generate {
    ^bb0(%ii: index, %jj: index):
      tensor.yield %c0 : f32
    } : tensor<8x8xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tile into %iter[%i, %j] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<32x64xf32>
    }
  }
  // Write from a different source, not the forall result
  pcf.write_slice %other into %dest[0, 0] [32, 64] [1, 1] : tensor<32x64xf32> into !pcf.sref<32x64xf32, sync(#pcf.sequential)>
  return
}

// CHECK-LABEL: @no_fusion_different_source(
//  CHECK-SAME:   %[[INIT:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[OTHER:[A-Za-z0-9_]+]]: tensor<32x64xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: !pcf.sref<32x64xf32, sync(#pcf.sequential)>

//       CHECK:   pcf.write_slice %[[OTHER]] into %[[DEST]][0, 0] [32, 64] [1, 1]
//       CHECK:   return
