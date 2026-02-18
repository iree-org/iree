// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-producers)" --split-input-file | FileCheck %s

// Positive Tests:
//* - linalg.fill producer into pcf.generic
//* - linalg.fill producer into pcf.loop
//* - linalg.transpose producer into pcf.generic
//* - Producer with multiple read sites
//* - Producer erased when no other uses
//* - Producer kept when other uses exist
//* - Fusion skips first init (no producer) and fuses second init
//* - Producer fusion with vector read_slice

// Basic: fuse a linalg.fill producer into a pcf.generic's init.
func.func @fuse_fill_into_generic(%arg0: tensor<8x16xf32>, %dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x16xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EXTRACT]] : tensor<4x8xf32>)
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TILED_FILL]]{{.*}} outs(%[[TILED_FILL]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Basic: fuse a linalg.fill producer into a pcf.loop's init.
func.func @fuse_fill_into_loop(%dest: tensor<8x16xf32>, %n0: index, %n1: index) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.loop scope(#pcf.test_scope) count(%n0, %n1)
    execute(%ref = %fill)[%id0: index, %id1: index]
            : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
           -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_into_loop
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[EXTRACT:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EXTRACT]] : tensor<4x8xf32>)
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TILED_FILL]]{{.*}} outs(%[[TILED_FILL]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Fuse a linalg.transpose producer. Verifies that the tiled transpose extracts
// from the input with permuted offsets and sizes.
func.func @fuse_transpose_into_generic(%input: tensor<16x8xf32>, %dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %empty = tensor.empty() : tensor<8x16xf32>
  %transpose = linalg.transpose ins(%input : tensor<16x8xf32>) outs(%empty : tensor<8x16xf32>) permutation = [1, 0]
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %transpose)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// The transpose has permutation [1,0], so a result tile at [id0, id1] of size
// [4, 8] maps to an input tile at [id1, id0] of size [8, 4].
//
// CHECK-LABEL: @fuse_transpose_into_generic
//  CHECK-SAME:   %[[INPUT:[A-Za-z0-9_]+]]: tensor<16x8xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[EMPTY:.+]] = tensor.empty() : tensor<8x16xf32>
//   CHECK-NOT:  linalg.transpose
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[EMPTY]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[INPUT]][%[[ID1]], %[[ID0]]] [8, 4] [1, 1] : tensor<16x8xf32> to tensor<8x4xf32>
//       CHECK:    %[[OUT_SLICE:.+]] = tensor.extract_slice %[[EMPTY]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1] : tensor<8x16xf32> to tensor<4x8xf32>
//       CHECK:    %[[TRANSPOSED:.+]] = linalg.transpose ins(%[[INPUT_SLICE]] : tensor<8x4xf32>) outs(%[[OUT_SLICE]] : tensor<4x8xf32>) permutation = [1, 0]
//       CHECK:    %[[EXP:.+]] = linalg.exp ins(%[[TRANSPOSED]]{{.*}} outs(%[[TRANSPOSED]]
//       CHECK:    pcf.write_slice %[[EXP]] into %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Multiple read_slice sites for the same init value.
func.func @fuse_fill_multiple_reads(%dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice0 = pcf.read_slice %ref[%id0, 0] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result0 = linalg.exp ins(%slice0 : tensor<4x8xf32>) outs(%slice0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result0 into %ref[%id0, 0] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    %slice1 = pcf.read_slice %ref[%id0, 8] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result1 = linalg.exp ins(%slice1 : tensor<4x8xf32>) outs(%slice1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result1 into %ref[%id0, 8] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_multiple_reads
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[EXT0:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], 0] [4, 8] [1, 1]
//       CHECK:    %[[FILL0:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EXT0]]
//       CHECK:    %[[EXP0:.+]] = linalg.exp ins(%[[FILL0]]
//       CHECK:    pcf.write_slice %[[EXP0]] into %[[REF]][%[[ID0]], 0] [4, 8] [1, 1]
//       CHECK:    %[[EXT1:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], 8] [4, 8] [1, 1]
//       CHECK:    %[[FILL1:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EXT1]]
//       CHECK:    %[[EXP1:.+]] = linalg.exp ins(%[[FILL1]]
//       CHECK:    pcf.write_slice %[[EXP1]] into %[[REF]][%[[ID0]], 8] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Producer kept when it has other uses.
func.func @keep_producer_with_other_uses(%dest: tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %slice = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %result = linalg.exp ins(%slice : tensor<4x8xf32>) outs(%slice : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %result into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0, %fill : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: @keep_producer_with_other_uses
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK:  %[[FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[DEST]]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[EXT:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EXT]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]], %[[FILL]]

// -----

// First init has no producer (block arg), second init has a fill producer.
// Verify that the second init is fused while the first is left unchanged.
func.func @fuse_second_init_only(%arg0: tensor<8x16xf32>, %dest: tensor<4x8xf32>) -> (tensor<8x16xf32>, tensor<4x8xf32>) {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<4x8xf32>) -> tensor<4x8xf32>
  %0:2 = pcf.generic scope(#pcf.test_scope)
    execute(%ref0 = %arg0, %ref1 = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>, !pcf.sref<4x8xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>, tensor<4x8xf32>) {
    %slice0 = pcf.read_slice %ref0[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %slice1 = pcf.read_slice %ref1[0, 0] [4, 8] [1, 1] : !pcf.sref<4x8xf32, sync(#pcf.test_scope)> to tensor<4x8xf32>
    %add = linalg.add ins(%slice0, %slice1 : tensor<4x8xf32>, tensor<4x8xf32>) outs(%slice1 : tensor<4x8xf32>) -> tensor<4x8xf32>
    pcf.write_slice %add into %ref0[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0#0, %0#1 : tensor<8x16xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: @fuse_second_init_only
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<8x16xf32>
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<4x8xf32>

//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]]:2 = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF0:.+]] = %[[ARG0]], %[[REF1:.+]] = %[[DEST]])
//       CHECK:    pcf.read_slice %[[REF0]][%{{.+}}, %{{.+}}] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[DEST]] : tensor<4x8xf32>)
//       CHECK:    linalg.add
//       CHECK:    pcf.write_slice
//       CHECK:    pcf.return

// -----

// Producer fusion with vector read_slice. The tiled producer result (tensor) is
// converted to a vector via vector.transfer_read.
func.func @fuse_fill_vector_read(%dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %vec = pcf.read_slice %ref[%id0, %id1] [4, 8] [1, 1] : !pcf.sref<8x16xf32, sync(#pcf.test_scope)> to vector<4x8xf32>
    pcf.write_slice %vec into %ref[%id0, %id1] [4, 8] [1, 1] : vector<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @fuse_fill_vector_read
//  CHECK-SAME:   %[[DEST:[A-Za-z0-9_]+]]: tensor<8x16xf32>

//       CHECK:  %[[POISON:.+]] = ub.poison : f32
//       CHECK:  %[[C0:.+]] = arith.constant 0 : index
//       CHECK:  %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//   CHECK-NOT:  linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[DEST]])[%[[ID0:[A-Za-z0-9_]+]]: index, %[[ID1:[A-Za-z0-9_]+]]: index
//       CHECK:    %[[EXT:.+]] = tensor.extract_slice %[[DEST]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    %[[TILED_FILL:.+]] = linalg.fill ins(%[[CST]]{{.*}} outs(%[[EXT]]
//       CHECK:    %[[VEC:.+]] = vector.transfer_read %[[TILED_FILL]][%[[C0]], %[[C0]]], %[[POISON]] {in_bounds = [true, true]}
//       CHECK:    pcf.write_slice %[[VEC]] into %[[REF]][%[[ID0]], %[[ID1]]] [4, 8] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Negative Tests:
//  - No read_slice on the sref

// Negative: no read_slice on the sref (only writes).
func.func @no_fuse_no_reads(%dest: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%dest : tensor<8x16xf32>) -> tensor<8x16xf32>
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %fill)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<8x16xf32, sync(#pcf.test_scope)>)
        -> (tensor<8x16xf32>) {
    %cst2 = arith.constant dense<1.0> : tensor<4x8xf32>
    pcf.write_slice %cst2 into %ref[%id0, %id1] [4, 8] [1, 1] : tensor<4x8xf32> into !pcf.sref<8x16xf32, sync(#pcf.test_scope)>
    pcf.return
  }
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: @no_fuse_no_reads
//       CHECK:  %[[FILL:.+]] = linalg.fill
//       CHECK:  %[[GENERIC:.+]] = pcf.generic
//  CHECK-NEXT:    execute(%{{.+}} = %[[FILL]])
//       CHECK:  return %[[GENERIC]]
