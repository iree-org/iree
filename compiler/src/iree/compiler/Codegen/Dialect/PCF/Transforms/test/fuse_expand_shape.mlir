// RUN: iree-opt %s --pass-pipeline="builtin.module(iree-pcf-fuse-consumers)" --split-input-file | FileCheck %s

// Test: Fuse tensor.expand_shape into pcf.generic with tied init.
// The 1D result is expanded to 2D, and the write_slice offsets/sizes are
// de-linearized accordingly. The constant source is folded to the expanded
// shape by canonicalization.

func.func @fuse_expand_shape_into_generic(%arg0: tensor<80xi32>) -> tensor<8x10xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
        -> (tensor<80xi32>) {
    %cst = arith.constant dense<5> : tensor<40xi32>
    pcf.write_slice %cst into %ref[%id0] [40] [1] : tensor<40xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1 : tensor<8x10xi32>
}

// CHECK-LABEL: @fuse_expand_shape_into_generic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<80xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x10xi32>
//       CHECK:  %[[INIT:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<8x10xi32>)
//       CHECK:    %[[EXP_OFF:.+]] = affine.apply
//       CHECK:    pcf.write_slice %[[CST]] into %[[REF]][%[[EXP_OFF]], 0] [4, 10] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Test: Fuse tensor.expand_shape into pcf.loop.

func.func @fuse_expand_shape_into_loop(%arg0: tensor<80xi32>, %n0: index, %n1: index) -> tensor<8x10xi32> {
  %0 = pcf.loop scope(#pcf.test_scope) count(%n0, %n1)
    execute(%ref = %arg0)[%id0: index, %id1: index]
            : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
           -> (tensor<80xi32>) {
    %cst = arith.constant dense<5> : tensor<40xi32>
    pcf.write_slice %cst into %ref[%id0] [40] [1] : tensor<40xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1 : tensor<8x10xi32>
}

// CHECK-LABEL: @fuse_expand_shape_into_loop
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<80xi32>

//       CHECK:  %[[CST:.+]] = arith.constant dense<5> : tensor<4x10xi32>
//       CHECK:  %[[INIT:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
//       CHECK:  %[[LOOP:.+]] = pcf.loop scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<8x10xi32>)
//       CHECK:    %[[EXP_OFF:.+]] = affine.apply
//       CHECK:    pcf.write_slice %[[CST]] into %[[REF]][%[[EXP_OFF]], 0] [4, 10] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[LOOP]]

// -----

// Test: Fuse tensor.expand_shape with multiple write_slices.

func.func @fuse_expand_shape_multiple_write_slices(%arg0: tensor<80xi32>) -> tensor<8x10xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
        -> (tensor<80xi32>) {
    %cst1 = arith.constant dense<5> : tensor<30xi32>
    %cst2 = arith.constant dense<7> : tensor<50xi32>
    pcf.write_slice %cst1 into %ref[%id0] [30] [1] : tensor<30xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.write_slice %cst2 into %ref[%id1] [50] [1] : tensor<50xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1 : tensor<8x10xi32>
}

// CHECK-LABEL: @fuse_expand_shape_multiple_write_slices
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<80xi32>

//   CHECK-DAG:  %[[CST1:.+]] = arith.constant dense<5> : tensor<3x10xi32>
//   CHECK-DAG:  %[[CST2:.+]] = arith.constant dense<7> : tensor<5x10xi32>
//       CHECK:  %[[INIT:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}}
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])
//       CHECK:    -> (tensor<8x10xi32>)
//       CHECK:    pcf.write_slice %[[CST1]] into %[[REF]]{{.*}} [3, 10] [1, 1]
//       CHECK:    pcf.write_slice %[[CST2]] into %[[REF]]{{.*}} [5, 10] [1, 1]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Negative test: producer result has multiple uses (expand_shape + direct use).

func.func @no_fuse_expand_shape_multiple_uses(%arg0: tensor<80xi32>) -> (tensor<8x10xi32>, tensor<80xi32>) {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
        -> (tensor<80xi32>) {
    %cst = arith.constant dense<5> : tensor<40xi32>
    pcf.write_slice %cst into %ref[%id0] [40] [1] : tensor<40xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1, %0 : tensor<8x10xi32>, tensor<80xi32>
}

// CHECK-LABEL: @no_fuse_expand_shape_multiple_uses

//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//       CHECK:  %[[EXPAND:.+]] = tensor.expand_shape %[[GENERIC]] {{\[}}[0, 1]{{\]}}
//       CHECK:  return %[[EXPAND]], %[[GENERIC]]

// -----

// Test: Fuse expand_shape with statically unaligned size (25 % 10 != 0).
// The sub-tile loop iterates over expanded rows, writing chunks.

func.func @fuse_expand_shape_unaligned_size(%arg0: tensor<80xi32>) -> tensor<8x10xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
        -> (tensor<80xi32>) {
    %cst = arith.constant dense<5> : tensor<25xi32>
    pcf.write_slice %cst into %ref[%id0] [25] [1] : tensor<25xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1 : tensor<8x10xi32>
}

// CHECK-LABEL: @fuse_expand_shape_unaligned_size
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<80xi32>

//   CHECK-DAG:  %[[CST:.+]] = arith.constant dense<5> : tensor<25xi32>
//       CHECK:  %[[INIT:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [8, 10]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<8x10xi32>)
//       CHECK:    scf.for %[[ROW:[A-Za-z0-9_]+]] =
//       CHECK:      %[[SUB:.+]] = tensor.extract_slice %[[CST]]
//       CHECK:      %[[EXP:.+]] = tensor.expand_shape %[[SUB]] {{\[}}[0, 1]{{\]}}
//       CHECK:      pcf.write_slice %[[EXP]] into %[[REF]][%[[ROW]]
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Test: Fuse expand_shape with dynamic offset and size. An scf.if is
// generated to branch between the aligned (simple) and unaligned (loop) paths.

func.func @fuse_expand_shape_dynamic(%arg0: tensor<80xi32>, %off: index, %sz: index) -> tensor<8x10xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
        -> (tensor<80xi32>) {
    %empty = tensor.empty(%sz) : tensor<?xi32>
    pcf.write_slice %empty into %ref[%off] [%sz] [1] : tensor<?xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1 : tensor<8x10xi32>
}

// CHECK-LABEL: @fuse_expand_shape_dynamic
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<80xi32>
//  CHECK-SAME:   %[[OFF:[A-Za-z0-9_]+]]: index
//  CHECK-SAME:   %[[SZ:[A-Za-z0-9_]+]]: index

//       CHECK:  %[[INIT:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [8, 10]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])[%[[ID0:[A-Za-z0-9_]+]]: index
//       CHECK:    -> (tensor<8x10xi32>)

//   Alignment check: off % 10 == 0 && sz % 10 == 0.
//       CHECK:    %[[OFF_REM:.+]] = arith.remui %[[OFF]]
//       CHECK:    %[[OFF_EQ:.+]] = arith.cmpi eq, %[[OFF_REM]]
//       CHECK:    %[[SZ_REM:.+]] = arith.remui %[[SZ]]
//       CHECK:    %[[SZ_EQ:.+]] = arith.cmpi eq, %[[SZ_REM]]
//       CHECK:    %[[ALIGNED:.+]] = arith.andi %[[OFF_EQ]], %[[SZ_EQ]]
//       CHECK:    scf.if %[[ALIGNED]] {

//   Aligned path: single expanded write_slice.
//       CHECK:      tensor.expand_shape
//       CHECK:      pcf.write_slice {{.*}} into %[[REF]]

//       CHECK:    } else {

//   Unaligned path: sub-tile loop.
//       CHECK:      scf.for %[[ROW:[A-Za-z0-9_]+]] =
//       CHECK:        tensor.extract_slice
//       CHECK:        tensor.expand_shape
//       CHECK:        pcf.write_slice {{.*}} into %[[REF]][%[[ROW]]
//       CHECK:    }
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]

// -----

// Test: Fuse expand_shape with dynamic offset but aligned static size.
// Since size is 40 (40 % 10 == 0) but offset is dynamic, an scf.if is
// still needed because offset alignment is unknown at compile time.

func.func @fuse_expand_shape_dynamic_offset_static_size(%arg0: tensor<80xi32>, %off: index) -> tensor<8x10xi32> {
  %0 = pcf.generic scope(#pcf.test_scope)
    execute(%ref = %arg0)[%id0: index, %id1: index, %n0: index, %n1: index]
         : (!pcf.sref<80xi32, sync(#pcf.test_scope)>)
        -> (tensor<80xi32>) {
    %cst = arith.constant dense<5> : tensor<40xi32>
    pcf.write_slice %cst into %ref[%off] [40] [1] : tensor<40xi32> into !pcf.sref<80xi32, sync(#pcf.test_scope)>
    pcf.return
  }
  %1 = tensor.expand_shape %0 [[0, 1]] output_shape [8, 10] : tensor<80xi32> into tensor<8x10xi32>
  return %1 : tensor<8x10xi32>
}

// CHECK-LABEL: @fuse_expand_shape_dynamic_offset_static_size
//  CHECK-SAME:   %[[ARG0:[A-Za-z0-9_]+]]: tensor<80xi32>
//  CHECK-SAME:   %[[OFF:[A-Za-z0-9_]+]]: index

//       CHECK:  %[[INIT:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]{{\]}} output_shape [8, 10]
//       CHECK:  %[[GENERIC:.+]] = pcf.generic scope(#pcf.test_scope)
//  CHECK-NEXT:    execute(%[[REF:.+]] = %[[INIT]])
//       CHECK:    -> (tensor<8x10xi32>)

//   Only offset alignment is checked (size 40 % 10 == 0 is known at compile time).
//       CHECK:    %[[OFF_REM:.+]] = arith.remui %[[OFF]]
//       CHECK:    %[[ALIGNED:.+]] = arith.cmpi eq, %[[OFF_REM]]
//       CHECK:    scf.if %[[ALIGNED]]
//       CHECK:      pcf.write_slice {{.*}} into %[[REF]]
//       CHECK:    } else {
//       CHECK:      scf.for
//       CHECK:        pcf.write_slice {{.*}} into %[[REF]]
//       CHECK:    }
//       CHECK:    pcf.return
//       CHECK:  return %[[GENERIC]]
