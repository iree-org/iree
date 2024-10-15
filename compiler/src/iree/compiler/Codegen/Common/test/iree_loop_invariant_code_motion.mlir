// RUN: iree-opt %s  -split-input-file --iree-loop-invariant-code-motion | FileCheck %s

func.func @nested_loops_code_invariant_to_both() {
  %m = memref.alloc() : memref<10xf32>
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }
  return
}

// CHECK-LABEL: @nested_loops_code_invariant_to_both
//       CHECK:   memref.alloc() : memref<10xf32>
//  CHECK-NEXT:   arith.constant 7
//  CHECK-NEXT:   arith.constant 8
//  CHECK-NEXT:   arith.addf

// -----

func.func @do_not_hoist_with_unknown_trip_count(%lb: index, %ub: index) {
  affine.for %arg1 = %lb to %ub {
    affine.for %arg0 = 0 to 10 {
    }
  }
  return
}

// CHECK-LABEL: @do_not_hoist_with_unknown_trip_count
//  CHECK-NEXT:   affine.for
//  CHECK-NEXT:     affine.for
//  CHECK-NEXT:     }
//  CHECK-NEXT:   }

// -----

func.func @do_not_hoist_scf_for_with_unknown_trip_count(%lb: index, %ub: index) {
  %c1 = arith.constant 1 : index
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  scf.for %arg0 = %lb to %ub step %c1 {
    %v0 = arith.addf %cf7, %cf8 : f32
  }
  return
}

// CHECK-LABEL: @do_not_hoist_scf_for_with_unknown_trip_count
//       CHECK:   scf.for
//  CHECK-NEXT:     arith.addf
//  CHECK-NEXT:   }

// -----

func.func @do_hoist_scf_for_with_known_trip_count() {
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %c1 = arith.constant 1 : index
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  scf.for %arg0 = %c4 to %c6 step %c1 {
    %v0 = arith.addf %cf7, %cf8 : f32
  }
  return
}

// CHECK-LABEL: @do_hoist_scf_for_with_known_trip_count
//       CHECK:   arith.addf
//       CHECK:   scf.for

// -----

func.func @do_not_hoist_scf_while() {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  scf.while (%iter = %c0) : (index) -> (index) {
    %cond = arith.cmpi slt, %iter, %c4 : index
    scf.condition(%cond) %iter : index
  } do {
  ^bb0(%arg1: index):
    %v0 = arith.addf %cf7, %cf8 : f32
    scf.yield %arg1 : index
  }
  return
}

// CHECK-LABEL: @do_not_hoist_scf_while
//       CHECK:   scf.while
//       CHECK:     scf.condition
//       CHECK:     arith.addf
//       CHECK:     scf.yield
