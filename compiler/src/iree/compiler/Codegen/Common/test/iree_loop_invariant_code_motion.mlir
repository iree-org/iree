// RUN: iree-opt %s  -split-input-file --iree-loop-invariant-code-motion | FileCheck %s

func.func @nested_loops_code_invariant_to_both() {
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
//   CHECK-DAG:   arith.constant 7
//   CHECK-DAG:   arith.constant 8
//       CHECK:   arith.addf
//       CHECK:   affine.for
//       CHECK:     affine.for

// -----

func.func @do_not_hoist_with_unknown_trip_count(%lb: index, %ub: index) {
  %cf7 = arith.constant 7.0 : f32
  %cf8 = arith.constant 8.0 : f32
  affine.for %arg1 = %lb to %ub {
    affine.for %arg0 = 0 to 10 {
      %v0 = arith.addf %cf7, %cf8 : f32
    }
  }
  return
}

// CHECK-LABEL: @do_not_hoist_with_unknown_trip_count
//       CHECK:   affine.for
//       CHECK:     arith.addf
//       CHECK:     affine.for
//       CHECK:     }
//       CHECK:   }

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

// -----

func.func @hoist_from_barrier_region(%arg0: tensor<6xf32>, %arg1: f32) -> tensor<1xf32> {
  %0 = iree_gpu.barrier_region ins(%arg0 : tensor<6xf32>) {
  ^bb0(%intermediate: tensor<6xf32>):
    %add = arith.addf %arg1, %arg1 : f32
    %slice = tensor.extract_slice %intermediate[1] [1] [1] : tensor<6xf32> to tensor<1xf32>
    iree_gpu.yield %slice : tensor<1xf32>
  } : tensor<1xf32>
  return %0 : tensor<1xf32>
}

// CHECK-LABEL: @hoist_from_barrier_region
//       CHECK:   arith.addf
//       CHECK:   iree_gpu.barrier_region
