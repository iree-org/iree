// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-vmvx-split-reduction{reassociate-fp-reductions=true reduction-chunk-size=128}))' --split-input-file %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-vmvx-split-reduction{reassociate-fp-reductions=true}))' --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-vmvx-split-reduction))' --split-input-file %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-vmvx-split-reduction{reassociate-fp-reductions=true}))' --split-input-file %s > %t
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-vmvx-split-reduction{reassociate-fp-reductions=true}))' --split-input-file %t | diff %t -

// The square stays in the partial reduction; only the add is repeated in
// the final reduction. The nonzero caller init must be used only by the final.
// CHECK-LABEL: func.func @dynamic_sum_squares(
// CHECK-SAME: %[[INPUT:[^:]+]]: tensor<?x576xf32>, %[[INIT:[^:]+]]: tensor<?xf32>
// CHECK: %[[DIM:.*]] = tensor.dim %[[INPUT]],
// CHECK: %[[EXPANDED:.*]] = tensor.expand_shape %[[INPUT]] {{.*}} output_shape [%[[DIM]], 9, 64]
// CHECK: %[[ZERO:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f32)
// CHECK: %[[PARTIAL:.*]] = linalg.generic {{.*}} ins(%[[EXPANDED]] : tensor<?x9x64xf32>) outs(%[[FILL]] : tensor<?x9xf32>)
// CHECK: ^bb0(%[[X:[^:]+]]: f32, %[[ACC:[^:]+]]: f32):
// CHECK: %[[SQUARE:.*]] = arith.mulf %[[X]], %[[X]] : f32
// CHECK: %[[SUM:.*]] = arith.addf %[[ACC]], %[[SQUARE]] fastmath<contract> : f32
// CHECK: linalg.yield %[[SUM]] : f32
// CHECK: %[[FINAL:.*]] = linalg.generic {{.*}} ins(%[[PARTIAL]] : tensor<?x9xf32>) outs(%[[INIT]] : tensor<?xf32>)
// CHECK: ^bb0(%[[P:[^:]+]]: f32, %[[A:[^:]+]]: f32):
// CHECK-NEXT: %[[ADDED:.*]] = arith.addf %[[A]], %[[P]] fastmath<contract> : f32
// CHECK-NEXT: linalg.yield %[[ADDED]] : f32
// CHECK: return %[[FINAL]]
// DEFAULT-LABEL: func.func @dynamic_sum_squares
// DEFAULT-NOT: tensor.expand_shape
// DEFAULT: ins(%{{.*}} : tensor<?x576xf32>)
// DEFAULT: return
func.func @dynamic_sum_squares(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %square = arith.mulf %x, %x : f32
    %sum = arith.addf %acc, %square fastmath<contract> : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Extents larger than 64 squared need a second level of partial sums.
// CHECK-LABEL: func.func @recursive_sum
// CHECK: tensor.expand_shape {{.*}} into tensor<1x128x64xf32>
// CHECK: tensor.expand_shape {{.*}} into tensor<1x2x64xf32>
// CHECK: ins(%{{.*}} : tensor<1x2xf32>) outs(%{{.*}} : tensor<1xf32>)
// CHECK: return
func.func @recursive_sum(%in: tensor<1x8192xf32>, %init: tensor<1xf32>) -> tensor<1xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<1x8192xf32>) outs(%init : tensor<1xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %x, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<1xf32>
  return %result : tensor<1xf32>
}

// -----

// Dynamic reduction extents have no supported split.
// CHECK-LABEL: func.func @dynamic_reduction
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @dynamic_reduction(%in: tensor<?x?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// A tail needs a separate algorithm and must be declined.
// CHECK-LABEL: func.func @nondivisible
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @nondivisible(%in: tensor<?x577xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x577xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// There is no work to split at the chunk boundary.
// CHECK-LABEL: func.func @small
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @small(%in: tensor<?x64xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x64xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// The precision policy is specific to f32 sums.
// CHECK-LABEL: func.func @half
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @half(%in: tensor<?x576xf16>, %init: tensor<?xf16>) -> tensor<?xf16> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf16>) outs(%init : tensor<?xf16>) {
  ^bb0(%x: f16, %acc: f16):
    %sum = arith.addf %acc, %x : f16
    linalg.yield %sum : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// -----

// Other floating-point types keep their association.
// CHECK-LABEL: func.func @double
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @double(%in: tensor<?x576xf64>, %init: tensor<?xf64>) -> tensor<?xf64> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf64>) outs(%init : tensor<?xf64>) {
  ^bb0(%x: f64, %acc: f64):
    %sum = arith.addf %acc, %x : f64
    linalg.yield %sum : f64
  } -> tensor<?xf64>
  return %result : tensor<?xf64>
}

// -----

// Integer reductions are outside this policy.
// CHECK-LABEL: func.func @integer
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @integer(%in: tensor<?x576xi32>, %init: tensor<?xi32>) -> tensor<?xi32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xi32>) outs(%init : tensor<?xi32>) {
  ^bb0(%x: i32, %acc: i32):
    %sum = arith.addi %x, %acc : i32
    linalg.yield %sum : i32
  } -> tensor<?xi32>
  return %result : tensor<?xi32>
}

// -----

// Only addition admits this partial-sum construction.
// CHECK-LABEL: func.func @product
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @product(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %product = arith.mulf %acc, %x : f32
    linalg.yield %product : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Contributions may not depend on the running accumulator.
// CHECK-LABEL: func.func @accumulator_dependency
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @accumulator_dependency(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %product = arith.mulf %acc, %x : f32
    %sum = arith.addf %acc, %product : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Splitting changes linalg.index, so indexed bodies must be declined.
// CHECK-LABEL: func.func @index_semantics
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @index_semantics(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %i = linalg.index 1 : index
    %integer = arith.index_cast %i : index to i32
    %f = arith.sitofp %integer : i32 to f32
    %sum = arith.addf %acc, %f : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Unknown annotations may depend on the old iteration space.
// CHECK-LABEL: func.func @unknown_annotation
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @unknown_annotation(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], test.semantic = 7 : i64} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// A library call promises a specific external implementation.
// CHECK-LABEL: func.func @library_call
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @library_call(%in: tensor<?x576xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"], library_call = "external_sum"} ins(%in : tensor<?x576xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Only contiguous innermost reductions are handled.
// CHECK-LABEL: func.func @transposed_input
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @transposed_input(%in: tensor<576x?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%in : tensor<576x?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// An outer reduction is outside the contiguous-chunk contract.
// CHECK-LABEL: func.func @noninnermost
// CHECK-NOT: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-NOT: tensor.expand_shape
// CHECK: return
func.func @noninnermost(%in: tensor<576x?xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>], iterator_types = ["reduction", "parallel"]} ins(%in : tensor<576x?xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %acc, %x : f32
    linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

// Region-bearing scalar bodies are outside the supported split contract.
// CHECK-LABEL: func.func @nested_region
// CHECK-NOT: tensor.expand_shape
// CHECK: scf.if
// CHECK: return
func.func @nested_region(%in: tensor<576xf32>, %init: tensor<f32>) -> tensor<f32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%in : tensor<576xf32>) outs(%init : tensor<f32>) {
  ^bb0(%x: f32, %acc: f32):
    %cond = arith.cmpf ogt, %x, %x : f32
    %a = scf.if %cond -> f32 {
      scf.yield %x : f32
    } else {
      %negative = arith.negf %x : f32
      scf.yield %negative : f32
    }
    %sum = arith.addf %a, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Retain the root configuration on the final reduction with the original rank.
// CHECK-DAG: #[[$CONFIG:.+]] = #iree_cpu.lowering_config<distribution = [0]>
// CHECK-LABEL: func.func @configured
// CHECK: linalg.generic
// CHECK-NOT: lowering_config
// CHECK: linalg.yield
// CHECK: linalg.generic
// CHECK-SAME: attrs = {lowering_config = #[[$CONFIG]]}
// CHECK: return
func.func @configured(%in: tensor<576xf32>, %init: tensor<f32>) -> tensor<f32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"], lowering_config = #iree_cpu.lowering_config<distribution = [0]>} ins(%in : tensor<576xf32>) outs(%init : tensor<f32>) {
  ^bb0(%x: f32, %acc: f32):
    %sum = arith.addf %x, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<f32>
  return %r : tensor<f32>
}

// -----

// Calls without a pure-effect contract must not move into partial sums.
// CHECK-LABEL: func.func @effectful_contribution
// CHECK-NOT: tensor.expand_shape
// CHECK: func.call @external_contribution
// CHECK: return
func.func @effectful_contribution(%in: tensor<576xf32>, %init: tensor<f32>) -> tensor<f32> {
  %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%in : tensor<576xf32>) outs(%init : tensor<f32>) {
  ^bb0(%x: f32, %acc: f32):
    %a = func.call @external_contribution(%x) : (f32) -> f32
    %sum = arith.addf %a, %acc : f32
    linalg.yield %sum : f32
  } -> tensor<f32>
  return %r : tensor<f32>
}
func.func private @external_contribution(f32) -> f32
