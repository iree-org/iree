// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s

func.func @argmax_2d_f32i64(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?xi64>) -> tensor<?xi64>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%3, %1 : tensor<?xf32>, tensor<?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<?xf32>, tensor<?xi64>)
  return %4#1 : tensor<?xi64>
}

//      CHECK: func @argmax_2d_f32i64(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0_index:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1_index:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C0_i64:.+]] = arith.constant 0
//  CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_i64]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "__iree_uk_rocm_argmax_F32I64"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @argmax_none_ukernel_enabled(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100", ukernels = "none"}>
} {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?xi64>) -> tensor<?xi64>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%3, %1 : tensor<?xf32>, tensor<?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<?xf32>, tensor<?xi64>)
  return %4#1 : tensor<?xi64>
}

//      CHECK: func @argmax_none_ukernel_enabled(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

func.func @argmax_only_argmax_ukernel_enabled(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx90a", ukernels = "argmax"}>
} {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?xi64>) -> tensor<?xi64>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%3, %1 : tensor<?xf32>, tensor<?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<?xf32>, tensor<?xi64>)
  return %4#1 : tensor<?xi64>
}

//      CHECK: func @argmax_only_argmax_ukernel_enabled(
//      CHECK: iree_codegen.ukernel.generic
//      CHECK-NOT: linalg.generic

// -----

func.func @argmax_only_foo_argmax_bar_ukernel_enabled(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100", ukernels = "foo,argmax,bar"}>
} {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?xi64>) -> tensor<?xi64>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%3, %1 : tensor<?xf32>, tensor<?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<?xf32>, tensor<?xi64>)
  return %4#1 : tensor<?xi64>
}

//      CHECK: func @argmax_only_foo_argmax_bar_ukernel_enabled(
//      CHECK: iree_codegen.ukernel.generic
//      CHECK-NOT: linalg.generic

// -----

func.func @argmax_only_foo_ukernel_enabled(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gfx1100", ukernels = "foo"}>
} {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?xi64>) -> tensor<?xi64>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%3, %1 : tensor<?xf32>, tensor<?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<?xf32>, tensor<?xi64>)
  return %4#1 : tensor<?xi64>
}

//      CHECK: func @argmax_only_foo_ukernel_enabled(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

// TODO: No technical reason this architecture is not supported.
//       Currently just picking out popular chips to support,
//       to minimize compile time and space.

func.func @argmax_ukernel_unsupported_arch(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_arch = "gf1100", ukernels = "all"}>
} {
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %0 = tensor.empty(%dim) : tensor<?xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<?xi64>) -> tensor<?xi64>
  %2 = tensor.empty(%dim) : tensor<?xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<?xf32>) -> tensor<?xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%3, %1 : tensor<?xf32>, tensor<?xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<?xf32>, tensor<?xi64>)
  return %4#1 : tensor<?xi64>
}

//      CHECK: func @argmax_ukernel_unsupported_arch(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic
