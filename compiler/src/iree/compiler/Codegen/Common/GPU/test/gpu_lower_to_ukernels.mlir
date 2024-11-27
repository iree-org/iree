// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx942 --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --iree-gpu-test-target=gfx908 --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s --check-prefix=CDNA1

#gfx942_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_2d_f32i64(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//CHECK-LABEL: func @argmax_2d_f32i64(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>
//  CHECK-DAG:   %[[C1_index:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C0_i64:.+]] = arith.constant 0
//  CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_i64]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic {hal.executable.objects = [{{.*}}]} "iree_uk_amdgpu_argmax_f32i64"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#gfx942_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_4d_unit_parallel_f32i64(%arg0 : tensor<1x1x1x?xf32>) -> tensor<1x1x1xi64> attributes {
  hal.executable.target = #gfx942_target
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1x1x1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1x1x1xi64>) -> tensor<1x1x1xi64>
  %2 = tensor.empty() : tensor<1x1x1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x1x1xf32>) -> tensor<1x1x1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<1x1x1x?xf32>) outs(%3, %1 : tensor<1x1x1xf32>, tensor<1x1x1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1x1x1xf32>, tensor<1x1x1xi64>)
  return %4#1 : tensor<1x1x1xi64>
}

//      CHECK-LABEL: func @argmax_4d_unit_parallel_f32i64(
//      CHECK: iree_codegen.ukernel.generic
//      CHECK-NOT: linalg.generic

// -----

#gfx942_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_2d_non_unit_parallel_f32i64(%arg0 : tensor<4x?xf32>) -> tensor<4xi64> attributes {
  hal.executable.target = #gfx942_target
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<4xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<4xi64>) -> tensor<4xi64>
  %2 = tensor.empty() : tensor<4xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<4xf32>) -> tensor<4xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<4x?xf32>) outs(%3, %1 : tensor<4xf32>, tensor<4xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<4xf32>, tensor<4xi64>)
  return %4#1 : tensor<4xi64>
}

//      CHECK-LABEL: func @argmax_2d_non_unit_parallel_f32i64(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

#gfx942_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_2d_dyn_parallel_f32i64(%arg0 : tensor<?x?xf32>) -> tensor<?xi64> attributes {
  hal.executable.target = #gfx942_target
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

//      CHECK-LABEL: func @argmax_2d_dyn_parallel_f32i64(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

#gfx942_target_ukernels_none = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "none"}>
func.func @argmax_none_ukernel_enabled(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target_ukernels_none
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//      CHECK-LABEL: func @argmax_none_ukernel_enabled(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

#gfx942_target_ukernels_argmax = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "argmax"}>
func.func @argmax_only_argmax_ukernel_enabled(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target_ukernels_argmax
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//      CDNA2-LABEL: func @argmax_only_argmax_ukernel_enabled(
//      CDNA2: iree_codegen.ukernel.generic
//      CDNA2-NOT: linalg.generic

// -----

#gfx942_target_ukernels_foo_argmax_bar = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "foo,argmax,bar"}>
func.func @argmax_only_foo_argmax_bar_ukernel_enabled(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target_ukernels_foo_argmax_bar
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//      CHECK-LABEL: func @argmax_only_foo_argmax_bar_ukernel_enabled(
//      CHECK: iree_codegen.ukernel.generic
//      CHECK-NOT: linalg.generic

//      CDNA2-LABEL: func @argmax_only_foo_argmax_bar_ukernel_enabled(

// -----

#gfx942_target_ukernels_foo_argmax_bar = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "foo"}>
func.func @argmax_only_foo_ukernel_enabled(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target_ukernels_foo_argmax_bar
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//      CHECK-LABEL: func @argmax_only_foo_ukernel_enabled(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

// Currently we do only handle -Inf case as initial values.
#gfx942_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_2d_f32i64_not_neg_inf_init(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//      CHECK-LABEL: func @argmax_2d_f32i64_not_neg_inf_init(
//      CHECK-NOT: iree_codegen.ukernel.generic
//      CHECK: linalg.generic

// -----

// TODO: No technical reason this architecture is not supported.
//       Currently just picking out popular chips to support,
//       to minimize compile time and space.

#gfx908_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx908", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_ukernel_unsupported_arch(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx908_target
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//      CDNA1-LABEL: func @argmax_ukernel_unsupported_arch(
//      CDNA1-NOT: iree_codegen.ukernel.generic
//      CDNA1: linalg.generic

// -----

// Test user-provided bitcode in the source IR.

#gfx942_target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute = fp32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [], max_workgroup_sizes = [], max_thread_count_per_workgroup = 0, max_workgroup_memory_bytes = 0, max_workgroup_counts = []>>, ukernels = "all"}>
func.func @argmax_2d_f32i64(%arg0 : tensor<1x?xf32>) -> tensor<1xi64> attributes {
  hal.executable.target = #gfx942_target,
  // Dummy bitcode with an unusual length of 12.
  hal.executable.objects = [#hal.executable.object<{path = "iree_uk_amdgpu_argmax_f32i64.c.gfx942.bc", data = dense<"0x4243C0DE0123456789ABCDEF"> : tensor<12xi8>}>]
} {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant 0xFF800000 : f32
  %0 = tensor.empty() : tensor<1xi64>
  %1 = linalg.fill ins(%c0_i64 : i64) outs(%0 : tensor<1xi64>) -> tensor<1xi64>
  %2 = tensor.empty() : tensor<1xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1xf32>) -> tensor<1xf32>
  %4:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<1x?xf32>) outs(%3, %1 : tensor<1xf32>, tensor<1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.maximumf %in, %out : f32
    %8 = arith.cmpf ogt, %in, %out : f32
    %9 = arith.select %8, %6, %out_0 : i64
    linalg.yield %7, %9 : f32, i64
  } -> (tensor<1xf32>, tensor<1xi64>)
  return %4#1 : tensor<1xi64>
}

//CHECK-LABEL: func @argmax_2d_f32i64(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x?xf32>
//  CHECK-DAG:   %[[C1_index:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C0_i64:.+]] = arith.constant 0
//  CHECK-DAG:   %[[FILL:.+]] = linalg.fill ins(%[[C0_i64]]
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic {hal.executable.objects = [#hal.executable.object<{path = "iree_uk_amdgpu_argmax_f32i64.c.gfx942.bc", data = dense<[66, 67, -64, -34, 1, 35, 69, 103, -119, -85, -51, -17]> : tensor<12xi8>}>]} "iree_uk_amdgpu_argmax_f32i64"
// CHECK-SAME:       ins(%[[ARG0]] :
// CHECK-SAME:       outs(%[[FILL]] :
//      CHECK:   return %[[MICRO_KERNEL]]
