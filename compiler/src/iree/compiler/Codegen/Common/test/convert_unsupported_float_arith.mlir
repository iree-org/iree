// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-convert-unsupported-float-arith))" %s | FileCheck %s

// CHECK-LABEL: func.func @negf_f8_unsupported
// CHECK-SAME: (%[[ARG0:.*]]: f8E4M3FNUZ) -> f8E4M3FNUZ
// CHECK: %[[EXT:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E4M3FNUZ to f32
// CHECK: %[[NEG:.*]] = arith.negf %[[EXT]] : f32
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[NEG]] {{.*}} : f32 to f8E4M3FNUZ
// CHECK: return %[[TRUNC]] : f8E4M3FNUZ
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @negf_f8_unsupported(%arg0 : f8E4M3FNUZ) -> f8E4M3FNUZ attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
    %0 = arith.negf %arg0 : f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}

// -----

// CHECK-LABEL: func.func @expand_f8(
// CHECK-SAME: %[[ARG0:.*]]: f8E5M2FNUZ
// CHECK: %[[EXT0:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E5M2FNUZ to f32
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f8E5M2FNUZ
// CHECK: %[[EXT1:.*]] = arith.extf %[[CST]] {{.*}} : f8E5M2FNUZ to f32
// CHECK: %[[SUM:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[SUM]] {{.*}} : f32 to f8E5M2FNUZ
// CHECK: return %[[TRUNC]] : f8E5M2FNUZ
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @expand_f8(%x: f8E5M2FNUZ) -> f8E5M2FNUZ attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
  %c = arith.constant 1.0 : f8E5M2FNUZ
  %y = arith.addf %x, %c : f8E5M2FNUZ
  func.return %y : f8E5M2FNUZ
}

// -----

// CHECK-LABEL: func.func @negf_f8_unsupported_ocp
// CHECK-SAME: (%[[ARG0:.*]]: f8E4M3FN) -> f8E4M3FN
// CHECK: %[[EXT:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E4M3FN to f32
// CHECK: %[[NEG:.*]] = arith.negf %[[EXT]] : f32
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[NEG]] {{.*}} : f32 to f8E4M3FN
// CHECK: return %[[TRUNC]] : f8E4M3FN
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @negf_f8_unsupported_ocp(%arg0 : f8E4M3FN) -> f8E4M3FN attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
    %0 = arith.negf %arg0 : f8E4M3FN
    return %0 : f8E4M3FN
}

// -----

// CHECK-LABEL: func.func @expand_f8_ocp(
// CHECK-SAME: %[[ARG0:.*]]: f8E5M2
// CHECK: %[[EXT0:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E5M2 to f32
// CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f8E5M2
// CHECK: %[[EXT1:.*]] = arith.extf %[[CST]] {{.*}} : f8E5M2 to f32
// CHECK: %[[SUM:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK: %[[TRUNC:.*]] = arith.truncf %[[SUM]] {{.*}} : f32 to f8E5M2
// CHECK: return %[[TRUNC]] : f8E5M2
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.gpu.target = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @expand_f8_ocp(%x: f8E5M2) -> f8E5M2 attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
  %c = arith.constant 1.0 : f8E5M2
  %y = arith.addf %x, %c : f8E5M2
  func.return %y : f8E5M2
}

// -----

// CHECK-LABEL: func.func @dont_expand_cpu_target
// CHECK: %[[NEG:.*]] = arith.negf {{.*}} : f8E4M3FNUZ
func.func @dont_expand_cpu_target(%arg0 : f8E4M3FNUZ) -> f8E4M3FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.negf %arg0 : f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}
