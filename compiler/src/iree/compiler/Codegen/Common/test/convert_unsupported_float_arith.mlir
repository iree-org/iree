// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-convert-unsupported-float-arith))" %s | FileCheck %s

// CHECK-LABEL: func.func @negf_f8_unsupported
// CHECK-SAME:    (%[[ARG0:.*]]: f8E4M3FNUZ) -> f8E4M3FNUZ
// CHECK:         %[[EXT:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E4M3FNUZ to f32
// CHECK:         %[[NEG:.*]] = arith.negf %[[EXT]] : f32
// CHECK:         %[[TRUNC:.*]] = arith.truncf %[[NEG]] {{.*}} : f32 to f8E4M3FNUZ
// CHECK:         return %[[TRUNC]] : f8E4M3FNUZ
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @negf_f8_unsupported(%arg0 : f8E4M3FNUZ) -> f8E4M3FNUZ attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
    %0 = arith.negf %arg0 : f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}

// -----

// CHECK-LABEL: func.func @expand_f8(
// CHECK-SAME:    %[[ARG0:.*]]: f8E5M2FNUZ
// CHECK:         %[[EXT0:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E5M2FNUZ to f32
// CHECK:         %[[CST:.*]] = arith.constant 1.000000e+00 : f8E5M2FNUZ
// CHECK:         %[[EXT1:.*]] = arith.extf %[[CST]] {{.*}} : f8E5M2FNUZ to f32
// CHECK:         %[[SUM:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK:         %[[TRUNC:.*]] = arith.truncf %[[SUM]] {{.*}} : f32 to f8E5M2FNUZ
// CHECK:         return %[[TRUNC]] : f8E5M2FNUZ
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @expand_f8(%x: f8E5M2FNUZ) -> f8E5M2FNUZ attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
  %c = arith.constant 1.0 : f8E5M2FNUZ
  %y = arith.addf %x, %c : f8E5M2FNUZ
  func.return %y : f8E5M2FNUZ
}

// -----

// CHECK-LABEL: func.func @negf_f8_unsupported_ocp
// CHECK-SAME:    (%[[ARG0:.*]]: f8E4M3FN) -> f8E4M3FN
// CHECK:         %[[EXT:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E4M3FN to f32
// CHECK:         %[[NEG:.*]] = arith.negf %[[EXT]] : f32
// CHECK:         %[[TRUNC:.*]] = arith.truncf %[[NEG]] {{.*}} : f32 to f8E4M3FN
// CHECK:         return %[[TRUNC]] : f8E4M3FN
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @negf_f8_unsupported_ocp(%arg0 : f8E4M3FN) -> f8E4M3FN attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
    %0 = arith.negf %arg0 : f8E4M3FN
    return %0 : f8E4M3FN
}

// -----

// CHECK-LABEL: func.func @expand_f8_ocp(
// CHECK-SAME:    %[[ARG0:.*]]: f8E5M2
// CHECK:         %[[EXT0:.*]] = arith.extf %[[ARG0]] {{.*}} : f8E5M2 to f32
// CHECK:         %[[CST:.*]] = arith.constant 1.000000e+00 : f8E5M2
// CHECK:         %[[EXT1:.*]] = arith.extf %[[CST]] {{.*}} : f8E5M2 to f32
// CHECK:         %[[SUM:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK:         %[[TRUNC:.*]] = arith.truncf %[[SUM]] {{.*}} : f32 to f8E5M2
// CHECK:         return %[[TRUNC]] : f8E5M2
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree_codegen.target_info = #iree_gpu.target<arch = "gfx950", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
func.func @expand_f8_ocp(%x: f8E5M2) -> f8E5M2 attributes
{ hal.executable.target = #executable_target_rocm_hsaco_fb }{
  %c = arith.constant 1.0 : f8E5M2
  %y = arith.addf %x, %c : f8E5M2
  func.return %y : f8E5M2
}

// -----

// Test CPU target with f8E4M3FNUZ - both extf and truncf should be emulated.
//
// CHECK-LABEL: func.func @expand_cpu_target_f8e4m3fnuz
// CHECK-SAME:    (%[[ARG0:.*]]: f8E4M3FNUZ) -> f8E4M3FNUZ
//
// ExtF emulation: fp8 -> i8 -> i32 -> extract fields -> pack f32 -> bitcast
// CHECK:         %[[BITCAST_IN:.*]] = arith.bitcast %[[ARG0]] : f8E4M3FNUZ to i8
// CHECK:         %[[EXT_I32:.*]] = arith.extui %[[BITCAST_IN]] : i8 to i32
// CHECK:         arith.shrui
// CHECK:         arith.andi
// CHECK:         arith.bitcast %{{.*}} : i32 to f32
//
// Negf on f32
// CHECK:         %[[NEG:.*]] = arith.negf %{{.*}} : f32
//
// TruncF emulation: f32 -> bitcast i32 -> extract fields -> pack i8 -> bitcast fp8
// CHECK:         arith.bitcast %[[NEG]] : f32 to i32
// CHECK:         arith.shrui
// CHECK:         arith.andi
// CHECK:         arith.trunci %{{.*}} : i32 to i8
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : i8 to f8E4M3FNUZ
// CHECK:         return %[[RESULT]] : f8E4M3FNUZ
func.func @expand_cpu_target_f8e4m3fnuz(%arg0 : f8E4M3FNUZ) -> f8E4M3FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.negf %arg0 : f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}

// -----

// Test CPU target with f8E5M2FNUZ.
//
// CHECK-LABEL: func.func @expand_cpu_target_f8e5m2fnuz
// CHECK-SAME:    (%[[ARG0:.*]]: f8E5M2FNUZ) -> f8E5M2FNUZ
// CHECK:         %[[BITCAST_IN:.*]] = arith.bitcast %[[ARG0]] : f8E5M2FNUZ to i8
// CHECK:         %[[EXT_I32:.*]] = arith.extui %[[BITCAST_IN]] : i8 to i32
// CHECK:         arith.bitcast %{{.*}} : i32 to f32
// CHECK:         %[[ADD:.*]] = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:         arith.bitcast %[[ADD]] : f32 to i32
// CHECK:         arith.trunci %{{.*}} : i32 to i8
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : i8 to f8E5M2FNUZ
// CHECK:         return %[[RESULT]] : f8E5M2FNUZ
func.func @expand_cpu_target_f8e5m2fnuz(%arg0 : f8E5M2FNUZ) -> f8E5M2FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %c = arith.constant 1.0 : f8E5M2FNUZ
    %0 = arith.addf %arg0, %c : f8E5M2FNUZ
    return %0 : f8E5M2FNUZ
}

// -----

// Test explicit truncf emulation for CPU target.
//
// CHECK-LABEL: func.func @truncf_cpu_f32_to_f8e5m2fnuz
// CHECK-SAME:    (%[[ARG0:.*]]: f32) -> f8E5M2FNUZ
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : f32 to i32
// CHECK:         arith.shrui %[[BITCAST]]
// CHECK:         arith.andi
// CHECK:         arith.subi
// CHECK:         arith.cmpi
// CHECK:         arith.select
// CHECK:         arith.trunci %{{.*}} : i32 to i8
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : i8 to f8E5M2FNUZ
// CHECK:         return %[[RESULT]] : f8E5M2FNUZ
func.func @truncf_cpu_f32_to_f8e5m2fnuz(%arg0 : f32) -> f8E5M2FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.truncf %arg0 : f32 to f8E5M2FNUZ
    return %0 : f8E5M2FNUZ
}

// -----

// Test explicit extf emulation for CPU target.
//
// CHECK-LABEL: func.func @extf_cpu_f8e5m2fnuz_to_f32
// CHECK-SAME:    (%[[ARG0:.*]]: f8E5M2FNUZ) -> f32
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : f8E5M2FNUZ to i8
// CHECK:         %[[EXT:.*]] = arith.extui %[[BITCAST]] : i8 to i32
// CHECK:         arith.shrui %[[EXT]]
// CHECK:         arith.andi
// Denormal handling uses uitofp + mulf instead of enumeration.
// CHECK:         arith.uitofp {{.*}} : i32 to f32
// CHECK:         arith.mulf {{.*}} : f32
// CHECK:         arith.select
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK:         return %[[RESULT]] : f32
func.func @extf_cpu_f8e5m2fnuz_to_f32(%arg0 : f8E5M2FNUZ) -> f32 attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.extf %arg0 : f8E5M2FNUZ to f32
    return %0 : f32
}

// -----

// Test vector truncf emulation for CPU target.
//
// CHECK-LABEL: func.func @truncf_cpu_vector_f32_to_f8e5m2fnuz
// CHECK-SAME:    (%[[ARG0:.*]]: vector<4xf32>) -> vector<4xf8E5M2FNUZ>
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : vector<4xf32> to vector<4xi32>
// CHECK:         arith.shrui
// CHECK:         arith.andi
// CHECK:         arith.trunci %{{.*}} : vector<4xi32> to vector<4xi8>
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : vector<4xi8> to vector<4xf8E5M2FNUZ>
// CHECK:         return %[[RESULT]] : vector<4xf8E5M2FNUZ>
func.func @truncf_cpu_vector_f32_to_f8e5m2fnuz(%arg0 : vector<4xf32>) -> vector<4xf8E5M2FNUZ> attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.truncf %arg0 : vector<4xf32> to vector<4xf8E5M2FNUZ>
    return %0 : vector<4xf8E5M2FNUZ>
}

// -----

// Test vector extf emulation for CPU target.
// CHECK-LABEL: func.func @extf_cpu_vector_f8e5m2fnuz_to_f32
// CHECK-SAME:    (%[[ARG0:.*]]: vector<4xf8E5M2FNUZ>) -> vector<4xf32>
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : vector<4xf8E5M2FNUZ> to vector<4xi8>
// CHECK:         %[[EXT:.*]] = arith.extui %[[BITCAST]] : vector<4xi8> to vector<4xi32>
// CHECK:         arith.shrui
// CHECK:         arith.andi
// Denormal handling uses uitofp + mulf.
// CHECK:         arith.uitofp {{.*}} : vector<4xi32> to vector<4xf32>
// CHECK:         arith.mulf {{.*}} : vector<4xf32>
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : vector<4xi32> to vector<4xf32>
// CHECK:         return %[[RESULT]] : vector<4xf32>
func.func @extf_cpu_vector_f8e5m2fnuz_to_f32(%arg0 : vector<4xf8E5M2FNUZ>) -> vector<4xf32> attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.extf %arg0 : vector<4xf8E5M2FNUZ> to vector<4xf32>
    return %0 : vector<4xf32>
}

// -----

// Test f8E4M3FNUZ truncf emulation for CPU target.
//
// CHECK-LABEL: func.func @truncf_cpu_f32_to_f8e4m3fnuz
// CHECK-SAME:    (%[[ARG0:.*]]: f32) -> f8E4M3FNUZ
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : f32 to i32
// CHECK:         arith.trunci %{{.*}} : i32 to i8
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : i8 to f8E4M3FNUZ
// CHECK:         return %[[RESULT]] : f8E4M3FNUZ
func.func @truncf_cpu_f32_to_f8e4m3fnuz(%arg0 : f32) -> f8E4M3FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.truncf %arg0 : f32 to f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}

// -----

// Test f8E4M3FNUZ extf emulation for CPU target.
//
// CHECK-LABEL: func.func @extf_cpu_f8e4m3fnuz_to_f32
// CHECK-SAME:    (%[[ARG0:.*]]: f8E4M3FNUZ) -> f32
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : f8E4M3FNUZ to i8
// CHECK:         %[[EXT:.*]] = arith.extui %[[BITCAST]] : i8 to i32
// Denormal handling uses uitofp + mulf.
// CHECK:         arith.uitofp {{.*}} : i32 to f32
// CHECK:         arith.mulf {{.*}} : f32
// CHECK:         %[[RESULT:.*]] = arith.bitcast %{{.*}} : i32 to f32
// CHECK:         return %[[RESULT]] : f32
func.func @extf_cpu_f8e4m3fnuz_to_f32(%arg0 : f8E4M3FNUZ) -> f32 attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.extf %arg0 : f8E4M3FNUZ to f32
    return %0 : f32
}

// -----

// Test FNUZ NaN handling in truncf: NaN/Inf/overflow must produce 0x80, not 0.
//
// Background: FNUZ types encode NaN as 0x80 (sign bit set, exponent and
// mantissa zero). This conflicts with negative zero correction, which also
// targets 0x80. The fix is to order the select cascade so negative zero
// correction happens BEFORE NaN/Inf/overflow handling, allowing special
// cases to override any incorrect intermediate results.
//
// Key invariant: The srcIsNan select must be the LAST select before trunci,
// ensuring NaN input produces 0x80 output regardless of other logic.
//
// CHECK-LABEL: func.func @truncf_fnuz_nan_handling
// CHECK-SAME:    (%[[ARG0:.*]]: f32) -> f8E5M2FNUZ
//
// 0x80 (128) = FNUZ NaN encoding (sign bit set, everything else zero).
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : i32
//
// 0xFF (255) = max f32 exponent, used to detect NaN/Inf in source.
// srcIsNanOrInf = (srcExp == 255)
// srcMantIsZero = (srcMant == 0)
// srcIsInf = srcIsNanOrInf && srcMantIsZero
// srcIsNan = srcIsNanOrInf XOR srcIsInf (avoids redundant comparison)
// CHECK-DAG:     %[[C255:.*]] = arith.constant 255 : i32
// CHECK:         %[[EXP_IS_MAX:.*]] = arith.cmpi eq, %{{.*}}, %[[C255]] : i32
// CHECK:         %[[MANT_EQ_ZERO:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK:         %[[SRC_IS_INF:.*]] = arith.andi %[[EXP_IS_MAX]], %[[MANT_EQ_ZERO]] : i1
// CHECK:         %[[SRC_IS_NAN:.*]] = arith.xori %[[EXP_IS_MAX]], %[[SRC_IS_INF]] : i1
//
// Critical ordering check: srcIsNan select must be immediately before trunci.
// This ensures it's the outermost select, overriding negative zero correction.
// CHECK:         %[[FINAL:.*]] = arith.select %[[SRC_IS_NAN]], %[[C128]], %{{.*}} : i32
// CHECK-NEXT:    arith.trunci %[[FINAL]] : i32 to i8
func.func @truncf_fnuz_nan_handling(%arg0 : f32) -> f8E5M2FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.truncf %arg0 : f32 to f8E5M2FNUZ
    return %0 : f8E5M2FNUZ
}

// -----

// Test FNUZ NaN handling for f8E4M3FNUZ type (same invariant as E5M2 above).
//
// CHECK-LABEL: func.func @truncf_fnuz_nan_handling_e4m3
// CHECK-SAME:    (%[[ARG0:.*]]: f32) -> f8E4M3FNUZ
// 0x80 = FNUZ NaN, 0xFF = max f32 exponent
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : i32
// CHECK-DAG:     %[[C255:.*]] = arith.constant 255 : i32
// CHECK:         %[[EXP_IS_MAX:.*]] = arith.cmpi eq, %{{.*}}, %[[C255]] : i32
// CHECK:         %[[MANT_EQ_ZERO:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : i32
// CHECK:         %[[SRC_IS_INF:.*]] = arith.andi %[[EXP_IS_MAX]], %[[MANT_EQ_ZERO]] : i1
// CHECK:         %[[SRC_IS_NAN:.*]] = arith.xori %[[EXP_IS_MAX]], %[[SRC_IS_INF]] : i1
// srcIsNan select must be immediately before trunci (outermost in cascade).
// CHECK:         %[[FINAL:.*]] = arith.select %[[SRC_IS_NAN]], %[[C128]], %{{.*}} : i32
// CHECK-NEXT:    arith.trunci %[[FINAL]] : i32 to i8
func.func @truncf_fnuz_nan_handling_e4m3(%arg0 : f32) -> f8E4M3FNUZ attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.truncf %arg0 : f32 to f8E4M3FNUZ
    return %0 : f8E4M3FNUZ
}

// -----

// Test FNUZ NaN detection in extf: 0x80 input must produce f32 NaN.
//
// Background: For FNUZ types, 0x80 is the NaN encoding (not negative zero
// like in IEEE formats). When extending to f32, we must detect this pattern
// and produce a canonical f32 NaN.
//
// CHECK-LABEL: func.func @extf_fnuz_nan_handling
// CHECK-SAME:    (%[[ARG0:.*]]: f8E5M2FNUZ) -> f32
//
// 0x80 (128) = FNUZ NaN encoding.
// 0x7FC00000 (2143289344) = canonical f32 quiet NaN.
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : i32
// CHECK-DAG:     %[[F32_NAN:.*]] = arith.constant 2143289344 : i32
// CHECK:         %[[BITCAST:.*]] = arith.bitcast %[[ARG0]] : f8E5M2FNUZ to i8
// CHECK:         %[[EXT:.*]] = arith.extui %[[BITCAST]] : i8 to i32
// isNan = (inputBits == 0x80)
// CHECK:         %[[IS_NAN:.*]] = arith.cmpi eq, %[[EXT]], %[[C128]] : i32
//
// If input is FNUZ NaN (0x80), output canonical f32 NaN (0x7FC00000).
// CHECK:         arith.select %[[IS_NAN]], %[[F32_NAN]], %{{.*}} : i32
func.func @extf_fnuz_nan_handling(%arg0 : f8E5M2FNUZ) -> f32 attributes
{ hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>}{
    %0 = arith.extf %arg0 : f8E5M2FNUZ to f32
    return %0 : f32
}
