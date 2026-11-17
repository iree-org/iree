// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-codegen-math-transform))' --split-input-file %s | FileCheck %s

// CHECK-LABEL: @cbrt_cpu_vector
func.func @cbrt_cpu_vector(%arg0: vector<4xf32>) -> vector<4xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} {
  // Vector types alone do not load the Vector dialect. The pass must load it
  // before the approximation creates vector.broadcast operations.
  // CHECK: math.absf
  // CHECK: arith.bitcast
  // CHECK: math.copysign
  // CHECK-NOT: math.cbrt
  // CHECK: return
  %0 = math.cbrt %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @rewrite_tan
func.func @rewrite_tan(%arg0: f16) -> f16 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} {
  // Tan should be directly approximated by a rational function. It's also possible
  // (though not good) that it gets rewritten as sin/cos and those get approximated by
  // rational functions. Either way, we expect to see rational arithmetic here, on f32
  // as the operands get casted to f32.
  // CHECK-NOT:     math.tan
  // CHECK-NOT:     math.sin
  // CHECK-NOT:     math.cos
  // CHECK:        math.fma {{.*}} : f32
  // Final division after cast to f16.
  // CHECK:         arith.divf {{.*}} : f16
  %0 = math.tan %arg0 : f16
  return %0 : f16
}

// -----

// CHECK-LABEL: @rewrite_pow
func.func @rewrite_pow(%arg0: f16, %arg1: f16) -> f16 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} {

  // Powf should be either directly approximated, or first rewritten into log and
  // exp and then those get approximated. Some targets with fast exponentials might
  // prefer to keep the exponential form, but this is not the case with the current
  // lowering for CPU, so we expect to see rational arithmetic here, on f32 as the
  // operands get casted to f32.
  // CHECK-NOT:     math.powf
  // CHECK-NOT:     math.exp
  // CHECK-NOT:     math.log
  // CHECK:        math.fma {{.*}} : f32
  %0 = math.powf %arg0, %arg1 : f16
  return %0 : f16
}

// -----

// CHECK-LABEL: @rewrite_erf
func.func @rewrite_erf(%arg0: f16) -> f16 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} {
  // Erf should be directly approximated by a rational function. Some targets
  // with fast exponentials might prefer an exponential approximation, but this
  // is not the case with the current lowering for CPU, so we expect to see rational
  // arithmetic here, on f32 as the operands get casted to f32.
  // CHECK-NOT:     math.erf
  // CHECK-NOT:     math.exp
  // CHECK-NOT:     math.log
  // CHECK:        math.fma {{.*}} : f32
  %0 = math.erf %arg0 : f16
  return %0 : f16
}

// -----

// CHECK-LABEL: func.func @erf_fastmath
func.func @erf_fastmath(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_triple = "amdgcn-amd-amdhsa"}>
} {
  // CHECK-NOT: math.erf

  // CHECK-DAG: arith.constant 1.000000e+00 : f32

  // Then region coefficients (|x| < 1.0)
  // CHECK-DAG: arith.constant 0.128689408 : f32
  // CHECK-DAG: arith.constant 0.634933292 : f32
  // CHECK-DAG: arith.constant 0.106668264 : f32
  // CHECK-DAG: arith.constant -0.0241816975 : f32
  // CHECK-DAG: arith.constant 0.00385781587 : f32
  // CHECK-DAG: arith.constant -3.78677854E-4 : f32

  // Else region coefficients (|x| >= 1.0)
  // CHECK-DAG: arith.constant 1.69988107E-5 : f32
  // CHECK-DAG: arith.constant 0.128379107 : f32
  // CHECK-DAG: arith.constant -0.376122952 : f32
  // CHECK-DAG: arith.constant 0.112800106 : f32
  // CHECK-DAG: arith.constant -0.0267075151 : f32
  // CHECK-DAG: arith.constant 0.00491381623 : f32
  // CHECK-DAG: arith.constant -5.61801775E-4 : f32

  // Structure checks
  // CHECK: %[[AX:.*]] = math.absf %arg0 : f32
  // CHECK: %[[CMP:.*]] = arith.cmpf olt, %[[AX]], %{{.*}} : f32
  // CHECK: %[[IF:.*]] = scf.if %[[CMP]] -> (f32) {

  // Then region - verify we have exactly 6 FMA operations.
  // CHECK: %[[T:.*]] = arith.mulf %[[AX]], %[[AX]] : f32
  // CHECK-COUNT-5: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %[[AX]] : f32
  // CHECK: scf.yield %{{.*}} : f32

  // CHECK: } else {

  // Else region - verify we have exactly 6 FMA operations.
  // CHECK-COUNT-6: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %[[AX]] : f32

  // CHECK: %{{.*}} = arith.negf %{{.*}} : f32
  // CHECK: %{{.*}} = math.exp %{{.*}} : f32
  // CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f32
  // CHECK: scf.yield %{{.*}} : f32
  // CHECK: }

  // CHECK: %{{.*}} = math.copysign %[[IF]], %arg0 : f32
  %0 = math.erf %arg0 : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func @erf_fastmath_vector
func.func @erf_fastmath_vector(%arg0: vector<4xf32>) -> vector<4xf32> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_triple = "amdgcn-amd-amdhsa"}>
} {
  // CHECK-NOT: math.erf

  // CHECK-DAG: arith.constant dense<1.000000e+00> : vector<4xf32>

  // Then region coefficients (|x| < 1.0)
  // CHECK-DAG: arith.constant dense<0.128689408> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.634933292> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.106668264> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-0.0241816975> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.00385781587> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-3.78677854E-4> : vector<4xf32>

  // Else region coefficients (|x| >= 1.0)
  // CHECK-DAG: arith.constant dense<1.69988107E-5> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.128379107> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-0.376122952> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.112800106> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-0.0267075151> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.00491381623> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-5.61801775E-4> : vector<4xf32>

  // Structure checks for vector implementation.
  // CHECK: %[[AX:.*]] = math.absf %arg0 : vector<4xf32>
  // CHECK: %[[CMP:.*]] = arith.cmpf olt, %[[AX]], %{{.*}} : vector<4xf32>
  // CHECK: %[[T:.*]] = arith.mulf %[[AX]], %[[AX]] : vector<4xf32>

  // First polynomial evaluation (using t = ax * ax).
  // CHECK-COUNT-5: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %[[AX]] : vector<4xf32>

  // Second polynomial evaluation (using ax directly).
  // CHECK-COUNT-6: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %[[AX]] : vector<4xf32>

  // CHECK: %{{.*}} = arith.negf %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = math.exp %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} : vector<4xf32>

  // CHECK: %{{.*}} = arith.select %[[CMP]], %{{.*}}, %{{.*}} : vector<4xi1>, vector<4xf32>
  // CHECK: %{{.*}} = math.copysign %{{.*}}, %arg0 : vector<4xf32>
  %0 = math.erf %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @no_approx_on_rocm
func.func @no_approx_on_rocm(%arg0: f16) -> f16 attributes {
  hal.executable.target =  #hal.executable.target<"rocm", "rocm-hsaco-fb", {}>
} {
  // On ROCm, we want to use the native device library functions.
  // It's OK for f16 to still get casted to f32, as
  // the device library functions for f16 are casting to f32 anyway.
  // math.erf will be expanded to the above lit-test. (func.func @erf_fastmath)
  // CHECK:         math.acos
  // CHECK:         math.atan
  // CHECK:         math.sin
  // CHECK:         math.tanh
  // CHECK:         math.log
  // CHECK:         math.log2
  // CHECK:         math.log1p
  // CHECK:         math.exp
  // CHECK:         math.exp2
  // CHECK:         math.expm1
  // CHECK:         math.cbrt
  %0 = math.acos %arg0 : f16
  %1 = math.atan %0 : f16
  %2 = math.sin %1 : f16
  %3 = math.tanh %2 : f16
  %4 = math.log %3 : f16
  %5 = math.log2 %4 : f16
  %6 = math.log1p %5 : f16
  %7 = math.exp %6 : f16
  %8 = math.exp2 %7 : f16
  %9 = math.expm1 %8 : f16
  %10 = math.cbrt %9 : f16
  %11 = math.erf %10 : f16
  return %11 : f16
}

// -----

// CHECK-LABEL: @rewrite_fpowi_const_exponent_on_llvmcpu
func.func @rewrite_fpowi_const_exponent_on_llvmcpu(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} {
  // math.fpowi with constant exponent should always be rewritten to muls.
  // CHECK-NOT:    math.fpowi
  // CHECK:        arith.mulf {{.*}} : f32
  %c2 = arith.constant 2 : i32
  %0 = math.fpowi %arg0, %c2 : f32, i32
  return %0 : f32
}

// -----

// CHECK-LABEL: @rewrite_fpowi_const_exponent_on_rocm
func.func @rewrite_fpowi_const_exponent_on_rocm(%arg0: f32) -> f32 attributes {
  hal.executable.target =  #hal.executable.target<"rocm", "rocm-hsaco-fb", {}>
} {
  // math.fpowi with constant exponent should always be rewritten to muls.
  // CHECK-NOT:     math.fpowi
  // CHECK-COUNT-2: arith.mulf {{.*}} : f32
  %c3 = arith.constant 3 : i32
  %0 = math.fpowi %arg0, %c3 : f32, i32
  return %0 : f32
}

// -----

// CHECK-LABEL: @roundeven_native_on_aarch64
func.func @roundeven_native_on_aarch64(%arg0: vector<4xf32>) -> vector<4xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "aarch64-xyz-xyz"}>
} {
  // CHECK: math.roundeven {{.*}} : vector<4xf32>
  %0 = math.roundeven %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @roundeven_native_on_x86_sse41
func.func @roundeven_native_on_x86_sse41(%arg0: vector<8xf32>) -> vector<8xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+sse4.1", target_triple = "x86_64-xyz-xyz"}>
} {
  // CHECK: math.roundeven {{.*}} : vector<8xf32>
  %0 = math.roundeven %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: @roundeven_native_on_x86_avx2_only
func.func @roundeven_native_on_x86_avx2_only(%arg0: vector<8xf32>) -> vector<8xf32> attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+avx2", target_triple = "x86_64-xyz-xyz"}>
} {
  // CHECK: math.roundeven {{.*}} : vector<8xf32>
  %0 = math.roundeven %arg0 : vector<8xf32>
  return %0 : vector<8xf32>
}

// -----

// CHECK-LABEL: @roundeven_expanded_on_x86_without_sse41
func.func @roundeven_expanded_on_x86_without_sse41(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+sse2", target_triple = "x86_64-xyz-xyz"}>
} {
  // CHECK-NOT: math.roundeven
  // CHECK-DAG: arith.constant 1.000000e+00 : f32
  // CHECK-DAG: arith.constant 23 : i32
  // CHECK-DAG: arith.constant 127 : i32
  // CHECK-DAG: arith.constant 255 : i32
  // CHECK-DAG: arith.constant 8388607 : i32
  // CHECK-DAG: arith.constant 4194304 : i32
  // CHECK:     arith.bitcast %arg0 : f32 to i32
  // CHECK:     %[[ROUND:.*]] = math.round %arg0 : f32
  // CHECK:     arith.bitcast %[[ROUND]] : f32 to i32
  // CHECK:     %[[SIGN:.*]] = math.copysign %{{.*}}, %arg0 : f32
  // CHECK:     %[[SHIFTED:.*]] = arith.subf %[[ROUND]], %[[SIGN]] : f32
  // CHECK:     %[[SELECT:.*]] = arith.select %{{.*}}, %[[SHIFTED]], %[[ROUND]] : f32
  // CHECK:     math.copysign %[[SELECT]], %arg0 : f32
  %0 = math.roundeven %arg0 : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @roundeven_native_on_riscv64_d
func.func @roundeven_native_on_riscv64_d(%arg0: f64) -> f64 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+m,+a,+f,+d,+c,+v", target_triple = "riscv64-xyz-xyz"}>
} {
  // CHECK: math.roundeven {{.*}} : f64
  %0 = math.roundeven %arg0 : f64
  return %0 : f64
}

// -----

// CHECK-LABEL: @roundeven_expanded_on_riscv64_without_d
func.func @roundeven_expanded_on_riscv64_without_d(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+m,+a,+f,+c", target_triple = "riscv64-xyz-xyz"}>
} {
  // CHECK-NOT: math.roundeven
  // CHECK-DAG: arith.constant 1.000000e+00 : f32
  // CHECK-DAG: arith.constant 23 : i32
  // CHECK-DAG: arith.constant 127 : i32
  // CHECK-DAG: arith.constant 255 : i32
  // CHECK-DAG: arith.constant 8388607 : i32
  // CHECK-DAG: arith.constant 4194304 : i32
  // CHECK:     arith.bitcast %arg0 : f32 to i32
  // CHECK:     %[[ROUND:.*]] = math.round %arg0 : f32
  // CHECK:     arith.bitcast %[[ROUND]] : f32 to i32
  // CHECK:     %[[SIGN:.*]] = math.copysign %{{.*}}, %arg0 : f32
  // CHECK:     %[[SHIFTED:.*]] = arith.subf %[[ROUND]], %[[SIGN]] : f32
  // CHECK:     %[[SELECT:.*]] = arith.select %{{.*}}, %[[SHIFTED]], %[[ROUND]] : f32
  // CHECK:     math.copysign %[[SELECT]], %arg0 : f32
  %0 = math.roundeven %arg0 : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: @roundeven_expanded_on_riscv32
func.func @roundeven_expanded_on_riscv32(%arg0: f64) -> f64 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {cpu_features = "+m,+a,+f,+d,+c", target_triple = "riscv32-xyz-xyz"}>
} {
  // CHECK-NOT: math.roundeven
  // CHECK-DAG: arith.constant 1.000000e+00 : f64
  // CHECK-DAG: arith.constant 52 : i64
  // CHECK-DAG: arith.constant 1023 : i64
  // CHECK-DAG: arith.constant 2047 : i64
  // CHECK-DAG: arith.constant 4503599627370495 : i64
  // CHECK-DAG: arith.constant 2251799813685248 : i64
  // CHECK:     arith.bitcast %arg0 : f64 to i64
  // CHECK:     %[[ROUND:.*]] = math.round %arg0 : f64
  // CHECK:     arith.bitcast %[[ROUND]] : f64 to i64
  // CHECK:     %[[SIGN:.*]] = math.copysign %{{.*}}, %arg0 : f64
  // CHECK:     %[[SHIFTED:.*]] = arith.subf %[[ROUND]], %[[SIGN]] : f64
  // CHECK:     %[[SELECT:.*]] = arith.select %{{.*}}, %[[SHIFTED]], %[[ROUND]] : f64
  // CHECK:     math.copysign %[[SELECT]], %arg0 : f64
  %0 = math.roundeven %arg0 : f64
  return %0 : f64
}

// -----

// CHECK-LABEL: @roundeven_expanded_without_target
func.func @roundeven_expanded_without_target(%arg0: f32) -> f32 {
  // CHECK-NOT: math.roundeven
  // CHECK-DAG: arith.constant 1.000000e+00 : f32
  // CHECK-DAG: arith.constant 23 : i32
  // CHECK-DAG: arith.constant 127 : i32
  // CHECK-DAG: arith.constant 255 : i32
  // CHECK-DAG: arith.constant 8388607 : i32
  // CHECK-DAG: arith.constant 4194304 : i32
  // CHECK:     arith.bitcast %arg0 : f32 to i32
  // CHECK:     %[[ROUND:.*]] = math.round %arg0 : f32
  // CHECK:     arith.bitcast %[[ROUND]] : f32 to i32
  // CHECK:     %[[SIGN:.*]] = math.copysign %{{.*}}, %arg0 : f32
  // CHECK:     %[[SHIFTED:.*]] = arith.subf %[[ROUND]], %[[SIGN]] : f32
  // CHECK:     %[[SELECT:.*]] = arith.select %{{.*}}, %[[SHIFTED]], %[[ROUND]] : f32
  // CHECK:     math.copysign %[[SELECT]], %arg0 : f32
  %0 = math.roundeven %arg0 : f32
  return %0 : f32
}
