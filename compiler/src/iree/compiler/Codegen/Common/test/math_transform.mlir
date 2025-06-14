// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-codegen-math-transform))' --split-input-file %s | FileCheck %s

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
  // CHECK-DAG: arith.constant -5.61801775E-4 : f32
  // CHECK-DAG: arith.constant 0.00491381623 : f32
  // CHECK-DAG: arith.constant -0.0267075151 : f32
  // CHECK-DAG: arith.constant 0.112800106 : f32
  // CHECK-DAG: arith.constant -0.376122952 : f32
  // CHECK-DAG: arith.constant 0.128379107 : f32

  // Else region coefficients (|x| >= 1.0)
  // CHECK-DAG: arith.constant 1.69988107E-5 : f32
  // CHECK-DAG: arith.constant -3.78677854E-4 : f32
  // CHECK-DAG: arith.constant 0.00385781587 : f32
  // CHECK-DAG: arith.constant -0.0241816975 : f32
  // CHECK-DAG: arith.constant 0.106668264 : f32
  // CHECK-DAG: arith.constant 0.634933292 : f32
  // CHECK-DAG: arith.constant 0.128689408 : f32

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
  // CHECK-NOT:    math.fpowi
  // CHECK:        arith.mulf {{.*}} : f32
  // CHECK:        arith.mulf {{.*}} : f32
  %c2 = arith.constant 3 : i32
  %0 = math.fpowi %arg0, %c2 : f32, i32
  return %0 : f32
}
