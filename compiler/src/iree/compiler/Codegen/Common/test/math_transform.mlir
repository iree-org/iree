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

// CHECK-LABEL: @no_approx_erf_on_rocm
func.func @no_approx_erf_on_rocm(%arg0: f16) -> f16 attributes {
  hal.executable.target =  #hal.executable.target<"rocm", "rocm-hsaco-fb", {}>
} {
  // On ROCm, we want to use the native device library function, so math.erf
  // should not get rewritten. It's OK for f16 to still get casted to f32, as
  // the device library function for f16 is casting to f32 anyway.
  // CHECK:         math.erf
  // CHECK-NOT:     math.exp
  // CHECK-NOT:     math.log
  // CHECK-NOT:     math.fma
  %0 = math.erf %arg0 : f16
  return %0 : f16
}
