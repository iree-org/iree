// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-math-transform{has-fast-exp})" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @erf_device_lib
func.func @erf_device_lib(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} {
  // CHECK-NOT: math.erf
  // CHECK: scf.if
  // CHECK: math.fma
  // CHECK: scf.yield
  // CHECK: else
  // CHECK: math.fma
  // CHECK: scf.yield
  // CHECK: math.copysign
  %0 = math.erf %arg0 : f32
  return %0 : f32
} 
