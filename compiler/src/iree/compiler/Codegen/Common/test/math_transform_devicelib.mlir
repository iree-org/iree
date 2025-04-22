// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-math-transform{has-fast-exp})" %s | FileCheck %s

// CHECK-LABEL: func.func @erf_device_lib
func.func @erf_device_lib(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz"}>
} { 
  %0 = math.erf %arg0 : f32
  return %0 : f32
} 