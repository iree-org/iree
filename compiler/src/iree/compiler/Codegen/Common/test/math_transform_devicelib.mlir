// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-codegen-math-transform))' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @erf_device_lib
func.func @erf_device_lib(%arg0: f32) -> f32 attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_triple = "amdgcn-amd-amdhsa"}>
} {
  // CHECK-NOT: math.erf
  
  // Check for the expected coefficients (using CHECK-DAG for order independence)
  // Main constant
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
  
  // Then region - verify we have exactly 6 FMA operations
  // CHECK: %[[T:.*]] = arith.mulf %[[AX]], %[[AX]] : f32
  // CHECK: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %[[AX]] : f32
  // CHECK: scf.yield %{{.*}} : f32
  
  // CHECK: } else {
  
  // Else region - verify we have exactly 6 FMA operations
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : f32
  
  // CHECK: %{{.*}} = arith.negf %{{.*}} : f32
  // CHECK: %{{.*}} = math.exp %{{.*}} : f32
  // CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} : f32
  // CHECK: scf.yield %{{.*}} : f32
  // CHECK: }
  
  // CHECK: %{{.*}} = math.copysign %[[IF]], %arg0 : f32
  %0 = math.erf %arg0 : f32
  return %0 : f32
} 
