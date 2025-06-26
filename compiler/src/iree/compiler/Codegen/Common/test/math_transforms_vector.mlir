// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-codegen-math-transform))' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @erf_fastmath_vector
func.func @erf_fastmath_vector(%arg0: vector<4xf32>) -> vector<4xf32> attributes {
  hal.executable.target = #hal.executable.target<"rocm", "rocm-hsaco-fb", {target_triple = "amdgcn-amd-amdhsa"}>
} {
  // CHECK-NOT: math.erf

  // CHECK-DAG: arith.constant dense<1.000000e+00> : vector<4xf32>

  // Then region coefficients (|x| < 1.0)
  // CHECK-DAG: arith.constant dense<-5.61801775E-4> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.00491381623> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-0.0267075151> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.112800106> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-0.376122952> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.128379107> : vector<4xf32>

  // Else region coefficients (|x| >= 1.0)
  // CHECK-DAG: arith.constant dense<1.69988107E-5> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-3.78677854E-4> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.00385781587> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<-0.0241816975> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.106668264> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.634933292> : vector<4xf32>
  // CHECK-DAG: arith.constant dense<0.128689408> : vector<4xf32>

  // Structure checks for vector implementation
  // CHECK: %[[AX:.*]] = math.absf %arg0 : vector<4xf32>
  // CHECK: %[[CMP:.*]] = arith.cmpf olt, %[[AX]], %{{.*}} : vector<4xf32>
  // CHECK: %[[T:.*]] = arith.mulf %[[AX]], %[[AX]] : vector<4xf32>

  // Then region - verify we have exactly 6 FMA operations for polynomial evaluation
  // CHECK-COUNT-5: %{{.*}} = math.fma %[[T]], %{{.*}}, %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = math.fma %[[AX]], %{{.*}}, %[[AX]] : vector<4xf32>
  
  // Else region - verify we have exactly 6 FMA operations
  // CHECK-COUNT-6: %{{.*}} = math.fma %[[AX]], %{{.*}}, %{{.*}} : vector<4xf32>

  // CHECK: %{{.*}} = arith.negf %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = math.exp %{{.*}} : vector<4xf32>
  // CHECK: %{{.*}} = arith.subf %{{.*}}, %{{.*}} : vector<4xf32>

  // CHECK: %{{.*}} = arith.select %[[CMP]], %{{.*}}, %{{.*}} : vector<4xi1>, vector<4xf32>
  // CHECK: %{{.*}} = math.copysign %{{.*}}, %arg0 : vector<4xf32>
  %0 = math.erf %arg0 : vector<4xf32>
  return %0 : vector<4xf32>
}
