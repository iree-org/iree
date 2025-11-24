// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-replace-slow-min-max-ops))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @min
func.func @min(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  // CHECK: arith.minnumf %{{.+}}, %{{.+}} : vector<4xf32>
  %min = arith.minimumf %arg0, %arg1 : vector<4xf32>
  return %min : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @max
func.func @max(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
  // CHECK: arith.maxnumf %{{.+}}, %{{.+}} : vector<4xf32>
  %max = arith.maximumf %arg0, %arg1 : vector<4xf32>
  return %max : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @reduction_min
func.func @reduction_min(%arg0: vector<4xf32>) -> f32 {
  // CHECK: vector.reduction <minnumf>, %{{.+}} : vector<4xf32> into f32
  %reduction_min = vector.reduction <minimumf>, %arg0 : vector<4xf32> into f32
  return %reduction_min : f32
}

// -----

// CHECK-LABEL: func.func @reduction_max
func.func @reduction_max(%arg0: vector<4xf32>) -> f32 {
  // CHECK: vector.reduction <maxnumf>, %{{.+}} : vector<4xf32> into f32
  %reduction_max = vector.reduction <maximumf>, %arg0 : vector<4xf32> into f32
  return %reduction_max : f32
}

// -----

// CHECK-LABEL: func.func @multi_reduction_min
func.func @multi_reduction_min(%arg0: vector<4xf32>, %arg1: f32) -> f32 {
  // CHECK: vector.multi_reduction <minnumf>, %{{.+}}, %{{.+}} [0] : vector<4xf32> to f32
  %multi_reduction_min = vector.multi_reduction <minimumf>, %arg0, %arg1 [0] : vector<4xf32> to f32
  return %multi_reduction_min : f32
}

// -----

// CHECK-LABEL: func.func @multi_reduction_max
func.func @multi_reduction_max(%arg0: vector<4xf32>, %arg1: f32) -> f32 {
  // CHECK: vector.multi_reduction <maxnumf>, %{{.+}}, %{{.+}} [0] : vector<4xf32> to f32
  %multi_reduction_max = vector.multi_reduction <maximumf>, %arg0, %arg1 [0] : vector<4xf32> to f32
  return %multi_reduction_max : f32
}
