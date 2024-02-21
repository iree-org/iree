// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-optimize-vector-transfer))" | FileCheck %s

// CHECK-LABEL: func.func @transpose
//  CHECK-NEXT:   vector.shape_cast %{{.*}} : vector<1x1x4xf32> to vector<1x4x1xf32>
func.func @transpose(%arg0: vector<1x1x4xf32>) -> vector<1x4x1xf32> {
  %0 = vector.transpose %arg0, [0, 2, 1] : vector<1x1x4xf32> to vector<1x4x1xf32>
  return %0: vector<1x4x1xf32>
}
