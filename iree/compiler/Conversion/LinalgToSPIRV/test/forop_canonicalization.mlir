// RUN: iree-opt %s -iree-codegen-forop-canonicalizatio-pass | FileCheck %s

func @loop_carried_cast(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = vector.shape_cast %arg0 : vector<4xf32> to vector<1x4xf32>
  %1 = vector.shape_cast %arg1 : vector<4xf32> to vector<1x4xf32>
  %20:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0, %arg5 = %1) -> (vector<1x4xf32>, vector<1x4xf32>) {
    %a = vector.shape_cast %arg4 : vector<1x4xf32> to vector<4xf32>
    %b = vector.shape_cast %arg5 : vector<1x4xf32> to vector<4xf32>
    %c = addf %a, %b : vector<4xf32>
    %d = mulf %a, %b : vector<4xf32>
    %cc = vector.shape_cast %c : vector<4xf32> to vector<1x4xf32>
    %dc = vector.shape_cast %d : vector<4xf32> to vector<1x4xf32>
    scf.yield %cc, %dc : vector<1x4xf32>, vector<1x4xf32>
  }
  %21 = vector.shape_cast %20#0 : vector<1x4xf32> to vector<4xf32>
  %22 = vector.shape_cast %20#1 : vector<1x4xf32> to vector<4xf32>
  return %21, %22 : vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL:   func @loop_carried_cast
//   CHECK-NOT:     vector.shape_cast
//       CHECK:     scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-NOT:       vector.shape_cast
//       CHECK:       scf.yield {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:     }
//   CHECK-NOT:     vector.shape_cast
//       CHECK:     return {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>

func @loop_carried_extract(%arg0: f32) -> f32 {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c10 = constant 10 : index
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %20 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0) -> (vector<4xf32>) {
    %a = vector.extract %arg4[0] : vector<4xf32>
    %c = addf %a, %a : f32
    %bc = vector.broadcast %c : f32 to vector<4xf32>
    scf.yield %bc : vector<4xf32>
  }
  %21 = vector.extract %20[0] : vector<4xf32>
  return %21 : f32
}

// CHECK-LABEL:   func @loop_carried_extract
//   CHECK-NOT:     vector.broadcast
//       CHECK:     scf.for {{.*}} -> (f32) {
//   CHECK-NOT:       vector.extract
//   CHECK-NOT:     vector.broadcast
//       CHECK:       scf.yield {{.*}} : f32
//       CHECK:     }
//   CHECK-NOT:     vector.extract
//       CHECK:     return {{.*}} : f32
