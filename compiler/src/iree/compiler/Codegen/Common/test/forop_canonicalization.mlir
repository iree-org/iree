// RUN: iree-opt %s --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-canonicalize-scf-for),canonicalize)" | FileCheck %s

func.func @loop_carried_vector_shape_cast(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = vector.shape_cast %arg0 : vector<4xf32> to vector<1x4xf32>
  %1 = vector.shape_cast %arg1 : vector<4xf32> to vector<1x4xf32>
  %20:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0, %arg5 = %1) -> (vector<1x4xf32>, vector<1x4xf32>) {
    %a = vector.shape_cast %arg4 : vector<1x4xf32> to vector<4xf32>
    %b = vector.shape_cast %arg5 : vector<1x4xf32> to vector<4xf32>
    %c = arith.addf %a, %b : vector<4xf32>
    %d = arith.mulf %a, %b : vector<4xf32>
    %cc = vector.shape_cast %c : vector<4xf32> to vector<1x4xf32>
    %dc = vector.shape_cast %d : vector<4xf32> to vector<1x4xf32>
    scf.yield %cc, %dc : vector<1x4xf32>, vector<1x4xf32>
  }
  %21 = vector.shape_cast %20#0 : vector<1x4xf32> to vector<4xf32>
  %22 = vector.shape_cast %20#1 : vector<1x4xf32> to vector<4xf32>
  return %21, %22 : vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL:   func.func @loop_carried_vector_shape_cast
//   CHECK-NOT:     vector.shape_cast
//       CHECK:     scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-NOT:       vector.shape_cast
//       CHECK:       scf.yield {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:     }
//   CHECK-NOT:     vector.shape_cast
//       CHECK:     return {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>

// -----

func.func @loop_carried_unrealized_conversion_cast(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = builtin.unrealized_conversion_cast %arg0 : vector<4xf32> to vector<1x4xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : vector<4xf32> to vector<1x4xf32>
  %20:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0, %arg5 = %1) -> (vector<1x4xf32>, vector<1x4xf32>) {
    %a = builtin.unrealized_conversion_cast %arg4 : vector<1x4xf32> to vector<4xf32>
    %b = builtin.unrealized_conversion_cast %arg5 : vector<1x4xf32> to vector<4xf32>
    %c = arith.addf %a, %b : vector<4xf32>
    %d = arith.mulf %a, %b : vector<4xf32>
    %cc = builtin.unrealized_conversion_cast %c : vector<4xf32> to vector<1x4xf32>
    %dc = builtin.unrealized_conversion_cast %d : vector<4xf32> to vector<1x4xf32>
    scf.yield %cc, %dc : vector<1x4xf32>, vector<1x4xf32>
  }
  %21 = builtin.unrealized_conversion_cast %20#0 : vector<1x4xf32> to vector<4xf32>
  %22 = builtin.unrealized_conversion_cast %20#1 : vector<1x4xf32> to vector<4xf32>
  return %21, %22 : vector<4xf32>, vector<4xf32>
}

// CHECK-LABEL:   func.func @loop_carried_unrealized_conversion_cast
//   CHECK-NOT:     unrealized_conversion_cast
//       CHECK:     scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-NOT:       unrealized_conversion_cast
//       CHECK:       scf.yield {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:     }
//   CHECK-NOT:     unrealized_conversion_cast
//       CHECK:     return {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>

// -----

func.func @loop_carried_extract(%arg0: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %20 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0) -> (vector<4xf32>) {
    %a = vector.extract %arg4[0] : f32 from vector<4xf32>
    %c = arith.addf %a, %a : f32
    %bc = vector.broadcast %c : f32 to vector<4xf32>
    scf.yield %bc : vector<4xf32>
  }
  %21 = vector.extract %20[0] : f32 from vector<4xf32>
  return %21 : f32
}

// CHECK-LABEL:   func.func @loop_carried_extract
//   CHECK-NOT:     vector.broadcast
//       CHECK:     scf.for {{.*}} -> (f32) {
//   CHECK-NOT:       vector.extract
//   CHECK-NOT:     vector.broadcast
//       CHECK:       scf.yield {{.*}} : f32
//       CHECK:     }
//   CHECK-NOT:     vector.extract
//       CHECK:     return {{.*}} : f32

// -----

func.func @loop_pack_v8f16(%arg0: vector<8xf16>, %arg1: vector<8xf16>, %arg2: vector<4xf16>)
                  -> (vector<8xf16>, vector<8xf16>, vector<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<1.0> : vector<4xf16>

  %0:3 = scf.for %iv = %c0 to %c10 step %c1
                 iter_args(%forarg0 = %arg0, %forarg1 = %arg1, %forarg2 = %arg2)
              -> (vector<8xf16>, vector<8xf16>, vector<4xf16>) {
    %add = arith.addf %forarg0, %forarg1: vector<8xf16>
    %inc = arith.addf %forarg2, %cst: vector<4xf16>
    scf.yield %forarg1, %add, %inc: vector<8xf16>, vector<8xf16>, vector<4xf16>
  }

  return %0#0, %0#1, %0#2 : vector<8xf16>, vector<8xf16>, vector<4xf16>
}

// CHECK-LABEL: func.func @loop_pack_v8f16
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8xf16>, %[[ARG1:.+]]: vector<8xf16>, %[[ARG2:.+]]: vector<4xf16>)
//       CHECK:    %[[CAST_ARG0:.+]] = vector.bitcast %[[ARG0]] : vector<8xf16> to vector<4xi32>
//       CHECK:    %[[CAST_ARG1:.+]] = vector.bitcast %[[ARG1]] : vector<8xf16> to vector<4xi32>
//       CHECK:    %[[FOR:.+]]:3 = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[FOR_ARG0:.+]] = %[[CAST_ARG0]], %[[FOR_ARG1:.+]] = %[[CAST_ARG1]], %[[FOR_ARG2:.+]] = %[[ARG2]]) -> (vector<4xi32>, vector<4xi32>, vector<4xf16>) {
//       CHECK:      %[[CAST_FOR_ARG0:.+]] = vector.bitcast %[[FOR_ARG0]] : vector<4xi32> to vector<8xf16>
//       CHECK:      %[[CAST_FOR_ARG1:.+]] = vector.bitcast %[[FOR_ARG1]] : vector<4xi32> to vector<8xf16>
//       CHECK:      %[[ADD:.+]] = arith.addf %[[CAST_FOR_ARG0]], %[[CAST_FOR_ARG1]] : vector<8xf16>
//       CHECK:      %[[INC:.+]] = arith.addf %[[FOR_ARG2]], {{.*}} : vector<4xf16>
//       CHECK:      %[[CAST_ADD:.+]] = vector.bitcast %[[ADD]] : vector<8xf16> to vector<4xi32>
//       CHECK:      scf.yield %[[FOR_ARG1]], %[[CAST_ADD]], %[[INC]] : vector<4xi32>, vector<4xi32>, vector<4xf16>
//       CHECK:    }
//       CHECK:    %[[CAST_FOR0:.+]] = vector.bitcast %[[FOR]]#0 : vector<4xi32> to vector<8xf16>
//       CHECK:    %[[CAST_FOR1:.+]] = vector.bitcast %[[FOR]]#1 : vector<4xi32> to vector<8xf16>
//       CHECK:    return %[[CAST_FOR0]], %[[CAST_FOR1]], %[[FOR]]#2 : vector<8xf16>, vector<8xf16>, vector<4xf16>

// -----

func.func @loop_pack_v1x8i4(%arg0: vector<1x8xi4>, %arg1: vector<1x8xf16>)
                  -> (vector<1x8xi4>, vector<1x8xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %cst = arith.constant dense<1> : vector<1x8xi4>

  %0:2 = scf.for %iv = %c0 to %c10 step %c1
                 iter_args(%forarg0 = %arg0, %forarg1 = %arg1)
              -> (vector<1x8xi4>, vector<1x8xf16>) {
    %fp = arith.uitofp %forarg0 : vector<1x8xi4> to vector<1x8xf16>
    %add = arith.addf %fp, %forarg1: vector<1x8xf16>
    %addi4 = arith.addi %forarg0, %cst: vector<1x8xi4>
    scf.yield %addi4, %add : vector<1x8xi4>, vector<1x8xf16>
  }

  return %0#0, %0#1 : vector<1x8xi4>, vector<1x8xf16>
}

// CHECK-LABEL: func.func @loop_pack_v1x8i4
//  CHECK-SAME: (%[[ARG0:.+]]: vector<1x8xi4>, %[[ARG1:.+]]: vector<1x8xf16>)

//  CHECK:      vector.shape_cast %[[ARG0]] : vector<1x8xi4> to vector<8xi4>
//  CHECK:      vector.bitcast {{.*}} : vector<8xi4> to vector<1xi32>
//  CHECK:      vector.shape_cast %[[ARG1]] : vector<1x8xf16> to vector<8xf16>
//  CHECK:      vector.bitcast {{.*}} : vector<8xf16> to vector<4xi32>
//  CHECK:      %[[FOR:.+]]:2 = scf.for {{.*}} iter_args(%[[ARG3:.+]] = {{.*}}, %[[ARG4:.+]] = {{.*}}) -> (vector<1xi32>, vector<4xi32>) {
//  CHECK:        vector.bitcast %[[ARG3]] : vector<1xi32> to vector<8xi4>
//  CHECK:        vector.shape_cast {{.*}} : vector<8xi4> to vector<1x8xi4>
//  CHECK:        vector.bitcast %[[ARG4]] : vector<4xi32> to vector<8xf16>
//  CHECK:        vector.shape_cast {{.*}} : vector<8xf16> to vector<1x8xf16>

//  CHECK:        vector.shape_cast {{.*}} : vector<1x8xi4> to vector<8xi4>
//  CHECK:        vector.bitcast {{.*}} : vector<8xi4> to vector<1xi32>
//  CHECK:        vector.shape_cast {{.*}} : vector<1x8xf16> to vector<8xf16>
//  CHECK:        vector.bitcast {{.*}} : vector<8xf16> to vector<4xi32>
//  CHECK:        scf.yield {{.*}} : vector<1xi32>, vector<4xi32>
//  CHECK:      }
//  CHECK:      vector.bitcast {{.*}} : vector<1xi32> to vector<8xi4>
//  CHECK:      vector.shape_cast {{.*}} : vector<8xi4> to vector<1x8xi4>
//  CHECK:      vector.bitcast {{.*}} : vector<4xi32> to vector<8xf16>
//  CHECK:      vector.shape_cast {{.*}} : vector<8xf16> to vector<1x8xf16>
//  CHECK:      return {{.*}} : vector<1x8xi4>, vector<1x8xf16>

// -----

func.func @pipelined_loop_extract(%arg0: f32, %arg1: f32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %1 = vector.broadcast %arg1 : f32 to vector<4xf32>
  %20:2 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0, %arg5 = %1) -> (vector<4xf32>, vector<4xf32>) {
    %a = vector.extract %arg4[0] : f32 from vector<4xf32>
    %c = arith.addf %a, %a : f32
    %bc = vector.broadcast %c : f32 to vector<4xf32>
    scf.yield %arg5, %bc : vector<4xf32>, vector<4xf32>
  }
  %21 = vector.extract %20#0[0] : f32 from vector<4xf32>
  %22 = vector.extract %20#1[0] : f32 from vector<4xf32>
  return %21, %22 : f32, f32
}

// Check that we don't crash (and currently don't successfully fold).
// CHECK-LABEL:   func.func @pipelined_loop_extract
//       CHECK:     vector.broadcast
//       CHECK:     vector.broadcast
//       CHECK:     scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//       CHECK:       vector.extract
//       CHECK:       vector.broadcast
//       CHECK:       scf.yield {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:     }
//       CHECK:     vector.extract
//       CHECK:     vector.extract
//       CHECK:     return {{.*}} : f32, f32
