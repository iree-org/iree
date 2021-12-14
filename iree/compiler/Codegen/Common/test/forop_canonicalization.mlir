// RUN: iree-opt %s -split-input-file -iree-codegen-fold-across-scf-region | FileCheck %s

func @loop_carried_vector_shape_cast(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
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

// CHECK-LABEL:   func @loop_carried_vector_shape_cast
//   CHECK-NOT:     vector.shape_cast
//       CHECK:     scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-NOT:       vector.shape_cast
//       CHECK:       scf.yield {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:     }
//   CHECK-NOT:     vector.shape_cast
//       CHECK:     return {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>

// -----

func @loop_carried_unrealized_conversion_cast(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
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

// CHECK-LABEL:   func @loop_carried_unrealized_conversion_cast
//   CHECK-NOT:     unrealized_conversion_cast
//       CHECK:     scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-NOT:       unrealized_conversion_cast
//       CHECK:       scf.yield {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:     }
//   CHECK-NOT:     unrealized_conversion_cast
//       CHECK:     return {{.*}}, {{.*}} : vector<4xf32>, vector<4xf32>

// -----

func @loop_carried_extract(%arg0: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = vector.broadcast %arg0 : f32 to vector<4xf32>
  %20 = scf.for %arg3 = %c0 to %c10 step %c1 iter_args(%arg4 = %0) -> (vector<4xf32>) {
    %a = vector.extract %arg4[0] : vector<4xf32>
    %c = arith.addf %a, %a : f32
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

// -----

func @loop_pack_v8f16(%arg0: vector<8xf16>, %arg1: vector<8xf16>, %arg2: vector<4xf16>)
                  -> (vector<8xf16>, vector<8xf16>, vector<4xf16>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  %0:3 = scf.for %iv = %c0 to %c10 step %c1
                 iter_args(%forarg0 = %arg0, %forarg1 = %arg1, %forarg2 = %arg2)
              -> (vector<8xf16>, vector<8xf16>, vector<4xf16>) {
    %add = arith.addf %forarg0, %forarg1: vector<8xf16>
    scf.yield %add, %forarg1, %forarg2: vector<8xf16>, vector<8xf16>, vector<4xf16>
  }

  return %0#0, %0#1, %0#2 : vector<8xf16>, vector<8xf16>, vector<4xf16>
}

// CHECK-LABEL: func @loop_pack_v8f16
//  CHECK-SAME: (%[[ARG0:.+]]: vector<8xf16>, %[[ARG1:.+]]: vector<8xf16>, %[[ARG2:.+]]: vector<4xf16>)
//       CHECK:    %[[CAST_ARG0:.+]] = vector.bitcast %[[ARG0]] : vector<8xf16> to vector<4xf32>
//       CHECK:    %[[CAST_ARG1:.+]] = vector.bitcast %[[ARG1]] : vector<8xf16> to vector<4xf32>
//       CHECK:    %[[FOR:.+]]:3 = scf.for %{{.+}} = %{{.+}} to %{{.+}} step %{{.+}} iter_args(%[[FOR_ARG0:.+]] = %[[CAST_ARG0]], %[[FOR_ARG1:.+]] = %[[CAST_ARG1]], %[[FOR_ARG2:.+]] = %[[ARG2]]) -> (vector<4xf32>, vector<4xf32>, vector<4xf16>) {
//       CHECK:      %[[CAST_FOR_ARG0:.+]] = vector.bitcast %[[FOR_ARG0]] : vector<4xf32> to vector<8xf16>
//       CHECK:      %[[CAST_FOR_ARG1:.+]] = vector.bitcast %[[FOR_ARG1]] : vector<4xf32> to vector<8xf16>
//       CHECK:      %[[ADD:.+]] = arith.addf %[[CAST_FOR_ARG0]], %[[CAST_FOR_ARG1]] : vector<8xf16>
//       CHECK:      %[[CAST_ADD:.+]] = vector.bitcast %[[ADD]] : vector<8xf16> to vector<4xf32>
//       CHECK:      scf.yield %[[CAST_ADD]], %[[FOR_ARG1]], %[[FOR_ARG2]] : vector<4xf32>, vector<4xf32>, vector<4xf16>
//       CHECK:    }
//       CHECK:    %[[CAST_FOR0:.+]] = vector.bitcast %[[FOR]]#0 : vector<4xf32> to vector<8xf16>
//       CHECK:    %[[CAST_FOR1:.+]] = vector.bitcast %[[FOR]]#1 : vector<4xf32> to vector<8xf16>
//       CHECK:    return %[[CAST_FOR0]], %[[CAST_FOR1]], %[[FOR]]#2 : vector<8xf16>, vector<8xf16>, vector<4xf16>

// -----

func @if_result_extract(%cond: i1, %v0: f32, %v1: vector<4xf32>, %v2: vector<3xf32>) -> (f32, f32, f32) {
  %c0 = arith.constant dense<0.0> : vector<1x4xf32>
  %c1 = arith.constant dense<0.0> : vector<4xf32>
  %c2 = arith.constant dense<0.0> : vector<1x3xf32>
  %0:3 = scf.if %cond -> (vector<1x4xf32>, vector<4xf32>, vector<1x3xf32>) {
    %1 = vector.broadcast %v0 : f32 to vector<1x4xf32>
    %2 = vector.broadcast %v2 : vector<3xf32> to vector<1x3xf32>
    scf.yield %1, %v1, %2 : vector<1x4xf32>, vector<4xf32>, vector<1x3xf32>
  } else {
    scf.yield %c0, %c1, %c2 : vector<1x4xf32>, vector<4xf32>, vector<1x3xf32>
  }
  %3 = vector.extract %0#0[0, 0] : vector<1x4xf32>
  %4 = vector.extract %0#1[0] : vector<4xf32>
  %5 = vector.extract %0#2[0, 0] : vector<1x3xf32>
  return %3, %4, %5: f32, f32, f32
}


// CHECK-LABEL: func @if_result_extract
//  CHECK-SAME: (%[[COND:.+]]: i1, %[[V0:.+]]: f32, %[[V1:.+]]: vector<4xf32>, %[[V2:.+]]: vector<3xf32>)
//   CHECK-DAG:   %[[CST0:.+]] = arith.constant dense<0.000000e+00> : vector<1x4xf32>
//   CHECK-DAG:   %[[CST1:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//   CHECK-DAG:   %[[CST2:.+]] = arith.constant dense<0.000000e+00> : vector<1x3xf32>
//       CHECK:   %[[IF:.+]]:3 = scf.if %[[COND]] -> (f32, vector<4xf32>, f32) {
//  CHECK-NEXT:     %[[V2_0:.+]] = vector.extract %[[V2]][0] : vector<3xf32>
//  CHECK-NEXT:     scf.yield %[[V0]], %[[V1]], %[[V2_0]]
//  CHECK-NEXT:   } else {
//  CHECK-NEXT:     %[[V0_0:.+]] = vector.extract %[[CST0]][0, 0] : vector<1x4xf32>
//  CHECK-NEXT:     %[[V2_0:.+]] = vector.extract %[[CST2]][0, 0] : vector<1x3xf32>
//  CHECK-NEXT:     scf.yield %[[V0_0]], %[[CST1]], %[[V2_0]]
//  CHECK-NEXT:   }
//       CHECK:   %[[EXTRACT1:.+]] = vector.extract %[[IF]]#1[0] : vector<4xf32>
//       CHECK:   return %[[IF]]#0, %[[EXTRACT1]], %[[IF]]#2
