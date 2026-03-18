// RUN: iree-opt --pass-pipeline="builtin.module(iree-codegen-nd-to-1d-vector-unrolling)" --split-input-file %s | FileCheck %s

func.func @for_2d_vector(%init : vector<4x8xf32>, %lb : index, %ub : index, %step : index) -> vector<4x8xf32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> vector<4x8xf32> {
    scf.yield %arg : vector<4x8xf32>
  }
  return %result : vector<4x8xf32>
}

// CHECK-LABEL: func.func @for_2d_vector
// CHECK-SAME:    %[[INIT:.+]]: vector<4x8xf32>
// CHECK-DAG:   %[[E0:.+]] = vector.extract %[[INIT]][0] : vector<8xf32> from vector<4x8xf32>
// CHECK-DAG:   %[[E1:.+]] = vector.extract %[[INIT]][1] : vector<8xf32> from vector<4x8xf32>
// CHECK-DAG:   %[[E2:.+]] = vector.extract %[[INIT]][2] : vector<8xf32> from vector<4x8xf32>
// CHECK-DAG:   %[[E3:.+]] = vector.extract %[[INIT]][3] : vector<8xf32> from vector<4x8xf32>
// CHECK:       %[[FOR:.+]]:4 = scf.for {{.+}} iter_args(%[[A0:.+]] = %[[E0]], %[[A1:.+]] = %[[E1]], %[[A2:.+]] = %[[E2]], %[[A3:.+]] = %[[E3]])
// CHECK-SAME:    -> (vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>)
// CHECK:         scf.yield %[[A0]], %[[A1]], %[[A2]], %[[A3]] : vector<8xf32>, vector<8xf32>, vector<8xf32>, vector<8xf32>
// CHECK:       %[[POISON:.+]] = ub.poison : vector<4x8xf32>
// CHECK:       %[[I0:.+]] = vector.insert %[[FOR]]#0, %[[POISON]] [0] : vector<8xf32> into vector<4x8xf32>
// CHECK:       %[[I1:.+]] = vector.insert %[[FOR]]#1, %[[I0]] [1] : vector<8xf32> into vector<4x8xf32>
// CHECK:       %[[I2:.+]] = vector.insert %[[FOR]]#2, %[[I1]] [2] : vector<8xf32> into vector<4x8xf32>
// CHECK:       %[[I3:.+]] = vector.insert %[[FOR]]#3, %[[I2]] [3] : vector<8xf32> into vector<4x8xf32>
// CHECK:       return %[[I3]] : vector<4x8xf32>

// -----

func.func @for_3d_vector(%init : vector<2x3x4xf32>, %lb : index, %ub : index, %step : index) -> vector<2x3x4xf32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> vector<2x3x4xf32> {
    scf.yield %arg : vector<2x3x4xf32>
  }
  return %result : vector<2x3x4xf32>
}

// CHECK-LABEL: func.func @for_3d_vector
// CHECK-SAME:    %[[INIT:.+]]: vector<2x3x4xf32>
// CHECK-DAG:   vector.extract %[[INIT]][0, 0] : vector<4xf32> from vector<2x3x4xf32>
// CHECK-DAG:   vector.extract %[[INIT]][0, 1] : vector<4xf32> from vector<2x3x4xf32>
// CHECK-DAG:   vector.extract %[[INIT]][0, 2] : vector<4xf32> from vector<2x3x4xf32>
// CHECK-DAG:   vector.extract %[[INIT]][1, 0] : vector<4xf32> from vector<2x3x4xf32>
// CHECK-DAG:   vector.extract %[[INIT]][1, 1] : vector<4xf32> from vector<2x3x4xf32>
// CHECK-DAG:   vector.extract %[[INIT]][1, 2] : vector<4xf32> from vector<2x3x4xf32>
// CHECK:       %{{.+}}:6 = scf.for
// CHECK-SAME:    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>)
// CHECK:       ub.poison : vector<2x3x4xf32>
// CHECK-COUNT-6: vector.insert {{.+}} : vector<4xf32> into vector<2x3x4xf32>

// -----

func.func @nounroll_1d_vector(%init : vector<8xf32>, %lb : index, %ub : index, %step : index) -> vector<8xf32> {
  %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> vector<8xf32> {
    scf.yield %arg : vector<8xf32>
  }
  return %result : vector<8xf32>
}

// CHECK-LABEL: func.func @nounroll_1d_vector
// CHECK-SAME:    %[[INIT:.+]]: vector<8xf32>
// CHECK:       %[[FOR:.+]] = scf.for {{.+}} iter_args(%[[ARG:.+]] = %[[INIT]]) -> (vector<8xf32>)
// CHECK:         scf.yield %[[ARG]] : vector<8xf32>
// CHECK:       return %[[FOR]] : vector<8xf32>
// CHECK-NOT:   ub.poison
// CHECK-NOT:   vector.extract
// CHECK-NOT:   vector.insert

// -----

func.func @for_mixed_iter_args(%v : vector<2x4xf32>, %s : f32, %lb : index, %ub : index, %step : index) -> (vector<2x4xf32>, f32) {
  %result:2 = scf.for %iv = %lb to %ub step %step iter_args(%varg = %v, %sarg = %s) -> (vector<2x4xf32>, f32) {
    scf.yield %varg, %sarg : vector<2x4xf32>, f32
  }
  return %result#0, %result#1 : vector<2x4xf32>, f32
}

// CHECK-LABEL: func.func @for_mixed_iter_args
// CHECK-SAME:    %[[V:.+]]: vector<2x4xf32>, %[[S:.+]]: f32
// CHECK-DAG:   %[[E0:.+]] = vector.extract %[[V]][0] : vector<4xf32> from vector<2x4xf32>
// CHECK-DAG:   %[[E1:.+]] = vector.extract %[[V]][1] : vector<4xf32> from vector<2x4xf32>
// CHECK:       %{{.+}}:3 = scf.for {{.+}} iter_args(%{{.+}} = %[[E0]], %{{.+}} = %[[E1]], %{{.+}} = %[[S]])
// CHECK-SAME:    -> (vector<4xf32>, vector<4xf32>, f32)
