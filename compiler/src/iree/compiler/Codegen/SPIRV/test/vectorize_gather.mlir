// RUN: iree-opt --split-input-file --iree-spirv-vectorize -canonicalize %s | FileCheck %s

func.func @tensor_extract(%arg0: tensor<6x4xf32>, %arg1: tensor<6xi32>, %data: tensor<1x2x512xf32>, %init: tensor<6x4xf32>, %i : index, %j: index) -> tensor<6x4xf32> {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.000000e+00 : f32
  %generic = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0, %arg1 : tensor<6x4xf32>, tensor<6xi32>) outs(%init : tensor<6x4xf32>) {
  ^bb0(%in: f32, %in_2: i32, %out: f32):
    %0 = linalg.index 1 : index
    %1 = affine.apply affine_map<(d0, d1)[s0] -> (d0 + d1 + s0 * 32)>(%0, %j)[%i]
    %2 = arith.addf %out, %f0 : f32
    %3 = arith.index_cast %in_2 : i32 to index
    %extracted = tensor.extract %data[%c0, %3, %1] : tensor<1x2x512xf32>
    %4 = arith.addf %2, %in : f32
    %5 = arith.addf %4, %extracted : f32
    linalg.yield %5 : f32
  } -> tensor<6x4xf32>
  return %generic : tensor<6x4xf32>
}

//    CHECK-LABEL: func.func @tensor_extract
//      CHECK-NOT: linalg.generic
// CHECK-COUNT-24: tensor.extract {{.+}} : tensor<1x2x512xf32>

// -----

func.func @vector_gather(%arg0: memref<16x1082x1922xi8>, %index_vec: vector<16xindex>) -> vector<16xi8> {
  %c0 = arith.constant 0 : index
  %mask = arith.constant dense<true> : vector<16xi1>
  %pass = arith.constant dense<0> : vector<16xi8>
  %0 = vector.gather %arg0[%c0, %c0, %c0] [%index_vec], %mask, %pass : memref<16x1082x1922xi8>, vector<16xindex>, vector<16xi1>, vector<16xi8> into vector<16xi8>
  return %0 : vector<16xi8>
}

// CHECK-LABEL: func.func @vector_gather
// CHECK-SAME:  %[[ARG0:.+]]: memref<16x1082x1922xi8>
// CHECK-SAME:  %[[INDEX_VEC:.+]]: vector<16xindex>
// CHECK-DAG:   %[[SLICE_INIT:.+]] = arith.constant dense<0> : vector<4xi8>
// CHECK-DAG:   %[[INIT:.+]] = arith.constant dense<0> : vector<16xi8>
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index

// CHECK:       %[[IND0:.+]] = vector.extract %[[INDEX_VEC]][0] : vector<16xindex>
// CHECK:       %[[LOAD0:.+]] = vector.load %[[ARG0]][%[[C0]], %[[C0]], %[[IND0]]] : memref<16x1082x1922xi8>, vector<1xi8>
// CHECK:       %[[EXTRACT0:.+]] = vector.extract %[[LOAD0]][0] : vector<1xi8>
// CHECK:       %[[INSERT0:.+]] = vector.insert %[[EXTRACT0]], %[[SLICE_INIT]] [0] : i8 into vector<4xi8>
// CHECK:       %[[IND1:.+]] = vector.extract %[[INDEX_VEC]][1] : vector<16xindex>
// CHECK:       %[[LOAD1:.+]] = vector.load %[[ARG0]][%[[C0]], %[[C0]], %[[IND1]]] : memref<16x1082x1922xi8>, vector<1xi8>
// CHECK:       %[[EXTRACT1:.+]] = vector.extract %[[LOAD1]][0] : vector<1xi8>
// CHECK:       %[[INSERT1:.+]] = vector.insert %[[EXTRACT1]], %[[INSERT0]] [1] : i8 into vector<4xi8>
// CHECK:       %[[IND2:.+]] = vector.extract %[[INDEX_VEC]][2] : vector<16xindex>
// CHECK:       %[[LOAD2:.+]] = vector.load %[[ARG0]][%[[C0]], %[[C0]], %[[IND2]]] : memref<16x1082x1922xi8>, vector<1xi8>
// CHECK:       %[[EXTRACT2:.+]] = vector.extract %[[LOAD2]][0] : vector<1xi8>
// CHECK:       %[[INSERT2:.+]] = vector.insert %[[EXTRACT2]], %[[INSERT1]] [2] : i8 into vector<4xi8>
// CHECK:       %[[IND3:.+]] = vector.extract %[[INDEX_VEC]][3] : vector<16xindex>
// CHECK:       %[[LOAD3:.+]] = vector.load %[[ARG0]][%[[C0]], %[[C0]], %[[IND3]]] : memref<16x1082x1922xi8>, vector<1xi8>
// CHECK:       %[[EXTRACT3:.+]] = vector.extract %[[LOAD3]][0] : vector<1xi8>
// CHECK:       %[[INSERT3:.+]] = vector.insert %[[EXTRACT3]], %[[INSERT2]] [3] : i8 into vector<4xi8>

// CHECK:       vector.insert_strided_slice %[[INSERT3]], %[[INIT]] {offsets = [0], strides = [1]} : vector<4xi8> into vector<16xi8>
// CHECK-12:    vector.load %[[ARG0]][%[[C0]], %[[C0]], %{{.*}}] : memref<16x1082x1922xi8>, vector<1xi8>
