// RUN: iree-opt --split-input-file --iree-spirv-vectorize -canonicalize %s | FileCheck %s

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
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[INIT:.+]] = arith.constant dense<0> : vector<16xi8>
// CHECK:       %[[IND:.+]] = vector.extract %[[INDEX_VEC]][0] : vector<16xindex>
// CHECK:       %[[LOAD:.+]] = vector.load %[[ARG0]][%[[C0]], %[[C0]], %[[IND]]] : memref<16x1082x1922xi8>, vector<1xi8>
// CHECK:       %[[EXTRACT:.+]] = vector.extract %[[LOAD]][0] : vector<1xi8>
// CHECK:       %[[INSERT:.+]] = vector.insert %[[EXTRACT]], %[[INIT]] [0] : i8 into vector<16xi8>
// CHECK-15:    vector.load %[[ARG0]][%[[C0]], %[[C0]], %{{.*}}] : memref<16x1082x1922xi8>, vector<1xi8>

