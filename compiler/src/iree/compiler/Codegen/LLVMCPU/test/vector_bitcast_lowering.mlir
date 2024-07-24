// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-llvmcpu-vector-bitcast-lowering))' --split-input-file %s | FileCheck %s

func.func @vector_bitcast_2d(%arg0: vector<2x16xi8>) -> vector<2x2xi64> {
  %0 = vector.bitcast %arg0 : vector<2x16xi8> to vector<2x2xi64>
  return %0 : vector<2x2xi64>
}
// CHECK-LABEL: func.func @vector_bitcast_2d
// CHECK-SAME:    %[[IN:[a-zA-Z0-9]+]]
// CHECK:         %[[INIT:.+]] = arith.constant {{.+}} : vector<2x2xi64>
// CHECK:         %[[V1:.+]] = vector.extract %[[IN]][0] : vector<16xi8> from vector<2x16xi8>
// CHECK:         %[[B1:.+]] = vector.bitcast %[[V1]] : vector<16xi8> to vector<2xi64>
// CHECK:         %[[R1:.+]] = vector.insert %[[B1]], %[[INIT]] [0]
// CHECK:         %[[V2:.+]] = vector.extract %[[IN]][1] : vector<16xi8> from vector<2x16xi8>
// CHECK:         %[[B2:.+]] = vector.bitcast %[[V2]] : vector<16xi8> to vector<2xi64>
// CHECK:         %[[R2:.+]] = vector.insert %[[B2]], %[[R1]] [1]
// CHECK:         return %[[R2]]
