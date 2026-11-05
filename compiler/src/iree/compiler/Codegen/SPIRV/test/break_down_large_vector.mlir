// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-spirv-breakdown-large-vector))" %s | FileCheck %s

// CHECK-LABEL: func @extract_strided_slice_8_elements
func.func @extract_strided_slice_8_elements(%input: vector<8xf16>) -> vector<4xf16> {
  // CHECK-COUNT-4: vector.extract
  // CHECK:         vector.from_elements
  %0 = vector.extract_strided_slice %input offsets = [1], sizes = [4], strides = [1] : vector<8xf16> to vector<4xf16>
  return %0: vector<4xf16>
}

// -----

// CHECK-LABEL: func @extract_strided_slice_4_elements
func.func @extract_strided_slice_4_elements(%input: vector<4xf16>) -> vector<2xf16> {
  // CHECK: vector.extract_strided_slice
  %0 = vector.extract_strided_slice %input offsets = [1], sizes = [2], strides = [1] : vector<4xf16> to vector<2xf16>
  return %0: vector<2xf16>
}

// -----

// CHECK-LABEL: func @bitcast_16_elements
func.func @bitcast_16_elements(%input: vector<16xi8>) -> vector<4xi32> {
  // CHECK-DAG:     %[[CST_I32:.*]] = arith.constant dense<0> : vector<4xi32>
  // CHECK-COUNT-4: vector.extract
  // CHECK:         vector.from_elements
  // CHECK:         vector.bitcast %{{.*}} : vector<4xi8> to vector<1xi32>
  // CHECK:         vector.insert_strided_slice {{.*}}, %[[CST_I32]]
  // CHECK-COUNT-3: vector.bitcast
  %0 = vector.bitcast %input : vector<16xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func.func @bitcast_extract_extend_0(%input: vector<1xi32>) -> vector<4xi32> {
  %bitcast = vector.bitcast %input : vector<1xi32> to vector<8xi4>
  %extract = vector.extract_strided_slice %bitcast offsets = [0], sizes = [4], strides = [1] : vector<8xi4> to vector<4xi4>
  %extend = arith.extui %extract : vector<4xi4> to vector<4xi32>
  return %extend : vector<4xi32>
}

// CHECK-LABEL: func @bitcast_extract_extend_0
//  CHECK-SAME:  (%[[INPUT:.+]]: vector<1xi32>)
//   CHECK-DAG:   %[[MASK:.+]] = arith.constant 15 : i32
//   CHECK-DAG:   %[[OFF1:.+]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[OFF2:.+]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[OFF3:.+]] = arith.constant 12 : i32
//       CHECK:   %[[BASE:.+]] = vector.extract %[[INPUT]][0] : i32 from vector<1xi32>
//       CHECK:   %[[AND0:.+]] = arith.andi %[[BASE]], %[[MASK]] : i32
//       CHECK:   %[[SHR1:.+]] = arith.shrui %[[BASE]], %[[OFF1]] : i32
//       CHECK:   %[[AND1:.+]] = arith.andi %[[SHR1]], %[[MASK]] : i32
//       CHECK:   %[[SHR2:.+]] = arith.shrui %[[BASE]], %[[OFF2]] : i32
//       CHECK:   %[[AND2:.+]] = arith.andi %[[SHR2]], %[[MASK]] : i32
//       CHECK:   %[[SHR3:.+]] = arith.shrui %[[BASE]], %[[OFF3]] : i32
//       CHECK:   %[[AND3:.+]] = arith.andi %[[SHR3]], %[[MASK]] : i32
//       CHECK:   %[[RES:.+]] = vector.from_elements %[[AND0]], %[[AND1]], %[[AND2]], %[[AND3]] : vector<4xi32>
//       CHECK:   return %[[RES]] : vector<4xi32>

// -----

func.func @bitcast_extract_extend_1(%input: vector<4xi32>) -> vector<4xi32> {
  %bitcast = vector.bitcast %input : vector<4xi32> to vector<32xi4>
  %extract = vector.extract_strided_slice %bitcast offsets = [20], sizes = [4], strides = [1] : vector<32xi4> to vector<4xi4>
  %extend = arith.extui %extract : vector<4xi4> to vector<4xi32>
  return %extend : vector<4xi32>
}

// CHECK-LABEL: func.func @bitcast_extract_extend_1
//  CHECK-SAME: (%[[INPUT:.+]]: vector<4xi32>)
//   CHECK-DAG:   %[[MASK:.+]] = arith.constant 15 : i32
//   CHECK-DAG:   %[[OFF0:.+]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[OFF1:.+]] = arith.constant 20 : i32
//   CHECK-DAG:   %[[OFF2:.+]] = arith.constant 24 : i32
//   CHECK-DAG:   %[[OFF3:.+]] = arith.constant 28 : i32
//       CHECK:   %[[BASE:.+]] = vector.extract %[[INPUT]][2] : i32 from vector<4xi32>
//       CHECK:   %[[SHR0:.+]] = arith.shrui %[[BASE]], %[[OFF0]] : i32
//       CHECK:   %[[AND0:.+]] = arith.andi %[[SHR0]], %[[MASK]] : i32
//       CHECK:   %[[SHR1:.+]] = arith.shrui %[[BASE]], %[[OFF1]] : i32
//       CHECK:   %[[AND1:.+]] = arith.andi %[[SHR1]], %[[MASK]] : i32
//       CHECK:   %[[SHR2:.+]] = arith.shrui %[[BASE]], %[[OFF2]] : i32
//       CHECK:   %[[AND2:.+]] = arith.andi %[[SHR2]], %[[MASK]] : i32
//       CHECK:   %[[SHR3:.+]] = arith.shrui %[[BASE]], %[[OFF3]] : i32
//       CHECK:   %[[AND3:.+]] = arith.andi %[[SHR3]], %[[MASK]] : i32
//       CHECK:   %[[RES:.+]] = vector.from_elements %[[AND0]], %[[AND1]], %[[AND2]], %[[AND3]] : vector<4xi32>
//       CHECK:   return %[[RES]] : vector<4xi32>

// -----

func.func @bitcast_extract_extend_8_elements(%input: vector<4xi32>) -> vector<8xi32> {
  %bitcast = vector.bitcast %input : vector<4xi32> to vector<32xi4>
  %extract = vector.extract_strided_slice %bitcast offsets = [8], sizes = [8], strides = [1] : vector<32xi4> to vector<8xi4>
  %extend = arith.extui %extract : vector<8xi4> to vector<8xi32>
  return %extend : vector<8xi32>
}

// CHECK-LABEL: func @bitcast_extract_extend_8_elements
//  CHECK-SAME: (%[[INPUT:.+]]: vector<4xi32>)
//       CHECK: %[[BASE:.+]] = vector.extract %[[INPUT]][1] : i32 from vector<4xi32>
// CHECK-COUNT-8: arith.andi {{.*}} : i32
//       CHECK: %[[RES:.+]] = vector.from_elements {{.*}} : vector<8xi32>
//       CHECK: return %[[RES]] : vector<8xi32>

// -----

// CHECK-LABEL: func @large_elementwise_add
// CHECK-NOT: arith.addf {{.*}}vector<32xf32>
// CHECK: arith.addf {{.*}}vector<4xf32>
// CHECK-NOT: arith.addf {{.*}}vector<32xf32>
func.func @large_elementwise_add(%lhs: vector<32xf32>, %rhs: vector<32xf32>) -> vector<4xf32> {
  %sum = arith.addf %lhs, %rhs : vector<32xf32>
  %part = vector.extract_strided_slice %sum offsets = [8], sizes = [4], strides = [1] : vector<32xf32> to vector<4xf32>
  return %part : vector<4xf32>
}

// -----

// CHECK-LABEL: func @native_elementwise_add
// CHECK: arith.addf {{.*}}vector<4xf32>
func.func @native_elementwise_add(%lhs: vector<4xf32>, %rhs: vector<4xf32>) -> vector<4xf32> {
  %sum = arith.addf %lhs, %rhs : vector<4xf32>
  return %sum : vector<4xf32>
}

// -----

// CHECK-LABEL: func @elementwise_add_16xi8
// CHECK-COUNT-4: arith.addi {{.*}} : vector<4xi8>
// CHECK-NOT: arith.addi
// CHECK: return {{.*}} : vector<16xi8>
func.func @elementwise_add_16xi8(%lhs: vector<16xi8>, %rhs: vector<16xi8>) -> vector<16xi8> {
  %sum = arith.addi %lhs, %rhs : vector<16xi8>
  return %sum : vector<16xi8>
}

// -----

// CHECK-LABEL: func @elementwise_add_8xf16
// CHECK-COUNT-2: arith.addf {{.*}} : vector<4xf16>
// CHECK-NOT: arith.addf
// CHECK: return {{.*}} : vector<8xf16>
func.func @elementwise_add_8xf16(%lhs: vector<8xf16>, %rhs: vector<8xf16>) -> vector<8xf16> {
  %sum = arith.addf %lhs, %rhs : vector<8xf16>
  return %sum : vector<8xf16>
}
