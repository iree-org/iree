// RUN: iree-opt --split-input-file --iree-spirv-breakdown-large-vector %s | FileCheck %s

// CHECK-LABEL: func @extract_strided_slice_8_elements
func.func @extract_strided_slice_8_elements(%input: vector<8xf16>) -> vector<4xf16> {
  // CHECK-COUNT-4: vector.extract
  // CHECK-COUNT-4: vector.insert
  %0 = vector.extract_strided_slice %input {offsets = [1], sizes = [4], strides = [1]} : vector<8xf16> to vector<4xf16>
  return %0: vector<4xf16>
}

// -----

// CHECK-LABEL: func @extract_strided_slice_4_elements
func.func @extract_strided_slice_4_elements(%input: vector<4xf16>) -> vector<2xf16> {
  // CHECK: vector.extract_strided_slice
  %0 = vector.extract_strided_slice %input {offsets = [1], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
  return %0: vector<2xf16>
}

// -----

// CHECK-LABEL: func @bitcast_16_elements
func.func @bitcast_16_elements(%input: vector<16xi8>) -> vector<4xi32> {
  // CHECK-DAG:     %[[CST_I32:.*]] = arith.constant dense<0> : vector<4xi32>
  // CHECK-DAG:     arith.constant dense<0> : vector<4xi8>
  // CHECK-COUNT-4: vector.extract
  // CHECK-COUNT-4: vector.insert
  // CHECK:         vector.bitcast %{{.*}} : vector<4xi8> to vector<1xi32>
  // CHECK:         vector.insert_strided_slice {{.*}}, %[[CST_I32]]
  // CHECK-COUNT-3: vector.bitcast
  %0 = vector.bitcast %input : vector<16xi8> to vector<4xi32>
  return %0 : vector<4xi32>
}

// -----

func.func @bitcast_extract_extend_0(%input: vector<1xi32>) -> vector<4xi32> {
  %bitcast = vector.bitcast %input : vector<1xi32> to vector<8xi4>
  %extract = vector.extract_strided_slice %bitcast {offsets = [0], sizes = [4], strides = [1]} : vector<8xi4> to vector<4xi4>
  %extend = arith.extui %extract : vector<4xi4> to vector<4xi32>
  return %extend : vector<4xi32>
}


// CHECK-LABEL: func @bitcast_extract_extend_0
//  CHECK-SAME:  (%[[INPUT:.+]]: vector<1xi32>)
//       CHECK:   %[[ZERO:.+]] = arith.constant dense<0> : vector<4xi32>
//   CHECK-DAG:   %[[MASK:.+]] = arith.constant 15 : i32
//   CHECK-DAG:   %[[OFF1:.+]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[OFF2:.+]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[OFF3:.+]] = arith.constant 12 : i32
//       CHECK:   %[[BASE:.+]] = vector.extract %[[INPUT]][0] : i32 from vector<1xi32>
//       CHECK:   %[[AND0:.+]] = arith.andi %[[BASE]], %[[MASK]] : i32
//       CHECK:   %[[INS0:.+]] = vector.insert %[[AND0]], %[[ZERO]] [0]
//       CHECK:   %[[SHR1:.+]] = arith.shrui %[[BASE]], %[[OFF1]] : i32
//       CHECK:   %[[AND1:.+]] = arith.andi %[[SHR1]], %[[MASK]] : i32
//       CHECK:   %[[INS1:.+]] = vector.insert %[[AND1]], %[[INS0]] [1]
//       CHECK:   %[[SHR2:.+]] = arith.shrui %[[BASE]], %[[OFF2]] : i32
//       CHECK:   %[[AND2:.+]] = arith.andi %[[SHR2]], %[[MASK]] : i32
//       CHECK:   %[[INS2:.+]] = vector.insert %[[AND2]], %[[INS1]] [2]
//       CHECK:   %[[SHR3:.+]] = arith.shrui %[[BASE]], %[[OFF3]] : i32
//       CHECK:   %[[AND3:.+]] = arith.andi %[[SHR3]], %[[MASK]] : i32
//       CHECK:   %[[INS3:.+]] = vector.insert %[[AND3]], %[[INS2]] [3]
//       CHECK:   return %[[INS3]] : vector<4xi32>


// -----

func.func @bitcast_extract_extend_1(%input: vector<4xi32>) -> vector<4xi32> {
  %bitcast = vector.bitcast %input : vector<4xi32> to vector<32xi4>
  %extract = vector.extract_strided_slice %bitcast {offsets = [20], sizes = [4], strides = [1]} : vector<32xi4> to vector<4xi4>
  %extend = arith.extui %extract : vector<4xi4> to vector<4xi32>
  return %extend : vector<4xi32>
}

// CHECK-LABEL: func.func @bitcast_extract_extend_1
//  CHECK-SAME: (%[[INPUT:.+]]: vector<4xi32>)
//       CHECK:   %[[ZERO:.+]] = arith.constant dense<0> : vector<4xi32>
//   CHECK-DAG:   %[[MASK:.+]] = arith.constant 15 : i32
//   CHECK-DAG:   %[[OFF0:.+]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[OFF1:.+]] = arith.constant 20 : i32
//   CHECK-DAG:   %[[OFF2:.+]] = arith.constant 24 : i32
//   CHECK-DAG:   %[[OFF3:.+]] = arith.constant 28 : i32
//       CHECK:   %[[BASE:.+]] = vector.extract %[[INPUT]][2] : i32 from vector<4xi32>
//       CHECK:   %[[SHR0:.+]] = arith.shrui %[[BASE]], %[[OFF0]] : i32
//       CHECK:   %[[AND0:.+]] = arith.andi %[[SHR0]], %[[MASK]] : i32
//       CHECK:   %[[INS0:.+]] = vector.insert %[[AND0]], %[[ZERO]] [0]
//       CHECK:   %[[SHR1:.+]] = arith.shrui %[[BASE]], %[[OFF1]] : i32
//       CHECK:   %[[AND1:.+]] = arith.andi %[[SHR1]], %[[MASK]] : i32
//       CHECK:   %[[INS1:.+]] = vector.insert %[[AND1]], %[[INS0]] [1]
//       CHECK:   %[[SHR2:.+]] = arith.shrui %[[BASE]], %[[OFF2]] : i32
//       CHECK:   %[[AND2:.+]] = arith.andi %[[SHR2]], %[[MASK]] : i32
//       CHECK:   %[[INS2:.+]] = vector.insert %[[AND2]], %[[INS1]] [2]
//       CHECK:   %[[SHR3:.+]] = arith.shrui %[[BASE]], %[[OFF3]] : i32
//       CHECK:   %[[AND3:.+]] = arith.andi %[[SHR3]], %[[MASK]] : i32
//       CHECK:   %[[INS3:.+]] = vector.insert %[[AND3]], %[[INS2]] [3]
//       CHECK:   return %[[INS3]] : vector<4xi32>
