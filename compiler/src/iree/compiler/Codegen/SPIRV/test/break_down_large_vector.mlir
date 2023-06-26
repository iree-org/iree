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
