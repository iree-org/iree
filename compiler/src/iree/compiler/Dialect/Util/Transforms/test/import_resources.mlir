// RUN: iree-opt --split-input-file --iree-util-import-resources %s | FileCheck %s

// CHECK-LABEL: util.func public @constant_splat_i64
util.func public @constant_splat_i64() -> tensor<4xi64> {
  // Splats should not convert.
  // CHECK-NEXT: constant dense<123>
  %c123 = arith.constant dense<123> : tensor<4xi64>
  util.return %c123 : tensor<4xi64>
}

// -----
// CHECK-LABEL: util.func public @dense_i1
util.func public @dense_i1() -> tensor<4xi1> {
  // CHECK: dense_resource<dense_elements_i1>
  %c123 = arith.constant dense<[true, false, false, true]> : tensor<4xi1>
  util.return %c123 : tensor<4xi1>
}

// CHECK: dense_elements_i1: "0x4000000001000001"

// -----
// CHECK-LABEL: util.func public @dense_i8
util.func public @dense_i8() -> tensor<4xi8> {
  // CHECK: dense_resource<dense_elements_i8>
  %c123 = arith.constant dense<[1, 2, 3, 127]> : tensor<4xi8>
  util.return %c123 : tensor<4xi8>
}

// CHECK: dense_elements_i8: "0x400000000102037F"

// -----
// CHECK-LABEL: util.func public @dense_i16
util.func public @dense_i16() -> tensor<4xi16> {
  // CHECK: dense_resource<dense_elements_i16>
  %c123 = arith.constant dense<[1, 2, 3, 127]> : tensor<4xi16>
  util.return %c123 : tensor<4xi16>
}

// CHECK: dense_elements_i16: "0x400000000100020003007F00"

// -----
// CHECK-LABEL: util.func public @dense_i32
util.func public @dense_i32() -> tensor<4xi32> {
  // CHECK: dense_resource<dense_elements_i32>
  %c123 = arith.constant dense<[1, 2, 3, 127]> : tensor<4xi32>
  util.return %c123 : tensor<4xi32>
}

// CHECK: dense_elements_i32: "0x400000000100000002000000030000007F000000"

// -----
// CHECK-LABEL: util.func public @dense_i64
util.func public @dense_i64() -> tensor<4xi64> {
  // CHECK: dense_resource<dense_elements_i64>
  %c123 = arith.constant dense<[1, 2, 3, 127]> : tensor<4xi64>
  util.return %c123 : tensor<4xi64>
}

// CHECK: dense_elements_i64: "0x400000000100000000000000020000000000000003000000000000007F00000000000000"

// -----
// CHECK-LABEL: util.func public @dense_f16
util.func public @dense_f16() -> tensor<4xf16> {
  // CHECK: dense_resource<dense_elements_f16>
  %c123 = arith.constant dense<[1.1, 2.2, 3.3, 0.0]> : tensor<4xf16>
  util.return %c123 : tensor<4xf16>
}

// CHECK: dense_elements_f16: "0x40000000663C66409A420000"

// -----
// CHECK-LABEL: util.func public @dense_f32
util.func public @dense_f32() -> tensor<4xf32> {
  // CHECK: dense_resource<dense_elements_f32>
  %c123 = arith.constant dense<[1.1, 2.2, 3.3, 0.0]> : tensor<4xf32>
  util.return %c123 : tensor<4xf32>
}

// CHECK: dense_elements_f32: "0x40000000CDCC8C3FCDCC0C403333534000000000"

// -----
// CHECK-LABEL: util.func public @dense_f64
util.func public @dense_f64() -> tensor<4xf64> {
  // CHECK: dense_resource<dense_elements_f64>
  %c123 = arith.constant dense<[1.1, 2.2, 3.3, 0.0]> : tensor<4xf64>
  util.return %c123 : tensor<4xf64>
}

// CHECK: dense_elements_f64: "0x400000009A9999999999F13F9A999999999901406666666666660A400000000000000000"
