// RUN: iree-opt --split-input-file %s | iree-opt --split-input-file | FileCheck %s

// CHECK-LABEL: @hoistable_conversion_eager
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9$._-]+]]
func.func @hoistable_conversion_eager(%arg0 : vector<4xf32>) -> vector<2x2xf32> {
  // CHECK: util.hoistable_conversion "shape_cast_from_intrinsic" inverts("shape_cast_to_intrinsic")
  // CHECK-SAME: (%[[B:.*]] = %[[ARG0]]) : (vector<4xf32>) -> vector<2x2xf32>
  // CHECK:   vector.shape_cast %[[B]] : vector<4xf32> to vector<2x2xf32>
  %0 = util.hoistable_conversion "shape_cast_from_intrinsic" inverts("shape_cast_to_intrinsic") (%b = %arg0) : (vector<4xf32>) -> vector<2x2xf32> {
    %1 = vector.shape_cast %b : vector<4xf32> to vector<2x2xf32>
    util.return %1 : vector<2x2xf32>
  }
  return %0 : vector<2x2xf32>
}

// -----

// CHECK-LABEL: @hoistable_conversion_multi_io
// CHECK-SAME: %[[A:[a-zA-Z0-9$._-]+]]
// CHECK-SAME: %[[B:[a-zA-Z0-9$._-]+]]
func.func @hoistable_conversion_multi_io(%a : vector<4xf32>, %b : vector<8xf16>) -> (vector<2x2xf32>, vector<4x2xf16>) {
  // CHECK: util.hoistable_conversion "distribute" inverts("reassemble")
  // CHECK-SAME: (%[[BA:.*]] = %[[A]], %[[BB:.*]] = %[[B]])
  // CHECK-SAME: : (vector<4xf32>, vector<8xf16>) -> (vector<2x2xf32>, vector<4x2xf16>)
  %0:2 = util.hoistable_conversion "distribute" inverts("reassemble") (%ba = %a, %bb = %b) : (vector<4xf32>, vector<8xf16>) -> (vector<2x2xf32>, vector<4x2xf16>) {
    %1 = vector.shape_cast %ba : vector<4xf32> to vector<2x2xf32>
    %2 = vector.shape_cast %bb : vector<8xf16> to vector<4x2xf16>
    util.return %1, %2 : vector<2x2xf32>, vector<4x2xf16>
  }
  return %0#0, %0#1 : vector<2x2xf32>, vector<4x2xf16>
}

// -----

// CHECK-LABEL: @hoistable_conversion_no_inputs
func.func @hoistable_conversion_no_inputs() -> vector<4xf32> {
  // CHECK: util.hoistable_conversion "create" inverts("destroy") ()
  // CHECK-SAME: : () -> vector<4xf32>
  %0 = util.hoistable_conversion "create" inverts("destroy") () : () -> vector<4xf32> {
    %cst = arith.constant dense<0.0> : vector<4xf32>
    util.return %cst : vector<4xf32>
  }
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: @hoistable_conversion_rdna3_interleave
// CHECK-SAME: %[[ACC:[a-zA-Z0-9$._-]+]]
func.func @hoistable_conversion_rdna3_interleave(%acc : vector<8xf16>) -> vector<16xf16> {
  // CHECK: util.hoistable_conversion "rdna3_interleave_acc" inverts("rdna3_deinterleave_acc")
  // CHECK-SAME: (%[[B:.*]] = %[[ACC]]) : (vector<8xf16>) -> vector<16xf16>
  %0 = util.hoistable_conversion "rdna3_interleave_acc" inverts("rdna3_deinterleave_acc") (%b = %acc) : (vector<8xf16>) -> vector<16xf16> {
    %zero = arith.constant dense<0.0> : vector<8xf16>
    %interleaved = vector.interleave %b, %zero : vector<8xf16> -> vector<16xf16>
    util.return %interleaved : vector<16xf16>
  }
  return %0 : vector<16xf16>
}
