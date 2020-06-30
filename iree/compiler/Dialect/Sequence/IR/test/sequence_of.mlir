// Tests the sequence type.

// RUN: iree-opt -verify-diagnostics -canonicalize -split-input-file %s | IreeFileCheck %s

// -----

// CHECK-LABEL: @comp1
func @comp1(%arg0 : !sequence.of<i32>) -> !sequence.of<i32> {
  // CHECK-NEXT: return %arg0 : !sequence.of<i32>
  return %arg0 : !sequence.of<i32>
}

// -----

// CHECK-LABEL: @comp2
func @comp2(%arg0 : !sequence.of<tensor<5xi32>>) -> !sequence.of<tensor<5xi32>> {
  // CHECK-NEXT: return %arg0 : !sequence.of<tensor<5xi32>>
  return %arg0 : !sequence.of<tensor<5xi32>>
}
