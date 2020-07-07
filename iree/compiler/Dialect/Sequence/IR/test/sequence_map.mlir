// Tests the sequence map op.

// RUN: iree-opt -verify-diagnostics -canonicalize -split-input-file %s | IreeFileCheck %s

// -----

func @comp1(%arg0 : tensor<1x1xi32>) -> tensor<1xi32> {
  %0 = flow.tensor.reshape %arg0 : tensor<1x1xi32> -> tensor<1xi32>
  return %0 : tensor<1xi32>
}

// CHECK-LABEL: @comp2
func @comp2(%arg0 : !sequence.of<tensor<1x1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // CHECK-NEXT: %0 = sequence.map @comp1, %arg0 : !sequence.of<tensor<1x1xi32>> -> !sequence.of<tensor<1xi32>>
  %0 = sequence.map @comp1, %arg0 : !sequence.of<tensor<1x1xi32>> -> !sequence.of<tensor<1xi32>>
  // CHECK-NEXT: return %0 : !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}

// -----

func @comp3(%arg0 : i32) -> i32 {
  // expected-error@+1 {{must be of}}
  %0 = sequence.map @comp1, %arg0 : i32 -> i32
  return %0 : i32
}

// -----

// expected-note@+1 {{prior use}}
func @comp4(%arg0 : !sequence.of<tensor<1x1xi32>>) -> i32 {
  // expected-error@+1 {{different type}}
  %0 = sequence.map @comp1, %arg0 : i32 -> i32
  return %0 : i32
}

// -----

func @comp5(%arg0 : !sequence.of<tensor<1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // expected-error@+1 {{mapping function nonexistent_comp not found}}
  %0 = sequence.map @nonexistent_comp, %arg0 : !sequence.of<tensor<1xi32>> -> !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}

// -----

func @comp6() -> tensor<1xi32> {
  %0 = mhlo.constant dense<10> : tensor<1xi32>
  return %0 : tensor<1xi32>
}

func @comp7(%arg0 : !sequence.of<tensor<1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // expected-error@+1 {{exactly one argument}}
  %0 = sequence.map @comp6, %arg0 : !sequence.of<tensor<1xi32>> -> !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}

// -----

func @comp8(%arg0 : tensor<1xi32>) -> () {
  return
}

func @comp9(%arg0 : !sequence.of<tensor<1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // expected-error@+1 {{exactly one result}}
  %0 = sequence.map @comp8, %arg0 : !sequence.of<tensor<1xi32>> -> !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}

// -----

func @comp10(%arg0 : tensor<100xi32>) -> tensor<100xi32> {
  return %arg0 : tensor<100xi32>
}

func @comp11(%arg0 : !sequence.of<tensor<1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // expected-error@+1 {{expects argument of type}}
  %0 = sequence.map @comp10, %arg0 : !sequence.of<tensor<1xi32>> -> !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}

// -----

func @comp12(%arg0 : tensor<1xi32>) -> tensor<1x1xi32> {
  %0 = flow.tensor.reshape %arg0 : tensor<1xi32> -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

func @comp13(%arg0 : !sequence.of<tensor<1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // expected-error@+1 {{returns result of type}}
  %0 = sequence.map @comp12, %arg0 : !sequence.of<tensor<1xi32>> -> !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}

// -----

flow.variable @comp14 : tensor<1xi32>

func @comp15(%arg0 : !sequence.of<tensor<1xi32>>) -> !sequence.of<tensor<1xi32>> {
  // expected-error@+1 {{not a function}}
  %0 = sequence.map @comp14, %arg0 : !sequence.of<tensor<1xi32>> -> !sequence.of<tensor<1xi32>>
  return %0 : !sequence.of<tensor<1xi32>>
}
