// RUN: iree-opt %s -split-input-file -verify-diagnostics

func @tensor(%arg0 : tensor<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_hl_interp.tanh_f"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return
}

// -----

func @scalar(%arg0 : f32) {
  // expected-error@+1 {{must be memref}}
  "iree_hl_interp.tanh_f"(%arg0) : (f32) -> f32
  return
}

// -----

func @vector(%arg0 : vector<1xf32>) {
  // expected-error@+1 {{must be memref}}
  "iree_hl_interp.tanh_f"(%arg0) : (vector<1xf32>) -> vector<1xf32>
  return
}

// -----

func @bad_bool(%a : memref<1xf32>) {
  // expected-error@+1 {{must be memref of boolean-storing type (1 or 8 -bit integer) values}}
  "iree_hl_interp.cmp_f"(%a, %a) {predicate = 0 : i32} : (memref<1xf32>, memref<1xf32>) -> memref<1xi32>
  return
}

// -----

func @not_scalar(%a : memref<2xf32>) {
  // expected-error@+1 {{0D memref of signless integer values}}
  "iree_hl_interp.length"(%a) : (memref<2xf32>) -> memref<2xi32>
  return
}

// -----

func @not_scalar_int(%a : memref<1xf32>) {
  // expected-error@+1 {{0D memref of signless integer values}}
  "iree_hl_interp.length"(%a) : (memref<1xf32>) -> memref<f32>
  return
}

// -----

func @not_scalar_bool(%cond : memref<i32>, %a : memref<1xf32>) {
  // expected-error@+1 {{0D memref of boolean-storing type (1 or 8 -bit integer) values}}
  "iree_hl_interp.cond_assign"(%cond, %a, %a) : (memref<i32>, memref<1xf32>, memref<1xf32>) -> memref<1xf32>
  return
}

// -----

func @bad_copy(%src : memref<2xf32>, %srcIndices : memref<2xi32>, %dst : memref<2xf32>, %dstIndices : memref<2xi32>, %lengths : memref<2xi32>) {
  // expected-error@+1 {{src/dst rank is the same as srcIndices/dstIndices/lengths size}}
  "iree_hl_interp.copy"(%src, %srcIndices, %dst, %dstIndices, %lengths) : (memref<2xf32>, memref<2xi32>, memref<2xf32>, memref<2xi32>, memref<2xi32>) -> ()
  return
}
