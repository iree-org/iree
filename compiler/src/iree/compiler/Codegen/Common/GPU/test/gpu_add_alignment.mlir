// CHECK-LABEL: @vector_load_already_aligned
func.func @vector_load_already_aligned() -> vector<8xi64> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8xi64>
  %c0 = arith.constant 0 : index
  // Verify that we did not propagate alignment from the allocation
  // to the load since the load is already annotated with an alignment.
  //
  // CHECK: vector.load
  // CHECK-SAME: alignment = 32
  %0 = vector.load %alloc[%c0] { alignment = 32 : i64 } : memref<8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

// -----

// CHECK-LABEL: @match_failure_no_definition
func.func @match_failure_no_definition(%alloc : memref<8xi64>) -> vector<8xi64> {
  %c0 = arith.constant 0 : index
  // CHECK: vector.load
  // CHECK-NOT: alignment
  %0 = vector.load %alloc[%c0] : memref<8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

// -----

// CHECK-LABEL: @trivial_case
func.func @trivial_case() -> vector<8xi64> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8xi64>
  %c0 = arith.constant 0 : index
  // CHECK: vector.load
  // CHECK-SAME: alignment = 64
  %0 = vector.load %alloc[%c0] : memref<8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

// -----

// CHECK-LABEL: @trivial_case_i8
func.func @trivial_case_i8() -> vector<8xi8> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8xi8>
  %c0 = arith.constant 0 : index
  // CHECK: vector.load
  // CHECK-SAME: alignment = 64
  %0 = vector.load %alloc[%c0] : memref<8xi8>, vector<8xi8>
  return %0 : vector<8xi8>
}


// -----

// CHECK-LABEL: @dyn_idxs_left
func.func @dyn_idxs_left(%dynidx : index) -> vector<8xi64> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8x8xi64>
  %c0 = arith.constant 0 : index
  // CHECK: vector.load
  // CHECK-SAME: alignment = 64
  %0 = vector.load %alloc[%dynidx, %c0] : memref<8x8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

// -----

// CHECK-LABEL: @dyn_idxs_right
func.func @dyn_idxs_right(%dynidx : index) -> vector<8xi64> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8x8xi64>
  %c0 = arith.constant 0 : index
  // CHECK: vector.load
  // CHECK-SAME: alignment = 8
  %0 = vector.load %alloc[%c0, %dynidx] : memref<8x8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

// -----

// CHECK-LABEL: @constant_one
func.func @constant_one(%dynidx : index) -> vector<8xi64> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8x8xi64>
  %c1 = arith.constant 1 : index
  // CHECK: vector.load
  // CHECK-SAME: alignment = 8
  %0 = vector.load %alloc[%dynidx, %c1] : memref<8x8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

// -----

// CHECK-LABEL: @constant_two
func.func @constant_two(%dynidx : index) -> vector<8xi64> {
  %alloc = memref.alloc() { alignment = 64 : i64 } : memref<8x8xi64>
  %c1 = arith.constant 2 : index
  // CHECK: vector.load
  // CHECK-SAME: alignment = 16
  %0 = vector.load %alloc[%dynidx, %c1] : memref<8x8xi64>, vector<8xi64>
  return %0 : vector<8xi64>
}

