// RUN: iree-opt --split-input-file --iree-util-test-integer-divisibility-analysis --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: @affine_apply_mul_divisibility
util.func @affine_apply_mul_divisibility(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 8> : index
  %1 = affine.apply affine_map<(d0) -> (d0 * 4)>(%0)
  // CHECK: divisibility = "udiv = 32, sdiv = 32"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_add_gcd
util.func @affine_apply_add_gcd(%arg0 : index, %arg1 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = util.assume.int %arg1<udiv = 24> : index
  %2 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%0, %1)
  // CHECK: divisibility = "udiv = 8, sdiv = 8"
  %3 = "iree_unregistered.test_int_divisibility"(%2) : (index) -> index
  util.return %3 : index
}

// -----

// CHECK-LABEL: @affine_apply_floordiv_exact
util.func @affine_apply_floordiv_exact(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 64> : index
  %1 = affine.apply affine_map<(d0) -> (d0 floordiv 4)>(%0)
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_ceildiv_exact
util.func @affine_apply_ceildiv_exact(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 64> : index
  %1 = affine.apply affine_map<(d0) -> (d0 ceildiv 4)>(%0)
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_floordiv_non_exact
util.func @affine_apply_floordiv_non_exact(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 20> : index
  %1 = affine.apply affine_map<(d0) -> (d0 floordiv 3)>(%0)
  // CHECK: divisibility = "udiv = 1, sdiv = 1"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_mod_invalidates
util.func @affine_apply_mod_invalidates(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = affine.apply affine_map<(d0) -> (d0 mod 16)>(%0)
  // CHECK: divisibility = "udiv = 1, sdiv = 1"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_composition
util.func @affine_apply_composition(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 8> : index
  %1 = affine.apply affine_map<(d0) -> (d0 * 4 + 16)>(%0)
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_with_symbol
util.func @affine_apply_with_symbol(%arg0 : index, %arg1 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = util.assume.int %arg1<udiv = 16> : index
  %2 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%0)[%1]
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %3 = "iree_unregistered.test_int_divisibility"(%2) : (index) -> index
  util.return %3 : index
}

// -----

// CHECK-LABEL: @affine_min_uniform_divisibility
util.func @affine_min_uniform_divisibility(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = affine.min affine_map<(d0) -> (d0, d0 + 64)>(%0)
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_min_different_divisibilities
util.func @affine_min_different_divisibilities(%arg0 : index, %arg1 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 16> : index
  %1 = util.assume.int %arg1<udiv = 24> : index
  %2 = affine.min affine_map<(d0, d1) -> (d0, d1)>(%0, %1)
  // CHECK: divisibility = "udiv = 8, sdiv = 8"
  %3 = "iree_unregistered.test_int_divisibility"(%2) : (index) -> index
  util.return %3 : index
}

// -----

// CHECK-LABEL: @affine_max_uniform_divisibility
util.func @affine_max_uniform_divisibility(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 32> : index
  %1 = affine.max affine_map<(d0) -> (d0, d0 - 64)>(%0)
  // CHECK: divisibility = "udiv = 32, sdiv = 32"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_max_different_divisibilities
util.func @affine_max_different_divisibilities(%arg0 : index, %arg1 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 12> : index
  %1 = util.assume.int %arg1<udiv = 18> : index
  %2 = affine.max affine_map<(d0, d1) -> (d0, d1)>(%0, %1)
  // CHECK: divisibility = "udiv = 6, sdiv = 6"
  %3 = "iree_unregistered.test_int_divisibility"(%2) : (index) -> index
  util.return %3 : index
}

// -----

// CHECK-LABEL: @affine_apply_constant
util.func @affine_apply_constant() -> index {
  %0 = affine.apply affine_map<() -> (64)>()
  // CHECK: divisibility = "udiv = 64, sdiv = 64"
  %1 = "iree_unregistered.test_int_divisibility"(%0) : (index) -> index
  util.return %1 : index
}

// -----

// CHECK-LABEL: @affine_apply_chained_operations
util.func @affine_apply_chained_operations(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 4> : index
  %1 = affine.apply affine_map<(d0) -> (d0 * 8)>(%0)
  %2 = affine.apply affine_map<(d0) -> (d0 + 16)>(%1)
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %3 = "iree_unregistered.test_int_divisibility"(%2) : (index) -> index
  util.return %3 : index
}
