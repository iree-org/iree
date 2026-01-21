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

// CHECK-LABEL: @affine_apply_mul_negative
util.func @affine_apply_mul_negative(%arg0 : index) -> index {
  %0 = util.assume.int %arg0<udiv = 8> : index
  %1 = affine.apply affine_map<(d0) -> (d0 * -4)>(%0)
  // CHECK: divisibility = "udiv = 32, sdiv = 32"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
}

// -----

// CHECK-LABEL: @affine_apply_add_gcd
util.func @affine_apply_add_gcd(%arg0 : index, %arg1 : index) -> index {
  %0:2 = util.assume.int %arg0<udiv = 16>,
                         %arg1<udiv = 24> : index, index
  %1 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%0#0, %0#1)
  // CHECK: divisibility = "udiv = 8, sdiv = 8"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
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

// CHECK-LABEL: @affine_apply_mod
util.func @affine_apply_mod(%arg0 : index) -> index {
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
  %0:2 = util.assume.int %arg0<udiv = 16>,
                         %arg1<udiv = 16> : index, index
  %1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%0#0)[%0#1]
  // CHECK: divisibility = "udiv = 16, sdiv = 16"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
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
  %0:2 = util.assume.int %arg0<udiv = 16>,
                         %arg1<udiv = 24> : index, index
  %1 = affine.min affine_map<(d0, d1) -> (d0, d1)>(%0#0, %0#1)
  // CHECK: divisibility = "udiv = 8, sdiv = 8"
  %2 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  util.return %2 : index
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
util.func @affine_max_different_divisibilities(%arg0 : index, %arg1 : index, %arg2 : index) -> index {
  %0:3 = util.assume.int %arg0<udiv = 12>,
                         %arg1<udiv = 24>,
                         %arg2<udiv = 18> : index, index, index
  %3 = affine.max affine_map<(d0, d1, d2) -> (d0, d1, d2)>(%0#0, %0#1, %0#2)
  // CHECK: divisibility = "udiv = 6, sdiv = 6"
  %4 = "iree_unregistered.test_int_divisibility"(%3) : (index) -> index
  util.return %4 : index
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

// -----

// CHECK-LABEL: @complex_chained_affine_ops
util.func @complex_chained_affine_ops(%arg0 : index, %arg1 : index, %arg2 : index) -> index {
  %0:3 = util.assume.int %arg0<udiv = 210>,
                         %arg1<udiv = 7>,
                         %arg2<udiv = 15> : index, index, index
  %1 = affine.apply affine_map<(d0, d1) -> (d0 + 2 * d1)>(%0#0, %0#1)
  // CHECK: divisibility = "udiv = 14, sdiv = 14"
  %div_1 = "iree_unregistered.test_int_divisibility"(%1) : (index) -> index
  %2 = affine.max affine_map<(d0, d1) -> (d0 floordiv 6, d1 * 3)>(%0#0, %0#2)
  // CHECK: divisibility = "udiv = 5, sdiv = 5"
  %div_2 = "iree_unregistered.test_int_divisibility"(%2) : (index) -> index
  %3 = affine.min affine_map<(d0)[s0] -> (2 * (s0 * d0 - 14) ceildiv 7, d0 floordiv 3 * 2)>(%2)[%1]
  // CHECK: divisibility = "udiv = 2, sdiv = 2"
  %div_3 = "iree_unregistered.test_int_divisibility"(%3) : (index) -> index
  util.return %div_3 : index
}

// -----

// CHECK-LABEL: @scf_for_constant_step_from_zero
util.func @scf_for_constant_step_from_zero() {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c0 to %c100 step %c8 {
    // CHECK: divisibility = "udiv = 8, sdiv = 8"
    %0 = "iree_unregistered.test_int_divisibility"(%iv) : (index) -> index
  }
  util.return
}

// -----

// CHECK-LABEL: @scf_for_nontrivial_gcd
util.func @scf_for_nontrivial_gcd() {
  %c12 = arith.constant 12 : index
  %c100 = arith.constant 100 : index
  %c18 = arith.constant 18 : index
  scf.for %iv = %c12 to %c100 step %c18 {
    // CHECK: divisibility = "udiv = 6, sdiv = 6"
    %0 = "iree_unregistered.test_int_divisibility"(%iv) : (index) -> index
  }
  util.return
}

// -----

// CHECK-LABEL: @scf_for_dynamic_bounds_and_step
util.func @scf_for_dynamic_bounds_and_step(%arg0 : index, %arg1 : index) {
  %lb = util.assume.int %arg0<udiv = 16> : index
  %step = util.assume.int %arg1<udiv = 24> : index
  %c100 = arith.constant 100 : index
  scf.for %iv = %lb to %c100 step %step {
    // CHECK: divisibility = "udiv = 8, sdiv = 8"
    %0 = "iree_unregistered.test_int_divisibility"(%iv) : (index) -> index
  }
  util.return
}

// -----

// CHECK-LABEL: @scf_for_coprime_bounds_and_step
util.func @scf_for_coprime_bounds_and_step() {
  %c15 = arith.constant 15 : index
  %c100 = arith.constant 100 : index
  %c8 = arith.constant 8 : index
  scf.for %iv = %c15 to %c100 step %c8 {
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %0 = "iree_unregistered.test_int_divisibility"(%iv) : (index) -> index
  }
  util.return
}

// -----

// CHECK-LABEL: @scf_forall_dynamic_bounds
util.func @scf_forall_dynamic_bounds(%arg0 : index, %arg1 : index, %arg2 : index) {
  %lb = util.assume.int %arg0<udiv = 20> : index
  %ub = util.assume.int %arg1<udiv = 1> : index
  %step = util.assume.int %arg2<udiv = 15> : index
  scf.forall (%iv) = (%lb) to (%ub) step (%step) {
    // CHECK: divisibility = "udiv = 5, sdiv = 5"
    %0 = "iree_unregistered.test_int_divisibility"(%iv) : (index) -> index
  }
  util.return
}

// -----

// CHECK-LABEL: @scf_forall_multiple_ivs
util.func @scf_forall_multiple_ivs(%arg0 : index, %arg1 : index, %arg2 : index,
                                   %arg3 : index, %arg4 : index, %arg5 : index) {
  %lbs:2 = util.assume.int %arg0<udiv = 20>,
                           %arg3<udiv = 10> : index, index
  %ubs:2 = util.assume.int %arg1<udiv = 1>,
                           %arg4<udiv = 10> : index, index
  %steps:2 = util.assume.int %arg2<udiv = 15>,
                             %arg5<udiv = 3> : index, index
  scf.forall (%iv0, %iv1) = (%lbs#0, %lbs#1) to (%ubs#0, %ubs#1) step (%steps#0, %steps#1) {
    // CHECK: divisibility = "udiv = 5, sdiv = 5"
    %0 = "iree_unregistered.test_int_divisibility"(%iv0) : (index) -> index
    // CHECK: divisibility = "udiv = 1, sdiv = 1"
    %1 = "iree_unregistered.test_int_divisibility"(%iv1) : (index) -> index
  }
  util.return
}
