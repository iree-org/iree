// RUN: iree-opt --split-input-file --iree-sanitize-module-names %s | FileCheck %s

// CHECK-LABEL: module @letters
builtin.module @letters {}

// -----

// CHECK-LABEL: module @digits0123456789
builtin.module @digits0123456789 {}

// -----

// CHECK-LABEL: module @u_n_d_e_r_s_c_o_r_e_s
builtin.module @u_n_d_e_r_s_c_o_r_e_s {}

// -----

// CHECK-LABEL: module @dollar__signs
builtin.module @dollar$$signs {}

// -----

// CHECK-LABEL: module @periods_change_to_underscores
builtin.module @periods.change.to.underscores {}
