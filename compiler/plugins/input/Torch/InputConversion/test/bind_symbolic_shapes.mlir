// RUN: iree-opt --pass-pipeline="builtin.module(func.func(torch-iree-bind-symbolic-shapes))" --split-input-file --verify-diagnostics %s | FileCheck %s

// This example was captured from a program which has a dynamic batch size and
// tiled inner dim on one of the arguments, causing a symbolic relationship on
// the second dimension.
// CHECK-LABEL: @basic_example
module @basic_example {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> attributes {torch.assume_strict_symbolic_shapes} {
    // CHECK-DAG: %[[ARG1_ANCHOR:.*]] = torch_c.to_builtin_tensor %arg1 : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
    // CHECK-DAG: %[[ARG0_ANCHOR:.*]] = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[?,?],f32> -> tensor<?x?xf32>
    // CHECK-DAG: %[[POS0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[POS1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[DIM0:.*]] = tensor.dim %1, %[[POS0]] :
    // CHECK-DAG: %[[DIM1:.*]] = tensor.dim %1, %[[POS1]] :
    // CHECK: %[[ARG0_DIM0_NARROW:.*]] = util.assume.narrow %[[DIM0]] : index to i32
    // CHECK: %[[ARG0_DIM0_RANGE:.*]] = util.assume.range %[[ARG0_DIM0_NARROW]] in [1, 1024] : index
    // CHECK: %[[ARG0_DIM1_NARROW:.*]] = util.assume.narrow %[[DIM1]] : index to i32
    // CHECK: %[[ARG0_DIM1_RANGE:.*]] = util.assume.range %[[ARG0_DIM1_NARROW]] in [1, 1024] : index
    // CHECK: %[[ARG0_TIE:.*]] = flow.tensor.tie_shape %[[ARG0_ANCHOR]] : tensor<?x?xf32>{%[[ARG0_DIM0_RANGE]], %[[ARG0_DIM1_RANGE]]}
    // CHECK: %[[ARG0_EXPORT:.*]] = torch_c.from_builtin_tensor %[[ARG0_TIE]]
    // CHECK: %[[ARG1_DIM0_NARROW:.*]] = util.assume.narrow %[[DIM0]] : index to i32
    // CHECK: %[[ARG1_DIM0_RANGE:.*]] = util.assume.range %[[ARG1_DIM0_NARROW]] in [1, 1024]
    // CHECK: %[[MULTIPLIER0:.*]] = arith.constant 2 : index
    // CHECK: %[[ARG1_DIM1:.*]] = arith.muli %[[DIM1]], %[[MULTIPLIER0]]
    // CHECK: %[[ARG1_DIM1_NARROW:.*]] = util.assume.narrow %[[ARG1_DIM1]] : index to i32
    // CHECK: %[[ARG1_DIM1_RANGE:.*]] = util.assume.range %[[ARG1_DIM1_NARROW]] in [2, 2048] : index
    // CHECK: %[[ARG1_DIM1_DIV:.*]] = util.assume.divisible %[[ARG1_DIM1_RANGE]] by 2
    // CHECK: %[[ARG1_TIE:.*]] = flow.tensor.tie_shape %[[ARG1_ANCHOR]] : tensor<?x?xf32>{%[[ARG1_DIM0_RANGE]], %[[ARG1_DIM1_DIV]]}
    // CHECK: %[[ARG1_EXPORT:.*]] = torch_c.from_builtin_tensor %[[ARG1_TIE]]
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 1024} : !torch.int
    %2 = torch.symbolic_int "2*s1" {min_val = 0, max_val = 2048} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %3 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.repeat %arg0, %3 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %4, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    %int1_0 = torch.constant.int 1
    %5 = torch.aten.add.Tensor %4, %arg1, %int1_0 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %5, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    return %5 : !torch.vtensor<[?,?],f32>
  }
}

// -----
// This example was captured from a torch program that used a symbol that did
// not correspond to any dimension (being used in an expression as part of
// distinct dimensions). This exercises a special case in the pass for deferring
// to runtime resolution of the dim.
// We just verify that the vital information has been captured.
// CHECK-LABEL: @unbacked_symbol
module @unbacked_symbol {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    // CHECK: util.assume.narrow
    // CHECK: util.assume.range{{.*}} [1, 1024]
    // CHECK: util.assume.narrow
    // CHECK: util.assume.range{{.*}} [2, 2048]
    // CHECK: util.assume.divisible{{.*}} by 2
    // CHECK: tie_shape
    // CHECK: util.assume.narrow
    // CHECK: util.assume.range{{.*}} [1, 1024]
    // CHECK: util.assume.narrow
    // CHECK: util.assume.range{{.*}} [4, 4096]
    // CHECK: util.assume.divisible{{.*}} by 4
    // CHECK: tie_shape
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "2*s4" {min_val = 0, max_val = 2048} : !torch.int
    %2 = torch.symbolic_int "4*s4" {min_val = 0, max_val = 4096} : !torch.int
    %3 = torch.symbolic_int "s4" {min_val = 2, max_val = 1024} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %3], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %3], affine_map<()[s0, s1] -> (s0, s1 * 4)> : !torch.vtensor<[?,?],f32>
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %4 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %5 = torch.aten.repeat %arg0, %4 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %5, [%0, %3], affine_map<()[s0, s1] -> (s0, s1 * 4)> : !torch.vtensor<[?,?],f32>
    %int1_0 = torch.constant.int 1
    %6 = torch.aten.add.Tensor %5, %arg1, %int1_0 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %6, [%0, %3], affine_map<()[s0, s1] -> (s0, s1 * 4)> : !torch.vtensor<[?,?],f32>
    return %6 : !torch.vtensor<[?,?],f32>
  }
}

// -----
// CHECK-LABEL: @all_bindings_dropped
module @all_bindings_dropped {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> {
    // CHECK-NOT: torch.symbolic_int
    // CHECK-NOT: torch.bind_symbolic_shape
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 1024} : !torch.int
    %2 = torch.symbolic_int "2*s1" {min_val = 0, max_val = 2048} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    %int1 = torch.constant.int 1
    %int2 = torch.constant.int 2
    %3 = torch.prim.ListConstruct %int1, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.aten.repeat %arg0, %3 : !torch.vtensor<[?,?],f32>, !torch.list<int> -> !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %4, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    %int1_0 = torch.constant.int 1
    %5 = torch.aten.add.Tensor %4, %arg1, %int1_0 : !torch.vtensor<[?,?],f32>, !torch.vtensor<[?,?],f32>, !torch.int -> !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %5, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 2)> : !torch.vtensor<[?,?],f32>
    return %5 : !torch.vtensor<[?,?],f32>
  }
}

// -----
// CHECK-LABEL: @add_expr
module @add_expr {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) {
    // CHECK: addi
    // CHECK-NOT: divisible
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 1024} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 + 2)> : !torch.vtensor<[?,?],f32>
    return
  }
}

// -----
// CHECK-LABEL: @mod_expr
module @mod_expr {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) {
    // CHECK: remui
    // CHECK-NOT: divisible
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 1024} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 mod 2)> : !torch.vtensor<[?,?],f32>
    return
  }
}

// -----
// CHECK-LABEL: @floordiv_expr
module @floordiv_expr {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) {
    // CHECK: divui
    // CHECK-NOT: divisible
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 1024} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 floordiv 2)> : !torch.vtensor<[?,?],f32>
    return
  }
}

// -----
// Verifies that unsupported dim expressions warn (and do not assert).
// CHECK-LABEL: @unsupported_non_symbolic
module @unsupported_non_symbolic {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) {
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 1024} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 1024} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    // expected-warning@+1 {{Symbolic shape expression not supported: d0}}
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<(d0)[s0, s1] -> (s0, s1 + d0)> : !torch.vtensor<[?,?],f32>
    return
  }
}

// -----
// Torch uses high values to signal unbounded ranges. Ensure they are
// suppressed.
// CHECK-LABEL: @torch_unbounded_max_range
module @torch_unbounded_max_range {
  func.func @main(%arg0: !torch.vtensor<[?,?],f32>, %arg1: !torch.vtensor<[?,?],f32>) {
    // CHECK-NOT: util.assume.range
    %0 = torch.symbolic_int "s0" {min_val = 0, max_val = 4611686018427387903} : !torch.int
    %1 = torch.symbolic_int "s1" {min_val = 0, max_val = 9223372036854775806} : !torch.int
    torch.bind_symbolic_shape %arg0, [%0, %1], affine_map<()[s0, s1] -> (s0, s1)> : !torch.vtensor<[?,?],f32>
    torch.bind_symbolic_shape %arg1, [%0, %1], affine_map<()[s0, s1] -> (s0, s1 * 10)> : !torch.vtensor<[?,?],f32>
    return
  }
}
