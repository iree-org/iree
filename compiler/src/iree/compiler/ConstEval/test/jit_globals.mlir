// RUN: iree-opt --split-input-file --iree-consteval-jit-globals --iree-consteval-jit-debug --verify-diagnostics %s | FileCheck %s

module @no_uninitialized {
  util.global private @hoisted : tensor<5x6xf32> = dense<4.0> : tensor<5x6xf32>
  util.func public @main() -> tensor<5x6xf32> {
    %hoisted = util.global.load @hoisted : tensor<5x6xf32>
    util.return %hoisted : tensor<5x6xf32>
  }
}

// -----

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @linalg_tensor_jit
module @linalg_tensor_jit {
 // CHECK: util.global private @[[EVALED:.+]] = dense<4.000000e+04> : tensor<5x6xf32>
  util.global private @hoisted : tensor<5x6xf32>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<2.0e+02> : tensor<f32>
    %0 = tensor.empty() : tensor<5x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : tensor<f32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      linalg.yield %arg0 : f32
    } -> tensor<5x6xf32>
    %2 = tensor.empty() : tensor<5x6xf32>
    %3 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%1, %1 : tensor<5x6xf32>, tensor<5x6xf32>) outs(%2 : tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %4 = arith.mulf %arg0, %arg1 : f32
      linalg.yield %4 : f32
    } -> tensor<5x6xf32>
    util.global.store %3, @hoisted : tensor<5x6xf32>
    util.return
  }
  util.func public @main() -> tensor<5x6xf32> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<5x6xf32>
    util.return %hoisted : tensor<5x6xf32>
  }
}

// -----

// CHECK-LABEL: @eval_splat_detection
module @eval_splat_detection {
  // CHECK: util.global private @[[EVALED:.+]] = dense<2> : tensor<2xi32>
  util.global private @hoisted : tensor<2xi32>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
    util.global.store %cst, @hoisted : tensor<2xi32>
    util.return
  }
  util.func public @main() -> tensor<2xi32> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xi32>
    util.return %hoisted : tensor<2xi32>
  }
}


// -----

// CHECK-LABEL: @eval_f16_tensor
// CHECK: util.global private @[[EVALED:.+]] = dense<2.000000e+02> : tensor<5x6xf16>
module @eval_f16_tensor {
  util.global private @hoisted : tensor<5x6xf16>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<2.0e+2> : tensor<5x6xf16>
    util.global.store %cst, @hoisted : tensor<5x6xf16>
    util.return
  }
  util.func public @main() -> tensor<5x6xf16> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<5x6xf16>
    util.return %hoisted : tensor<5x6xf16>
  }
}

// -----

// CHECK-LABEL: @eval_bf16_tensor
// CHECK: util.global private @[[EVALED:.+]] = dense<2.000000e+02> : tensor<5x6xbf16>
module @eval_bf16_tensor {
  util.global private @hoisted : tensor<5x6xbf16>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<2.0e+2> : tensor<5x6xbf16>
    util.global.store %cst, @hoisted : tensor<5x6xbf16>
    util.return
  }
  util.func public @main() -> tensor<5x6xbf16> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<5x6xbf16>
    util.return %hoisted : tensor<5x6xbf16>
  }
}

// -----

// CHECK-LABEL: @eval_f32_tensor
module @eval_f32_tensor {
  // CHECK: util.global private @[[EVALED:.+]] = dense<[2.000000e+02, 3.200000e+03]> : tensor<2xf32>
  util.global private @hoisted : tensor<2xf32>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2.0e+2, 3.2e+3]> : tensor<2xf32>
    util.global.store %cst, @hoisted : tensor<2xf32>
    util.return
  }
  util.func public @main() -> tensor<2xf32> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xf32>
    util.return %hoisted : tensor<2xf32>
  }
}

// -----

// CHECK-LABEL: @eval_f64_tensor
// CHECK: util.global private @[[EVALED:.+]] = dense<[2.000000e+02, 3.200000e+03]> : tensor<2xf64>
module @eval_f64_tensor {
  util.global private @hoisted : tensor<2xf64>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2.0e+2, 3.2e+3]> : tensor<2xf64>
    util.global.store %cst, @hoisted : tensor<2xf64>
    util.return
  }
  util.func public @main() -> tensor<2xf64> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xf64>
    util.return %hoisted : tensor<2xf64>
  }
}

// -----

// CHECK-LABEL: @eval_i1_tensor
module @eval_i1_tensor {
  // CHECK: util.global private @[[EVALED:.+]] = dense<[false, true, false, true, true, false]> : tensor<6xi1>
  util.global private @hoisted : tensor<6xi1>
  // CHECK-NOT: util.initializer
  util.initializer {
    // Note that the level we are testing at is a bit odd in the way i1 vs
    // i8 are handled.
    %cst = arith.constant dense<[0, 1, 0, 1, 1, 0]> : tensor<6xi8>
    %casted = arith.trunci %cst : tensor<6xi8> to tensor<6xi1>
    util.global.store %casted, @hoisted : tensor<6xi1>
    util.return
  }
  util.func public @main() -> tensor<6xi1> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<6xi1>
    util.return %hoisted : tensor<6xi1>
  }
}

// -----

// CHECK-LABEL: @eval_i4_tensor
module @eval_i4_tensor {
  util.global private @hoisted : tensor<5x6xi4>
  // expected-warning @+1 {{unsupported type for current jit configuration}}
  util.initializer {
    %cst = arith.constant dense<3> : tensor<5x6xi4>
    util.global.store %cst, @hoisted : tensor<5x6xi4>
    util.return
  }
  util.func public @main() -> tensor<5x6xi4> {
    %hoisted = util.global.load @hoisted : tensor<5x6xi4>
    util.return %hoisted : tensor<5x6xi4>
  }
}

// -----

// CHECK-LABEL: @eval_i8_tensor
module @eval_i8_tensor {
  // CHECK: util.global private @[[EVALED:.+]] = dense<[2, 3]> : tensor<2xi8>
  util.global private @hoisted : tensor<2xi8>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi8>
    util.global.store %cst, @hoisted : tensor<2xi8>
    util.return
  }
  util.func public @main() -> tensor<2xi8> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xi8>
    util.return %hoisted : tensor<2xi8>
  }
}

// -----

// CHECK-LABEL: @eval_i16_tensor
module @eval_i16_tensor {
  // CHECK: util.global private @[[EVALED:.+]] = dense<[2, 3]> : tensor<2xi16>
  util.global private @hoisted : tensor<2xi16>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi16>
    util.global.store %cst, @hoisted : tensor<2xi16>
    util.return
  }
  util.func public @main() -> tensor<2xi16> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xi16>
    util.return %hoisted : tensor<2xi16>
  }
}

// -----

// CHECK-LABEL: @eval_i32_tensor
module @eval_i32_tensor {
  // CHECK: util.global private @[[EVALED:.+]] = dense<[2, 3]> : tensor<2xi32>
  util.global private @hoisted : tensor<2xi32>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi32>
    util.global.store %cst, @hoisted : tensor<2xi32>
    util.return
  }
  util.func public @main() -> tensor<2xi32> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xi32>
    util.return %hoisted : tensor<2xi32>
  }
}

// -----

// CHECK-LABEL: @eval_i64_tensor
module @eval_i64_tensor {
  // CHECK: util.global private @[[EVALED:.+]] = dense<[2, 3]> : tensor<2xi64>
  util.global private @hoisted : tensor<2xi64>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<[2, 3]> : tensor<2xi64>
    util.global.store %cst, @hoisted : tensor<2xi64>
    util.return
  }
  util.func public @main() -> tensor<2xi64> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xi64>
    util.return %hoisted : tensor<2xi64>
  }
}

// -----

// Splat of an 8byte value ensures that large fills are possible.

// CHECK-LABEL: @eval_i64_tensor_splat
module @eval_i64_tensor_splat {
  // CHECK: util.global private @[[EVALED:.+]] = dense<2> : tensor<2xi64>
  util.global private @hoisted : tensor<2xi64>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = arith.constant dense<2> : tensor<2xi64>
    util.global.store %cst, @hoisted : tensor<2xi64>
    util.return
  }
  util.func public @main() -> tensor<2xi64> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<2xi64>
    util.return %hoisted : tensor<2xi64>
  }
}

// -----

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @serializable_attrs
module @serializable_attrs {
  util.global private @constant = #util.byte_pattern<1> : tensor<5x6xi8>
  // CHECK: util.global private @[[EVALED:.+]] = dense<2> : tensor<5x6xi8>
  util.global private @hoisted : tensor<5x6xi8>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst = util.global.load @constant : tensor<5x6xi8>
    %0 = tensor.empty() : tensor<5x6xi8>
    %1 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst, %cst : tensor<5x6xi8>, tensor<5x6xi8>) outs(%0 : tensor<5x6xi8>) {
    ^bb0(%arg0: i8, %arg1: i8, %arg2: i8):
      %2 = arith.addi %arg0, %arg1 : i8
      linalg.yield %2 : i8
    } -> tensor<5x6xi8>
    util.global.store %1, @hoisted : tensor<5x6xi8>
    util.return
  }
  util.func public @main() -> tensor<5x6xi8> {
    // CHECK: util.global.load @[[EVALED]]
    %hoisted = util.global.load @hoisted : tensor<5x6xi8>
    util.return %hoisted : tensor<5x6xi8>
  }
}

// -----

// CHECK-LABEL: @skip_dependent_initializers
module @skip_dependent_initializers {
  // A global that cannot be evaluated at compile-time because of an external
  // call. This taints the dependent global below.
  util.func private @extern() -> tensor<4xf32>
  // CHECK: util.global private @runtime_global
  util.global private @runtime_global : tensor<4xf32>
  // CHECK-NEXT: util.initializer
  // expected-warning @+1 {{skipping consteval initializer}}
  util.initializer {
    %0 = util.call @extern() : () -> tensor<4xf32>
    util.global.store %0, @runtime_global : tensor<4xf32>
    util.return
  }

  // A global that uses the non-evaluatable global that should not get evaled.
  // CHECK: util.global private @dependent_global
  util.global private @dependent_global : tensor<4xf32>
  // CHECK-NEXT: util.initializer
  // expected-warning @+1 {{skipping consteval initializer}}
  util.initializer {
    %cst = arith.constant dense<2.0> : tensor<4xf32>
    %0 = util.global.load immutable @runtime_global : tensor<4xf32>
    %1 = arith.addf %0, %cst : tensor<4xf32>
    util.global.store %1, @dependent_global : tensor<4xf32>
    util.return
  }
}

// -----

// TODO(benvanik): rewrite availability to use proper analysis - currently the
// pass uses ConstExprAnalysis which can't actually indicate what we want when
// we want it (here that this iota is available for evaluation at compile-time).
// This test is here as a reminder of what we should be able to do.

// CHECK-LABEL: @eval_op_with_no_inputs_currently_broken
module @eval_op_with_no_inputs_currently_broken {
  util.global private @hoisted : tensor<100xi64>
  // CHECK: util.initializer
  // expected-warning @+1 {{skipping consteval initializer}}
  util.initializer {
    %0 = tensor.empty() : tensor<100xi64>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} outs(%0 : tensor<100xi64>) {
    ^bb0(%out: i64):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i64
      linalg.yield %3 : i64
    } -> tensor<100xi64>
    util.global.store %1, @hoisted : tensor<100xi64>
    util.return
  }
}

// -----

// Tests that dispatches to inlined dispatch regions are JITed.
// This calculates 42 + 4 + 4 to ensure we can handle primitive and tensor arg
// constants.

// CHECK-LABEL: @dispatch_inline
module @dispatch_inline {
  // CHECK: util.global private @hoisted = dense<50> : tensor<4xi8>
  util.global private @hoisted : tensor<4xi8>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst0 = arith.constant 42 : i8
    %cst1 = arith.constant dense<4> : tensor<4xi8>
    %c0 = arith.constant 0 : index
    %x = tensor.dim %cst1, %c0 : tensor<4xi8>
    %0 = flow.dispatch.workgroups[%x](%cst0, %cst1) : (i8, tensor<4xi8>) -> tensor<4xi8> =
        (%arg0: i8, %arg1: !flow.dispatch.tensor<readonly:tensor<4xi8>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<4xi8>>) {
      %empty = tensor.empty() : tensor<4xi8>
      %input = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[4], strides=[1] : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<4xi8>
      %output = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%input, %input : tensor<4xi8>, tensor<4xi8>) outs(%empty : tensor<4xi8>) {
      ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):
        %addi_x2 = arith.addi %arg3, %arg4 : i8
        %result = arith.addi %addi_x2, %arg0 : i8
        linalg.yield %result : i8
      } -> tensor<4xi8>
      flow.dispatch.tensor.store %output, %arg2, offsets=[0], sizes=[4], strides=[1] : tensor<4xi8> -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
      flow.return
    }
    util.global.store %0, @hoisted : tensor<4xi8>
    util.return
  }
}

// -----

// Tests that dispatches to executable functions are JITed by cloning referenced
// executables to the JIT module. This calculates 42 + 4 + 4 to ensure we can
// handle primitive and tensor arg constants.

// CHECK-LABEL: @dispatch_executable
module @dispatch_executable {
  flow.executable private @exe {
    flow.executable.export public @dispatch_fn workgroups(%arg0: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func public @dispatch_fn(%arg0: i8, %arg1: !flow.dispatch.tensor<readonly:tensor<4xi8>>, %arg2: !flow.dispatch.tensor<writeonly:tensor<4xi8>>) {
        %empty = tensor.empty() : tensor<4xi8>
        %input = flow.dispatch.tensor.load %arg1, offsets=[0], sizes=[4], strides=[1] : !flow.dispatch.tensor<readonly:tensor<4xi8>> -> tensor<4xi8>
        %output = linalg.generic {
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = ["parallel"]
        } ins(%input, %input : tensor<4xi8>, tensor<4xi8>) outs(%empty : tensor<4xi8>) {
        ^bb0(%arg3: i8, %arg4: i8, %arg5: i8):
          %addi_x2 = arith.addi %arg3, %arg4 : i8
          %result = arith.addi %addi_x2, %arg0 : i8
          linalg.yield %result : i8
        } -> tensor<4xi8>
        flow.dispatch.tensor.store %output, %arg2, offsets=[0], sizes=[4], strides=[1] : tensor<4xi8> -> !flow.dispatch.tensor<writeonly:tensor<4xi8>>
        return
      }
    }
  }
  // CHECK: util.global private @hoisted = dense<50> : tensor<4xi8>
  util.global private @hoisted : tensor<4xi8>
  // CHECK-NOT: util.initializer
  util.initializer {
    %cst0 = arith.constant 42 : i8
    %cst1 = arith.constant dense<4> : tensor<4xi8>
    %c0 = arith.constant 0 : index
    %x = tensor.dim %cst1, %c0 : tensor<4xi8>
    %0 = flow.dispatch @exe::@dispatch_fn[%x](%cst0, %cst1) : (i8, tensor<4xi8>) -> tensor<4xi8>
    util.global.store %0, @hoisted : tensor<4xi8>
    util.return
  }
}
