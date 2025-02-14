// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/mlp_tosa_spec.pdl.mlir})" %s | \
// RUN: iree-compile - | \
// RUN: iree-run-module --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/mlp_plugin/mlp_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=mlp_invocation \
// RUN:     --input="2x4xf32=[[2.0, 2.0, 2.0, 2.0], [-2.0, -2.0, -2.0, -2.0]]" \
// RUN:     --input="4x8xf32=[[3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0]]" | \
// RUN: FileCheck %s

// Rewrite function to rewrite a matched DAG into a flow.dispatch. Conceptually,
// the matched DAG at the tensor level gets replaced by a function
//
// ```
//   <results> = <external fn>(<input operands>, <initial value of results>,
//   <other operands>)
// ```
//
// `<other operands>` is handled same as `<input operands>`. The split is to
// allow freedom for where the result buffers are passed in through the ABI.
// `<results>` and `<initial values of result>` get tied to the same `memref`.
// So conceptually, at a `memref` level the DAG gets replaced by
//
// ```
//   <external fn>(<input operands>, <result operands in-out>, <other operands>)
// ```
//
// Each buffer object (input or output) is passed as a `pointer, offset` pair
// and value at location `index` is expected to be accessed as `pointer[offset +
// index]` (note: `offset` is number of elements)

#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf",
  cpu = "generic"
}>

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support llvm-cpu here.
#cpu_target = #hal.device.target<"llvm-cpu", [
  #x86_64_target
]> : !hal.device

module @example attributes {hal.device.targets = [#cpu_target]} {
  func.func @mlp_invocation(%lhs: tensor<2x4xf32>, %rhs : tensor<4x8xf32>) -> tensor<2x8xf32> {
    %lhs_shape = tosa.const_shape {value = dense<[1, 2, 4]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %rhs_shape = tosa.const_shape {value = dense<[1, 4, 8]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %lhs_3D = tosa.reshape %lhs, %lhs_shape : (tensor<2x4xf32>, !tosa.shape<3>) -> tensor<1x2x4xf32>
    %rhs_3D = tosa.reshape %rhs, %rhs_shape : (tensor<4x8xf32>, !tosa.shape<3>) -> tensor<1x4x8xf32>
    %0 = tosa.matmul %lhs_3D, %rhs_3D : (tensor<1x2x4xf32>, tensor<1x4x8xf32>) -> tensor<1x2x8xf32>
    %1 = tosa.clamp %0 {
        min_val = 0.0 : f32, max_val = 3.4028235e+38 : f32}
        : (tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
    %2 = tosa.negate %1 : (tensor<1x2x8xf32>) -> tensor<1x2x8xf32>
    %result_shape = tosa.const_shape {value = dense<[2, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.reshape %2, %result_shape : (tensor<1x2x8xf32>, !tosa.shape<2>) -> tensor<2x8xf32>
    return %3 : tensor<2x8xf32>
  }
}
// CHECK-LABEL: EXEC @mlp_invocation
//       CHECK: [Plugin]: M = 2, N = 8, K = 4, doRelu = 1
//       CHECK: 2x8xf32=[-24 -0 -24 -0 -24 -0 -24 -0][-0 -24 -0 -24 -0 -24 -0 -24]
