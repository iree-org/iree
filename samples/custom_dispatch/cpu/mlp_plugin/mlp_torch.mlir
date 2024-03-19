// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/mlp_torch_spec.pdl.mlir})" %s | \
// RUN: iree-compile - | \
// RUN: iree-run-module --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/mlp_plugin/mlp_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=mlp_invocation \
// RUN:     --input="2x4xf32=[[2.0, 2.0, 2.0, 2.0], [-2.0, -2.0, -2.0, -2.0]]" \
// RUN:     --input="4x8xf32=[[3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0], [3.0, -3.0, 3.0, -3.0]]"

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
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support llvm-cpu here.
#cpu_target = #hal.device.target<"llvm-cpu", [
  #x86_64_target
]>

#map = affine_map<(d0, d1) -> (d0, d1)>
module @example attributes {hal.device.targets = [#cpu_target]} {

  // CHECK-LABEL: EXEC @mlp_invocation
  //       CHECK: [Plugin]: M = 2, N = 8, K = 4
  //       CHECK: 2x8xf32=[-24 -0 -24 -0 -24 -0 -24 -0][-0 -24 -0 -24 -0 -24 -0 -24]
  func.func @mlp_invocation(%lhs: tensor<?x?xf32>,
                            %rhs: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %torch_lhs = torch_c.from_builtin_tensor %lhs : tensor<?x?xf32> -> !torch.vtensor<[?, ?], f32>
    %torch_rhs = torch_c.from_builtin_tensor %rhs : tensor<?x?xf32> -> !torch.vtensor<[?, ?], f32>
    %mm = torch.aten.mm %torch_lhs, %torch_rhs
        : !torch.vtensor<[?, ?], f32>, !torch.vtensor<[?, ?], f32> -> !torch.vtensor<[?, ?], f32>
    %relu = torch.aten.relu %mm : !torch.vtensor<[?, ?], f32> -> !torch.vtensor<[?, ?], f32>
    %cast= torch_c.to_builtin_tensor %relu : !torch.vtensor<[?, ?], f32> ->  tensor<?x?xf32>
    %negf = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types  = ["parallel", "parallel"]}
        ins(%cast : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.negf %b0 : f32
        linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    return %negf : tensor<?x?xf32>
  }
}  // module

// CHECK-LABEL: EXEC @mlp_invocation
//       CHECK: [Plugin]: M = 2, N = 8, K = 4
//       CHECK: 2x8xf32=[-24 -0 -24 -0 -24 -0 -24 -0][-0 -24 -0 -24 -0 -24 -0 -24]
