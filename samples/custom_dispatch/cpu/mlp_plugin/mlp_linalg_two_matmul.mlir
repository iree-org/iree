// RUN: iree-compile --iree-preprocessing-pdl-spec-filename=%p/mlp_linalg_spec.pdl.mlir %s | \
// RUN:   iree-run-module --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/mlp_plugin/mlp_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=mlp_invocation \
// RUN:     --input="8x8xf32=6" \
// RUN:     --input="8x8xf32=5" | \
// RUN:   FileCheck %s

// RUN: iree-compile --iree-preprocessing-transform-spec-filename=%p/mlp_spec_matmul.mlir %s | \
// RUN:   iree-run-module --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/mlp_plugin/mlp_plugin$IREE_DYLIB_EXT \
// RUN:     --module=- \
// RUN:     --function=mlp_invocation \
// RUN:     --input="8x8xf32=6" \
// RUN:     --input="8x8xf32=5" | \
// RUN:   FileCheck %s --check-prefix=TRANSFORM


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
]> : !hal.device

#map = affine_map<(d0, d1) -> (d0, d1)>
module @example attributes {hal.device.targets = [#cpu_target]} {
  func.func @mlp_invocation(%lhs: tensor<?x?xf32>,
                            %rhs: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.0 : f32
    %dim0 = tensor.dim %lhs, %c0 : tensor<?x?xf32>
    %dim1 = tensor.dim %rhs, %c1 : tensor<?x?xf32>
    %empty = tensor.empty(%dim0, %dim1) : tensor<?x?xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %matmul = linalg.matmul ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
    %matmul2 = linalg.matmul ins(%matmul, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
    %relu = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%matmul2 : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.maximumf %b0, %cst : f32
        linalg.yield %0 : f32
      } -> tensor<?x?xf32>
    %neg = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types  = ["parallel", "parallel"]}
        ins(%relu : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) {
      ^bb0(%b0 : f32, %b1 : f32):
        %0 = arith.negf %b0 : f32
        linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    return %neg : tensor<?x?xf32>
  }
}  // module

  // CHECK-LABEL: EXEC @mlp_invocation
  //       CHECK: [Plugin]: M = 8, N = 8, K = 8
  //       CHECK: [Plugin]: M = 8, N = 8, K = 8
  //       CHECK: 8x8xf32=[-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600]

  // TRANSFORM-LABEL: EXEC @mlp_invocation
  //       TRANSFORM: [Plugin]: M = 8, N = 8, K = 8
  //       TRANSFORM: [Plugin]: M = 8, N = 8, K = 8
  //       TRANSFORM: 8x8xf32=[-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600][-9600 -9600 -9600 -9600 -9600 -9600 -9600 -9600]
