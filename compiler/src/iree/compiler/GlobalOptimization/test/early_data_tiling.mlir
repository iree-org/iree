// RUN: iree-opt --iree-global-opt-early-data-tiling %s | FileCheck %s
// RUN: iree-opt --iree-global-opt-early-data-tiling --iree-dispatch-creation-assign-data-tiling-encodings %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(iree-global-opt-early-data-tiling,iree-global-opt-early-data-tiling)' %s | FileCheck %s
// RUN: iree-opt --pass-pipeline='builtin.module(iree-global-opt-early-data-tiling{data-tiling-op-types=convolution})' %s | FileCheck %s --check-prefix=CONV-ONLY
// RUN: iree-opt --pass-pipeline='builtin.module(iree-global-opt-early-data-tiling{encoding-option=padding})' %s | FileCheck %s --check-prefix=PADDING

// The epilogue is propagated and materialized with the contraction before any
// dispatches exist. Repeating the pass or late assignment preserves the layout.
// CHECK: module attributes {iree.encoding.materialized_layout_target =
// CHECK-LABEL: util.func public @matmul_relu
// CHECK: %[[LHS:.*]] = linalg.pack
// CHECK: %[[RHS:.*]] = linalg.pack
// CHECK: %[[INIT:.*]] = linalg.fill
// CHECK: %[[MM:.*]] = linalg.mmt4d ins(%[[LHS]], %[[RHS]]
// CHECK-SAME: outs(%[[INIT]] : tensor<8x8x8x8xf32>)
// CHECK: %[[RELU:.*]] = linalg.generic
// CHECK-SAME: ins(%[[MM]] : tensor<8x8x8x8xf32>)
// CHECK: arith.maximumf
// CHECK: %[[RESULT:.*]] = linalg.unpack %[[RELU]]
// CHECK-SAME: tensor<8x8x8x8xf32> -> tensor<64x64xf32>
// CHECK: util.return %[[RESULT]]

// With only convolutions selected, the matmul is left alone, and the module is
// still marked as taking the early route.
// CONV-ONLY: module attributes {iree.encoding.materialized_layout_target =
// CONV-ONLY-LABEL: util.func public @matmul_relu
// CONV-ONLY-NOT: linalg.mmt4d
// CONV-ONLY: linalg.matmul
// CONV-ONLY-NOT: linalg.mmt4d
// CONV-ONLY: util.return

// Only the default encoding strategy materializes into packed layouts. With
// padding, the module is left to dispatch-time data tiling and is not marked.
// PADDING-NOT: iree.encoding.materialized_layout_target
// PADDING-LABEL: util.func public @matmul_relu
// PADDING-NOT: iree_encoding
// PADDING: linalg.matmul
// PADDING-NOT: iree_encoding
// PADDING: util.return

module attributes {stream.affinity.default = #hal.device.affinity<@device>} {
  util.global private @device = #hal.device.target<"local", [#hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {target_triple = "aarch64-unknown-unknown-eabi-elf", cpu_features = "+neon", native_vector_size = 16 : i64, iree.encoding.resolver = #iree_cpu.cpu_encoding_resolver<>}>]> : !hal.device

  util.func public @matmul_relu(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<64x128xf32>, tensor<128x64xf32>) outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    %relu = linalg.generic {
      indexing_maps = [affine_map<(m, n) -> (m, n)>, affine_map<(m, n) -> (m, n)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<64x64xf32>) outs(%empty : tensor<64x64xf32>) {
    ^bb0(%value: f32, %unused: f32):
      %result = arith.maximumf %value, %zero : f32
      linalg.yield %result : f32
    } -> tensor<64x64xf32>
    util.return %relu : tensor<64x64xf32>
  }

  // Packing the broadcast bias is only an expansion of its shape. Resolve it
  // before dispatch formation so it cannot become a copy dispatch.
  // CHECK-LABEL: util.func public @matmul_bias
  // CHECK: %[[MM_BIAS:.*]] = linalg.mmt4d
  // CHECK: %[[BIAS:.*]] = tensor.expand_shape %arg2
  // CHECK-SAME: tensor<64xf32> into tensor<8x8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: ins(%[[MM_BIAS]], %[[BIAS]]
  util.func public @matmul_bias(%a: tensor<?x128xf32>, %b: tensor<128x64xf32>, %bias: tensor<64xf32>) -> tensor<?x64xf32> {
    %zero = arith.constant 0.0 : f32
    %c0 = arith.constant 0 : index
    %m = tensor.dim %a, %c0 : tensor<?x128xf32>
    %empty = tensor.empty(%m) : tensor<?x64xf32>
    %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<?x64xf32>) -> tensor<?x64xf32>
    %mm = linalg.matmul ins(%a, %b : tensor<?x128xf32>, tensor<128x64xf32>) outs(%fill : tensor<?x64xf32>) -> tensor<?x64xf32>
    %out = tensor.empty(%m) : tensor<?x64xf32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (j)>, affine_map<(i, j) -> (i, j)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%mm, %bias : tensor<?x64xf32>, tensor<64xf32>) outs(%out : tensor<?x64xf32>) {
    ^bb0(%x: f32, %bv: f32, %unused: f32):
      %y = arith.addf %x, %bv : f32
      linalg.yield %y : f32
    } -> tensor<?x64xf32>
    util.return %r : tensor<?x64xf32>
  }
}
