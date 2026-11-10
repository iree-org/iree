// RUN: iree-compile %s --iree-hal-target-backends=vmvx --iree-opt-experimental-early-data-tiling --compile-to=dispatch-creation | FileCheck %s --check-prefix=LATE --implicit-check-not=materialized_layout_target
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --compile-to=global-optimization | FileCheck %s --check-prefix=GLOBAL --implicit-check-not=flow.dispatch --implicit-check-not=iree_encoding.set_encoding

// GLOBAL: iree.encoding.materialized_layout_target =
// GLOBAL-LABEL: util.func public @matmul_relu
// GLOBAL: %[[MM:.*]] = linalg.mmt4d
// GLOBAL: linalg.generic
// GLOBAL-SAME: ins(%[[MM]] : tensor<{{[0-9]+x[0-9]+x[0-9]+x[0-9]+}}xf32>)
// GLOBAL: arith.maximumf
// GLOBAL: linalg.unpack

// LATE-LABEL: util.func public @matmul_relu
// LATE: flow.tensor.encode
// LATE-SAME: #iree_encoding.encoding<
// LATE: flow.dispatch.workgroups
// LATE: linalg.matmul
// LATE-SAME: #iree_encoding.encoding<
// LATE: arith.maximumf

  func.func @matmul_relu(%lhs: tensor<64x128xf32>, %rhs: tensor<128x64xf32>) -> tensor<64x64xf32> {
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
    return %relu : tensor<64x64xf32>
  }
