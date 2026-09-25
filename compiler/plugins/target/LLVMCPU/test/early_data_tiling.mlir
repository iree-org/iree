// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --compile-to=global-optimization -o %t.global
// RUN: FileCheck %s --check-prefix=GLOBAL --implicit-check-not=flow.dispatch --implicit-check-not=iree_encoding.set_encoding < %t.global
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --compile-to=global-optimization | FileCheck %s --check-prefix=UNPACKED --implicit-check-not=materialized_layout_target --implicit-check-not=linalg.pack
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --iree-dispatch-creation-experimental-set-data-tiling-ops=scaled_matmul --compile-to=global-optimization | FileCheck %s --check-prefix=UNPACKED --implicit-check-not=linalg.pack
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --iree-dispatch-creation-set-encoding-strategy=padding --compile-to=global-optimization | FileCheck %s --check-prefix=UNPACKED --implicit-check-not=materialized_layout_target --implicit-check-not=linalg.pack
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=aarch64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --iree-opt-data-tiling --iree-global-opt-data-tiling --iree-dispatch-creation-data-tiling --compile-to=dispatch-creation | FileCheck %s --check-prefix=DISPATCH --implicit-check-not=flow.tensor.encode
// RUN: iree-compile %t.global --iree-opt-experimental-early-data-tiling --compile-from=global-optimization --compile-to=dispatch-creation | FileCheck %s --check-prefix=DISPATCH --implicit-check-not=flow.tensor.encode
// RUN: iree-compile %s --iree-hal-target-backends=vmvx --iree-opt-experimental-early-data-tiling --compile-to=dispatch-creation | FileCheck %s --check-prefix=LATE --implicit-check-not=materialized_layout_target
// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf --iree-opt-experimental-early-data-tiling --compile-to=global-optimization | FileCheck %s --check-prefix=GLOBAL --implicit-check-not=flow.dispatch --implicit-check-not=iree_encoding.set_encoding

// GLOBAL: iree.encoding.materialized_layout_target =
// GLOBAL-LABEL: util.func public @matmul_relu
// GLOBAL: %[[MM:.*]] = linalg.mmt4d
// GLOBAL: linalg.generic
// GLOBAL-SAME: ins(%[[MM]] : tensor<{{[0-9]+x[0-9]+x[0-9]+x[0-9]+}}xf32>)
// GLOBAL: arith.maximumf
// GLOBAL: linalg.unpack

// UNPACKED-LABEL: util.func public @matmul_relu
// UNPACKED: linalg.matmul
// UNPACKED-SAME: tensor<64x128xf32>, tensor<128x64xf32>
// UNPACKED: arith.maximumf

// DISPATCH: iree.encoding.materialized_layout_target =
// DISPATCH-LABEL: util.func public @matmul_relu
// DISPATCH: flow.dispatch.workgroups
// DISPATCH: linalg.mmt4d
// DISPATCH-NOT: flow.return
// DISPATCH: arith.maximumf
// DISPATCH-NOT: flow.return
// DISPATCH: linalg.unpack
// DISPATCH: flow.return

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
