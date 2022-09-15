
!A_size = tensor<3x5xf32>
!B_size = tensor<5x3xf32>
!C_size = tensor<3x3xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-flow-dispatch-use-transform-dialect=%p/matmul_dispatch_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=DISPATCH

// TODO: make this test drop transform dialect usage at the flow level and use:
//   --iree-flow-transformation-pipeline --iree-flow-convert-region-to-workgroups
// Atm the 3rd flow.dispatch.tensor.load shows as readonly instead of readwrite.

// DISPATCH: flow.executable private @matmul_static_dispatch_0 {
// DISPATCH:   flow.executable.export public @matmul_static_dispatch_0_matmul_3x3x5
// DISPATCH:     builtin.module {
// DISPATCH:       func.func @matmul_static_dispatch_0_matmul_3x3x5
// DISPATCH:         flow.dispatch.tensor.load {{.*}}, offsets = [0, 0], sizes = [3, 5], strides = [1, 1] : !flow.dispatch.tensor<readonly:3x5xf32> -> tensor<3x5xf32>
// DISPATCH:         flow.dispatch.tensor.load {{.*}}, offsets = [0, 0], sizes = [5, 3], strides = [1, 1] : !flow.dispatch.tensor<readonly:5x3xf32> -> tensor<5x3xf32>
// DISPATCH:         flow.dispatch.tensor.load {{.*}}, offsets = [0, 0], sizes = [3, 3], strides = [1, 1] : !flow.dispatch.tensor<readwrite:3x3xf32> -> tensor<3x3xf32>
// DISPATCH:         linalg.matmul ins({{.*}} : tensor<3x5xf32>, tensor<5x3xf32>) outs({{.*}} : tensor<3x3xf32>) -> tensor<3x3xf32>
// DISPATCH:         flow.dispatch.tensor.store {{.*}} offsets = [0, 0], sizes = [3, 3], strides = [1, 1] : tensor<3x3xf32> -> !flow.dispatch.tensor<readwrite:3x3xf32>
// DISPATCH:         return

// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline --iree-flow-transformation-pipeline  --iree-flow-dispatch-use-transform-dialect=%p/matmul_dispatch_spec.mlir \
// RUN:   --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN

// Run with C++ dispatch region formation but transform dialect codegen
// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline --iree-flow-transformation-pipeline \
// RUN:   --iree-flow-dispatch-via-region-ops --iree-flow-dispatch-via-region-ops-generate-workload-region=false \
// RUN:   --iree-stream-transformation-pipeline --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN

// CODEGEN: hal.executable private @matmul_static_dispatch_0 {
// CODEGEN:   hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
// CODEGEN:     hal.executable.export public @matmul_static_dispatch_0_matmul_3x3x5 ordinal(0) layout(#{{.*}}) attributes {translation_info = #translation} {
// CODEGEN:       ^bb0(%{{.*}}: !hal.device):
// CODEGEN:         arith.constant 2 : index
// CODEGEN:         arith.constant 1 : index
// CODEGEN:         hal.return %{{.*}}, %{{.*}}, %{{.*}} : index, index, index
// CODEGEN:       }
// CODEGEN:       builtin.module {
// CODEGEN:         func.func @matmul_static_dispatch_0_matmul_3x3x5() {
// CODEGEN:           arith.constant 0 : index
// CODEGEN:           hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) offset({{.*}}) alignment(64) : memref<3x5xf32>
// CODEGEN:           memref.assume_alignment %{{.*}}, 64 : memref<3x5xf32>
// CODEGEN:           hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) offset({{.*}}) alignment(64) : memref<5x3xf32>
// CODEGEN:           memref.assume_alignment %{{.*}}, 64 : memref<5x3xf32>
// CODEGEN:           hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) offset({{.*}}) alignment(64) : memref<3x3xf32>
// CODEGEN:           memref.assume_alignment %{{.*}}, 64 : memref<3x3xf32>
// CODEGEN:           %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
// CODEGEN:           affine.apply {{.*}}()[%workgroup_id_x]
// CODEGEN:           memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 5] [1, 1] : memref<3x5xf32> to memref<?x5xf32, strided<[5, 1], offset: ?>>
// CODEGEN:           memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 3] [1, 1] : memref<3x3xf32> to memref<?x3xf32, strided<[3, 1], offset: ?>>
// CODEGEN:           linalg.matmul ins(%{{.*}}, %{{.*}} : memref<?x5xf32, strided<[5, 1], offset: ?>>, memref<5x3xf32>) outs(%{{.*}} : memref<?x3xf32, strided<[3, 1], offset: ?>>)

// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:    --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_default_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN-DEFAULT

// CODEGEN-DEFAULT: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CODEGEN-DEFAULT:     hal.executable.export public @matmul_static_dispatch_0_matmul_3x3x5
// CODEGEN-DEFAULT:       ^bb0(%[[DEVICE:[a-zA-Z0-9]+]]: !hal.device, %[[ARG0:[a-zA-Z0-9]+]]: index,
// CODEGEN-DEFAULT:         %[[C1:.+]] = arith.constant 1 : index
// CODEGEN-DEFAULT:         %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
// CODEGEN-DEFAULT:         hal.return %[[D0]], %[[C1]], %[[C1]]

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-flow-dispatch-use-transform-dialect=%p/matmul_dispatch_spec.mlir \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: iree-run-module --entry_function=matmul_static \
// RUN:   --function_input="3x5xf32=1 1 1 1 1 1 1 1 1 1 1 1 1 1 1" \
// RUN:   --function_input="5x3xf32=1 1 1 1 1 1 1 1 1 1 1 1 1 1 1" \
// RUN:   --function_input="3x3xf32=0 0 0 0 0 0 0 0 0"| \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 3x3xf32=[5 5 5][5 5 5][5 5 5]

// RUN: iree-compile --iree-hal-target-backends=llvm-cpu \
// RUN:     --iree-flow-dispatch-use-transform-dialect=%p/matmul_tiled_dispatch_spec.mlir \
// RUN:     --iree-flow-export-benchmark-funcs %s | \
// RUN: iree-benchmark-module --device=local-task | \
// RUN: FileCheck %s --check-prefixes=BENCHMARK-MODULE

// When running iree-benchmark-module, we only check the existence of the func.
// BENCHMARK-MODULE: matmul_static
