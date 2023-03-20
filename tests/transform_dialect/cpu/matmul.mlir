
!A_size = tensor<3x5xf32>
!B_size = tensor<5x3xf32>
!C_size = tensor<3x3xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

// Run with C++ dispatch region formation but transform dialect codegen
// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline --iree-flow-transformation-pipeline \
// RUN:   --iree-flow-dispatch-generate-workload-region=false \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:   --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_custom_dispatch_formation_spec.mlir | \
// RUN: FileCheck %s --check-prefix=CODEGEN-CUSTOM-DISPATCH-FORMATION

// CODEGEN-CUSTOM-DISPATCH-FORMATION: hal.executable private @matmul_static_dispatch_0 {
// CODEGEN-CUSTOM-DISPATCH-FORMATION:   hal.executable.variant public @embedded_elf_{{.+}}, target = #executable_target_embedded_elf_{{.+}} {
// CODEGEN-CUSTOM-DISPATCH-FORMATION:     hal.executable.export public @matmul_static_dispatch_0_matmul_3x3x5 ordinal(0) layout(#{{.*}}) attributes {translation_info = #translation} {
// CODEGEN-CUSTOM-DISPATCH-FORMATION:       ^bb0(%{{.*}}: !hal.device):
// CODEGEN-CUSTOM-DISPATCH-FORMATION:         %[[C2:.*]] = arith.constant 2 : index
// CODEGEN-CUSTOM-DISPATCH-FORMATION:         %[[C1:.*]] = arith.constant 1 : index
// CODEGEN-CUSTOM-DISPATCH-FORMATION:         hal.return %[[C2]], %[[C1]], %[[C1]] : index, index, index
// CODEGEN-CUSTOM-DISPATCH-FORMATION:       }
// CODEGEN-CUSTOM-DISPATCH-FORMATION:       builtin.module {
// CODEGEN-CUSTOM-DISPATCH-FORMATION:         func.func @matmul_static_dispatch_0_matmul_3x3x5() {
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           arith.constant 0 : index
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset({{.*}}) flags(ReadOnly) : memref<3x5xf32>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           memref.assume_alignment %{{.*}}, 64 : memref<3x5xf32>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset({{.*}}) flags(ReadOnly) : memref<5x3xf32>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           memref.assume_alignment %{{.*}}, 64 : memref<5x3xf32>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset({{.*}}) : memref<3x3xf32>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           memref.assume_alignment %{{.*}}, 64 : memref<3x3xf32>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           %[[workgroup_id_x:.*]] = hal.interface.workgroup.id[0] : index
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           affine.apply {{.*}}(%workgroup_id_x)
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 5] [1, 1] : memref<3x5xf32> to memref<?x5xf32, strided<[5, 1], offset: ?>>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 3] [1, 1] : memref<3x3xf32> to memref<?x3xf32, strided<[3, 1], offset: ?>>
// CODEGEN-CUSTOM-DISPATCH-FORMATION:           linalg.matmul ins(%{{.*}}, %{{.*}} : memref<?x5xf32, strided<[5, 1], offset: ?>>, memref<5x3xf32>) outs(%{{.*}} : memref<?x3xf32, strided<[3, 1], offset: ?>>)

// RUN: iree-opt %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:   --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_default_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN-DEFAULT

// CODEGEN-DEFAULT: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CODEGEN-DEFAULT:     hal.executable.export public @matmul_static_dispatch_0_matmul_3x3x5
// CODEGEN-DEFAULT:       ^bb0(%[[DEVICE:[a-zA-Z0-9]+]]: !hal.device, %[[ARG0:[a-zA-Z0-9]+]]: index,
// CODEGEN-DEFAULT:         %[[C1:.+]] = arith.constant 1 : index
// CODEGEN-DEFAULT:         %[[D0:.+]] = affine.apply #[[MAP0]]()[%[[ARG0]]]
// CODEGEN-DEFAULT:         hal.return %[[D0]], %[[C1]], %[[C1]]

// RUN: iree-compile %s --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_default_spec.mlir | \
// RUN: iree-run-module --function=matmul_static \
// RUN:   --input="3x5xf32=1" \
// RUN:   --input="5x3xf32=2" \
// RUN:   --input="3x3xf32=42" | \
// RUN: FileCheck %s --check-prefixes=EXEC

// EXEC: 3x3xf32=[52 52 52][52 52 52][52 52 52]
