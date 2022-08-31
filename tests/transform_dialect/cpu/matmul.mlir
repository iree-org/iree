
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
// RUN:   --iree-abi-transformation-pipeline \
// RUN:   --iree-flow-transformation-pipeline \
// RUN:   --iree-flow-dispatch-use-transform-dialect=%p/matmul_dispatch_spec.mlir \
// RUN:   --iree-stream-transformation-pipeline \
// RUN:    --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target))' \
// RUN:   --iree-codegen-llvmcpu-use-transform-dialect=%p/matmul_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefixes=CODEGEN

// CODEGEN: hal.executable private @matmul_static_dispatch_0 {
// CODEGEN:   hal.executable.variant public @embedded_elf_x86_64, target = #executable_target_embedded_elf_x86_64_ {
//
// The signature of the hal.executable.export region is subject to conventions
// at the flow level. These conventions are materialized in IR e.g. into 
// stream.cmd.dispatch before codegen gets invoked.
// As a consequence, the tile_size/num_threads/workgroup_count passed to 
// transform.tile_to_foreach_thread needs to be aware of this convention.
// For now we use our own convention that sizes are static and no other bbArg
// than !hal.device is present.
// 
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
// CODEGEN:           memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 5] [1, 1] : memref<3x5xf32> to memref<?x5xf32, #{{.*}}>
// CODEGEN:           memref.subview %{{.*}}[%{{.*}}, 0] [%{{.*}}, 3] [1, 1] : memref<3x3xf32> to memref<?x3xf32, #{{.*}}>
// CODEGEN:           linalg.matmul ins(%{{.*}}, %{{.*}} : memref<?x5xf32, #map3>, memref<5x3xf32>) outs(%{{.*}} : memref<?x3xf32, #{{.*}}>)

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
