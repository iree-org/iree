// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmcpu-configuration-pipeline, iree-codegen-llvmcpu-lowering-pipeline))))' --split-input-file %s | FileCheck %s

// Smoke test: verify the full LLVMCPU pipeline (configuration + lowering
// including LLVM conversion) runs end-to-end on a simple dynamic elementwise
// add using the production IR structure with hal.executable nesting.

#pipeline_layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<
    "llvm-cpu", "embedded-elf-x86_64",
    {data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
     native_vector_size = 16 : index,
     target_triple = "x86_64-none-elf"}>
hal.executable private @elementwise_add_exe {
  hal.executable.variant public @embedded_elf_x86_64 target(#executable_target) {
    hal.executable.export public @elementwise_add ordinal(0) layout(#pipeline_layout) count(%device: !hal.device, %workload: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%workload)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @elementwise_add() {
        %c0 = arith.constant 0 : index
        %dim_i32 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i32
        %dim_cast = arith.index_castui %dim_i32 : i32 to index
        %dim = iree_tensor_ext.dispatch.workload.ordinal %dim_cast, 0 : index
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%dim}
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%dim}
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [%dim], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%dim} -> tensor<?xf32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [%dim], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xf32>>{%dim} -> tensor<?xf32>
        %5 = tensor.empty(%dim) : tensor<?xf32>
        %6 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>,
                             affine_map<(d0) -> (d0)>,
                             affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
            ins(%3, %4 : tensor<?xf32>, tensor<?xf32>)
            outs(%5 : tensor<?xf32>) {
        ^bb0(%in0: f32, %in1: f32, %out: f32):
          %7 = arith.addf %in0, %in1 : f32
          linalg.yield %7 : f32
        } -> tensor<?xf32>
        iree_tensor_ext.dispatch.tensor.store %6, %2, offsets = [0], sizes = [%dim], strides = [1] : tensor<?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xf32>>{%dim}
        return
      }
    }
  }
}

// Verify the pipeline produces LLVM IR.
// CHECK-LABEL: hal.executable private @elementwise_add_exe
//       CHECK:   llvm.func @elementwise_add
//       CHECK:     llvm.fadd
