// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' -split-input-file %s | FileCheck %s

// Test peeling + vectorization using CPUDoubleTilingPeelingExpert.

#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[64, 64, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>,
    translation_info  = <CPUDoubleTilingPeelingExpert>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul  {
  hal.executable.variant @system_elf_x86_64 target(<"llvm-cpu", "system-elf-x86_64">) {
    hal.executable.export @no_peel_static_matmul layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @no_peel_static_matmul() {
        %cst = arith.constant 0.000000e+00 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<128x64xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<64x512xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [128, 64], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x64xf32>> -> tensor<128x64xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [64, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<64x512xf32>> -> tensor<64x512xf32>
        %init = tensor.empty() : tensor<128x512xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<128x64xf32>, tensor<64x512xf32>)
            outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        return
      }
    }
  }
}

// No peel loop should be generated since dims are multiple of the vector dims.

// CHECK-LABEL: func @no_peel_static_matmul()
// Vectorization:
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               vector.fma
// CHECK-NOT: scf.for

// -----

#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[65, 65, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>,
    translation_info  = <CPUDoubleTilingPeelingExpert>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul  {
  hal.executable.variant @system_elf_x86_64 target(<"llvm-cpu", "system-elf-x86_64">) {
    hal.executable.export @peel_static_matmul layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @peel_static_matmul() {
        %cst = arith.constant 0.000000e+00 : f32
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<128x49xf32>>
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<49x512xf32>>
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [128, 49], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<128x49xf32>> -> tensor<128x49xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [49, 512], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<49x512xf32>> -> tensor<49x512xf32>
        %init = tensor.empty() : tensor<128x512xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<128x512xf32>) -> tensor<128x512xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<128x49xf32>, tensor<49x512xf32>)
            outs(%fill : tensor<128x512xf32>) -> tensor<128x512xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [128, 512], strides = [1, 1]
            : tensor<128x512xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x512xf32>>
        return
      }
    }
  }
}

// Peel loops should be generated for the 2nd and 3rd dims they are not multiple of the vector dims.

// CHECK-LABEL: func @peel_static_matmul()
// Vectorization:
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               vector.fma

// 2nd dim peeling:
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               linalg.matmul

// 3rd dim peeling:
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               linalg.matmul

// CHECK-NOT: scf.for

// -----

#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[64, 64, 0], [8, 32, 0], [0, 0, 16], [0, 0, 0]]>,
    translation_info  = <CPUDoubleTilingPeelingExpert>>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_matmul  {
  hal.executable.variant @system_elf_x86_64 target(<"llvm-cpu", "system-elf-x86_64">) {
    hal.executable.export @peel_dynamic_matmul layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @peel_dynamic_matmul() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %2 = hal.interface.constant.load[2] : i32
        %dim0 = arith.index_cast %0 : i32 to index
        %dim1 = arith.index_cast %1 : i32 to index
        %dim2 = arith.index_cast %2 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim1, %dim0}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim2}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim1, %dim2}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%dim1, %dim0], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim1, %dim0} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%dim0, %dim2], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%dim0, %dim2} -> tensor<?x?xf32>
        %init = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %gemm = linalg.matmul {compilation_info = #compilation}
            ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
        flow.dispatch.tensor.store %gemm, %result_binding, offsets = [0, 0], sizes = [%dim1, %dim2], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%dim1, %dim2}
        return
      }
    }
  }
}

// Peel loops should be generated for all the dims since they are dynamic.

// CHECK-LABEL: func @peel_dynamic_matmul()
// Distribution:
// CHECK:         scf.for
// CHECK:           scf.for

// Vectorization:
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   vector.fma

// 1nd dim peeling:
// CHECK:                 scf.for
// CHECK:                   linalg.matmul

// 2nd dim peeling:
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   linalg.matmul

// 3nd dim peeling:
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   linalg.matmul

// CHECK-NOT: scf.for
