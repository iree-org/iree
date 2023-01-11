// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' -split-input-file %s | FileCheck %s


#compilation = #iree_codegen.compilation_info<
    lowering_config = <tile_sizes = [[127, 255], [8, 32], [0, 0]]>,
    translation_info  = <CPUDoubleTilingExpert>,
    workgroup_size = []>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
hal.executable private @preset_config_generic_add  {
  hal.executable.variant @system_elf_x86_64, target = <"llvm-cpu", "system-elf-x86_64"> {
    hal.executable.export @mask_dynamic_generic_add layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index, %arg3 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @mask_dynamic_generic_add() {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = hal.interface.constant.load[0] : i32
        %1 = hal.interface.constant.load[1] : i32
        %6 = arith.index_cast %0 : i32 to index
        %7 = arith.index_cast %1 : i32 to index
        %lhs_binding = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %rhs_binding = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer)
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7}
        %result_binding = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer)
            : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        %lhs = flow.dispatch.tensor.load %lhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %rhs = flow.dispatch.tensor.load %rhs_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%6, %7} -> tensor<?x?xf32>
        %init = tensor.empty(%6, %7) : tensor<?x?xf32>
        %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %generic = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>,
                                                    affine_map<(d0, d1) -> (d0, d1)>],
                                   iterator_types = ["parallel", "parallel"]}
          ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>) outs(%fill : tensor<?x?xf32>) {
            ^bb0(%in0: f32, %in1: f32, %out: f32):
          %add = arith.addf %in0, %in1 : f32
          linalg.yield %add: f32
        } -> tensor<?x?xf32>
        flow.dispatch.tensor.store %generic, %result_binding, offsets = [0, 0], sizes = [%6, %7], strides = [1, 1]
            : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%6, %7}
        return
      }
    }
  }
}

// Masking is not applied to the main vector loop when the peeling is used.

// CHECK-LABEL: func.func @mask_dynamic_generic_add
// Main loop
// CHECK:       scf.for
// CHECK:         vector.load
// CHECK:         vector.load
// CHECK:         vector.store
// Peel loop
// CHECK:       scf.for
// CHECK:         vector.maskedload
// CHECK:         vector.maskedload
// CHECK:         vector.maskedstore

