// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target{test-lowering-configuration=true})))' --iree-codegen-llvmcpu-enable-transform-dialect-jit --split-input-file %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable private @pack  {
  hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {
    cpu_features = "+avx512f,+avx2",
    data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
    native_vector_size = 16 : index,
    target_triple = "x86_64-unknown-unknown-eabi-elf"
  }> {
    hal.executable.export public @pack layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @pack() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<384x512xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
        %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [384, 512], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<384x512xf32>> -> tensor<384x512xf32>
        %3 = tensor.empty() : tensor<24x512x16x1xf32>
        %pack = tensor.pack %2 inner_dims_pos = [0, 1] inner_tiles = [16, 1] into %3: tensor<384x512xf32> -> tensor<24x512x16x1xf32>
        flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [24, 512, 16, 1], strides = [1, 1, 1, 1] : tensor<24x512x16x1xf32> -> !flow.dispatch.tensor<writeonly:tensor<24x512x16x1xf32>>
        return
      }
    }
  }
}
// CHECK-DAG:   #[[TRANSLATION:.+]] = #iree_codegen.translation_info<TransformDialectCodegen>
// CHECK:         hal.executable.export public @pack
// CHECK-SAME:       translation_info = #[[TRANSLATION]]
// CHECK-LABEL: func.func @pack
// CHECK:         transform.sequence failures(propagate)
// CHECK-NEXT:    ^bb0(%[[VAR:.+]]: !pdl.operation)
// CHECK:         %[[PACK:.+]] = transform.structured.match ops{["tensor.pack"]} in %[[VAR]]
// CHECK:         %[[FORALL:.+]], %[[TILED_PACK:.+]] = transform.iree.tile_to_forall_and_workgroup_count_region
// CHECK-SAME:      %[[PACK:.+]] num_threads [1]
// CHECK:         %[[CAST:.+]] = cast %[[TILED_PACK]] : !pdl.operation to !transform.op<"tensor.pack">
// CHECK:         %[[PAD:.+]], %[[RESHAPE:.+]], %[[TRANS:.+]] =
// CHECK-SAME:      transform.structured.lower_pack %[[CAST]]
// CHECK:         %[[TILED_TRANS:.+]], %{{.+}} = transform.structured.tile %[[TRANS]][1, 8, 8, 1]
// CHECK:         %[[CAST:.+]] = cast %[[TILED_TRANS]] : !transform.op<"linalg.transpose"> to !pdl.operation
// CHECK:         %[[GEN:.+]] = transform.structured.generalize %[[CAST]]
// CHECK:         transform.iree.apply_patterns %[[VAR]]
// CHECK-SAME:      rank_reducing_linalg
// CHECK:         %{{.*}} = transform.structured.vectorize
// CHECK:         %{{.*}} = transform.structured.hoist_redundant_tensor_subsets
// CHECK:         %{{.*}} = transform.iree.bufferize %[[VAR]]
// CHECK:         transform.iree.erase_hal_descriptor_type_from_memref
// CHECK:         transform.iree.forall_to_workgroup
// CHECK:         %{{.*}} = transform.vector.apply_transfer_permutation_patterns
// CHECK:         %{{.*}} = transform.vector.split_transfer_full_partial
// CHECK:         %{{.*}} = transform.vector.transfer_to_scf
// CHECK:         %{{.*}} = transform.vector.lower_transfer
// CHECK:         %{{.*}} = transform.vector.lower_shape_cast
// CHECK:         %{{.*}} = transform.vector.lower_transpose
// CHECK-SAME:      lowering_strategy = shuffle
// CHECK-SAME:      avx2_lowering_strategy = true
