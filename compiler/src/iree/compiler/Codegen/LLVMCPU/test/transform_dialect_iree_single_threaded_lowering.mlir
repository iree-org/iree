// RUN: iree-opt %s --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | FileCheck %s

hal.executable private @static_pack_simple_dispatch_0 {
hal.executable.variant public @embedded_elf_x86_64, target = <"llvm-cpu", "embedded-elf-x86_64", {cpu = "generic", cpu_features = "", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", native_vector_size = 16 : index, target_triple = "x86_64-unknown-unknown-eabi-elf"}> {
  // CHECK: hal.executable.export public
  // CHECK:   %[[C1:.+]] = arith.constant 1 : index
  // CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]]
  hal.executable.export public @static_pack_simple_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<TransformDialectCodegen>} {
  ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @static_pack_simple_dispatch_0() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x4xi32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2x2x2x2xi32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4, 4], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<4x4xi32>> -> tensor<4x4xi32>
      %3 = tensor.empty() : tensor<2x2x2x2xi32>
      %pack = tensor.pack %2 inner_dims_pos = [0, 1] inner_tiles = [2, 2] into %3 : tensor<4x4xi32> -> tensor<2x2x2x2xi32>
      flow.dispatch.tensor.store %pack, %1, offsets = [0, 0, 0, 0], sizes = [2, 2, 2, 2], strides = [1, 1, 1, 1] : tensor<2x2x2x2xi32> -> !flow.dispatch.tensor<writeonly:tensor<2x2x2x2xi32>>
      return
    }
  }
}
}

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %pack = transform.structured.match ops{["tensor.pack"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  transform.iree.lower_single_threaded_workgroup_count_region_op %pack
}
