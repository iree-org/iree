// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>
hal.executable @static_unpack {
hal.executable.variant @cuda, target = <"cuda", "cuda-nvptx-fb"> {
  hal.executable.export @static_unpack layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2 : index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
  builtin.module {
    func.func @static_unpack() {
      %c0 = arith.constant 0 : index
      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<4x16x32x16xi32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x256xi32>>
      %2 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0, 0], sizes = [4, 16, 32, 16], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<4x16x32x16xi32>> -> tensor<4x16x32x16xi32>
      %3 = tensor.empty() : tensor<128x256xi32>
      %unpack = tensor.unpack %2 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %3 : tensor<4x16x32x16xi32> -> tensor<128x256xi32>
      flow.dispatch.tensor.store %unpack, %1, offsets = [0, 0], sizes = [128, 256], strides = [1, 1] : tensor<128x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<128x256xi32>>
      return
    }
  }
}
}
//   CHECK-LABEL:  func.func @static_unpack
//         CHECK:    scf.for
//         CHECK:      scf.for
//         CHECK:        vector.transfer_read
//         CHECK:        vector.transfer_write

