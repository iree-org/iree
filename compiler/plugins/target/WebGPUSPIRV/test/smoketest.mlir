// RUN: iree-opt --split-input-file --iree-hal-transformation-pipeline %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module attributes {
  hal.device.targets = [
    #hal.device.target<"webgpu", [
      #hal.executable.target<"webgpu-spirv", "webgpu-wgsl-fb", {
        iree.gpu.target = #iree_gpu.target<arch = "", features = "spirv:v1.0,cap:Shader,ext:SPV_KHR_storage_buffer_storage_class", wgp = <
          compute = fp32|int32, storage = b32, subgroup = none, dot = none, mma = [], subgroup_size_choices = [32],
          max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128, max_workgroup_memory_bytes = 16384,
          max_workgroup_counts = [65535, 65535, 65535]>>
      }>
    ]> : !hal.device
  ]
} {

stream.executable public @reduce_dispatch {
  stream.executable.export @reduce_dispatch workgroups(%arg0 : index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg0
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @reduce_dispatch(%arg0_binding: !stream.binding, %arg1_binding: !stream.binding) {
      %c0 = arith.constant 0 : index
      %arg0 = stream.binding.subspan %arg0_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
      %arg1 = stream.binding.subspan %arg1_binding[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
      %0 = tensor.empty() : tensor<f32>
      %1 = iree_tensor_ext.dispatch.tensor.load %arg0, offsets=[0], sizes=[16], strides=[1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>> -> tensor<16xf32>
      %3 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%1 : tensor<16xf32>) outs(%0 : tensor<f32>) {
      ^bb0(%arg2: f32, %arg3: f32):
        %4 = arith.addf %arg2, %arg3 : f32
        linalg.yield %4 : f32
      } -> tensor<f32>
      iree_tensor_ext.dispatch.tensor.store %3, %arg1, offsets=[], sizes=[], strides=[] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<f32>>
      return
    }
  }
}

}

//      CHECK:   hal.executable.binary public @webgpu_wgsl_fb attributes
// CHECK-SAME:     data = dense
// CHECK-SAME:     format = "webgpu-wgsl-fb"
