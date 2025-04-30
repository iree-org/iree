// Bootstrapped from this source IR:
//
// func.func @abs(%input : tensor<2xf32>) -> (tensor<2xf32>) {
//   %result = math.absf %input : tensor<2xf32>
//   return %result : tensor<2xf32>
// }

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @abs ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @abs() {
      %c0 = arith.constant 0 : index

      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(4) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(4) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2xf32>>

      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [2], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
      %3 = tensor.empty() : tensor<2xf32>
      %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<2xf32>) outs(%3 : tensor<2xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %5 = math.absf %arg0 : f32
        linalg.yield %5 : f32
      } -> tensor<2xf32>
      iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2xf32>>

      return
    }
  }
}
