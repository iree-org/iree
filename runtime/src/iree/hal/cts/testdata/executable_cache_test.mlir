// Bootstrapped from this source IR:
//
// func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
//   %result = math.absf %input : tensor<f32>
//   return %result : tensor<f32>
// }

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @abs ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root()
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @abs() {
      %c0 = arith.constant 0 : index

      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<readonly:f32>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(32) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:f32>

      %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !iree_tensor_ext.dispatch.tensor<readonly:f32> -> tensor<f32>
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2 : tensor<f32>) outs(%3 : tensor<f32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %5 = math.absf %arg0 : f32
        linalg.yield %5 : f32
      } -> tensor<f32>
      iree_tensor_ext.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !iree_tensor_ext.dispatch.tensor<writeonly:f32>

      return
    }
  }
}
