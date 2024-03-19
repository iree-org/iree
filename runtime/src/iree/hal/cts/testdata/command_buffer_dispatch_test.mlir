// Bootstrapped from this source IR:
//
// func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
//   %result = math.absf %input : tensor<f32>
//   return %result : tensor<f32>
// }

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>
  ]>
]>

hal.executable.source public @executable {
  hal.executable.export public @abs ordinal(0) layout(#pipeline_layout) {
  ^bb0(%arg0: !hal.device):
    %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root
    hal.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @abs() {
      %c0 = arith.constant 0 : index

      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<readonly:f32>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%c0) : !flow.dispatch.tensor<writeonly:f32>

      %2 = flow.dispatch.tensor.load %0, offsets = [], sizes = [], strides = [] : !flow.dispatch.tensor<readonly:f32> -> tensor<f32>
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.generic {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>], iterator_types = []} ins(%2 : tensor<f32>) outs(%3 : tensor<f32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %5 = math.absf %arg0 : f32
        linalg.yield %5 : f32
      } -> tensor<f32>
      flow.dispatch.tensor.store %4, %1, offsets = [], sizes = [], strides = [] : tensor<f32> -> !flow.dispatch.tensor<writeonly:f32>

      return
    }
  }
}
