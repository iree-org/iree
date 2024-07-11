// Bootstrapped from this source IR:
//
// func.func @abs(%input : tensor<2xf32>) -> (tensor<2xf32>) {
//   %result = math.absf %input : tensor<2xf32>
//   return %result : tensor<2xf32>
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

      %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(4) offset(%c0) : !flow.dispatch.tensor<readonly:tensor<2xf32>>
      %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(4) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2xf32>>

      %2 = flow.dispatch.tensor.load %0, offsets = [0], sizes = [2], strides = [1] : !flow.dispatch.tensor<readonly:tensor<2xf32>> -> tensor<2xf32>
      %3 = tensor.empty() : tensor<2xf32>
      %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2 : tensor<2xf32>) outs(%3 : tensor<2xf32>) {
      ^bb0(%arg0: f32, %arg1: f32):
        %5 = math.absf %arg0 : f32
        linalg.yield %5 : f32
      } -> tensor<2xf32>
      flow.dispatch.tensor.store %4, %1, offsets = [0], sizes = [2], strides = [1] : tensor<2xf32> -> !flow.dispatch.tensor<writeonly:tensor<2xf32>>

      return
    }
  }
}
