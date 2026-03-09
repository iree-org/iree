// Each workgroup writes its workgroup ID to the corresponding output element.
// Dispatched with (N, 1, 1) workgroups to produce output = [0, 1, ..., N-1].

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>

hal.executable.source public @executable {
  hal.executable.export public @write_workgroup_ids ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    hal.return %c32, %c1, %c1 : index, index, index
  } attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}
  builtin.module {
    func.func @write_workgroup_ids() {
      %out = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<32xi32>
      %id = hal.interface.workgroup.id[0] : index
      %id_i32 = arith.index_cast %id : index to i32
      memref.store %id_i32, %out[%id] : memref<32xi32>
      return
    }
  }
}
