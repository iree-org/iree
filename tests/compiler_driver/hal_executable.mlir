// RUN: iree-compile --compile-mode=hal-executable \
// RUN:   --mlir-print-ir-after=iree-hal-serialize-executables \
// RUN:   --iree-hal-target-backends=vmvx %s \
// RUN:   --o=/dev/null 2>&1 | FileCheck %s

// Each entry point has a layout specification indicating the total number of
// push constants available and the descriptor sets and their bindings.
// Push constants are dense (0..N) while the sets/bindings are sparse and may
// contain unused or omitted entries.
#pipeline_layout = #hal.pipeline.layout<push_constants = 1, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>

// A single executable source definition is allowed per translation in this mode
// as linking and multi-executable embedding support requires our host-side IR.
hal.executable.source public @executable {
  // Exported functions are declared with the layout they use and may optionally
  // contain other information - though when hand-authoring that's usually
  // omitted.
  hal.executable.export public @mul layout(#pipeline_layout) {
    ^bb0(%arg0: !hal.device):
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
  }

  // The inner module defining the executable. This may have any number of
  // private functions and only those with declared entry points will be
  // exported.
  builtin.module {
    func.func @mul() {
      // Push constants are loaded by ordinal.
      %offset = hal.interface.constant.load[0] : index
      %length = hal.interface.constant.load[1] : index

      // Bindings are dereferenced by their set/binding ordinal and may have a
      // byte offset from the base of the descriptor. Alignment information when
      // available can help code generation emit better loads/stores.
      %s0b0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %s0b1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(32) offset(%offset) : !flow.dispatch.tensor<readonly:tensor<4xf32>>
      %s0b2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(32) : !flow.dispatch.tensor<writeonly:tensor<4xf32>>

      // Workgroup information can be queried from the interface.
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index
      %workgroup_size_x = hal.interface.workgroup.size[0] : index

      // Actual program:
      %base_index = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_id_x, %workgroup_size_x]
      %per_step = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%workgroup_count_x, %workgroup_size_x]
      scf.for %arg0 = %base_index to %length step %per_step {
        %5 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + 4)>(%arg0)[%workgroup_size_x]
        %6 = flow.dispatch.tensor.load %s0b0, offsets = [%arg0], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<?xf32>
        %7 = flow.dispatch.tensor.load %s0b1, offsets = [%arg0], sizes = [%5], strides = [1] : !flow.dispatch.tensor<readonly:tensor<4xf32>> -> tensor<?xf32>
        %8 = tensor.empty(%5) : tensor<?xf32>
        %9 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%6, %7 : tensor<?xf32>, tensor<?xf32>) outs(%8 : tensor<?xf32>) attrs =  {name = "mul.1"} {
        ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
          %s0b10 = arith.mulf %arg1, %arg2 : f32
          linalg.yield %s0b10 : f32
        } -> tensor<?xf32>
        flow.dispatch.tensor.store %9, %s0b2, offsets = [%arg0], sizes = [%5], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<4xf32>>
      }

      return
    }
  }
}

// Just check that there's the expected flatbuffers prefix bytes.
// CHECK: hal.executable.binary public @vmvx_bytecode_fb attributes {data = dense<{{.+}}> : vector<{{.+}}xi8>, format = "vmvx-bytecode-fb", mime_type = "application/x-flatbuffers"}
