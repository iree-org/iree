// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline))))" \
// RUN:    --split-input-file %s | FileCheck %s

// Make sure that the GPU configuration pipelines materialize encoding ops.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>, iree_codegen.target_info = #iree_gpu.target<arch = "gfx942", features = "", wgp = <compute =  int8, storage =  b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_I32_16x16x32_I8>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536, max_workgroup_counts = [2147483647, 2147483647, 2147483647], max_load_instruction_bits = 128, simds_per_wgp = 4, vgpr_space_bits = 16384>>, ukernels = "none"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [i8, i8, i32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [32768, 1280, ?]>
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  hal.executable private @executable {
    hal.executable.variant public @rocm_hsaco_fb target(#executable_target_rocm_hsaco_fb) {
      hal.executable.export public @export ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @set_encoding() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32768x1280xi8>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32768x1280xi8, #encoding>>
          %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32768, 1280], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32768x1280xi8>> -> tensor<32768x1280xi8>
          %3 = iree_encoding.set_encoding %2 : tensor<32768x1280xi8> -> tensor<32768x1280xi8, #encoding>
          iree_tensor_ext.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [32768, 1280], strides = [1, 1] : tensor<32768x1280xi8, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32768x1280xi8, #encoding>>
          return
        }
      }
    }
  }
}

// CHECK:      @set_encoding()
// CHECK:        linalg.pack
// CHECK:        tensor.expand_shape
// CHECK:        linalg.generic
