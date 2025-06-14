// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-codegen-materialize-encoding-into-padding))" \
// RUN:   --iree-gpu-test-target=gfx942 \
// RUN:   --split-input-file %s | FileCheck %s

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#encoding_mmt = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16]>
#pad_encoding = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 64]>]>
func.func @set_pad_encoding_and_store() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect")
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
  %6 = iree_encoding.set_encoding %5 : tensor<2048x2048xf16> -> tensor<2048x2048xf16, #encoding_mmt>
  iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : tensor<2048x2048xf16, #encoding_mmt> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  return
}

// CHECK-LABEL: @set_pad_encoding_and_store
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2112xf16>>
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<2048x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2112xf16>>

// -----

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#encoding_mmt = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16]>
#pad_encoding = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 0]>]>
func.func @set_zero_pad_encoding_and_store() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect")
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
  %6 = iree_encoding.set_encoding %5 : tensor<2048x2048xf16> -> tensor<2048x2048xf16, #encoding_mmt>
  iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : tensor<2048x2048xf16, #encoding_mmt> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  return
}

// CHECK-LABEL: @set_zero_pad_encoding_and_store
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<2048x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>

// -----

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#encoding_mmt = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16]>
#pad_encoding = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 64]>]>
func.func @dynamic_set_zero_pad_encoding_and_store() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 2, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.constant.load layout(<constants = 2, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(1) : i32
  %dynamic_sz = arith.index_castui %2 : i32 to index
  %3 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect")
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>
  %4 = hal.interface.binding.subspan layout(<constants = 2, bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf16, #pad_encoding>>
  %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [%dynamic_sz, 2048], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>{%dynamic_sz} -> tensor<?x2048xf16>
  %6 = iree_encoding.set_encoding %5 : tensor<?x2048xf16> -> tensor<?x2048xf16, #encoding_mmt>
  iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, 0], sizes = [%dynamic_sz, 2048], strides = [1, 1]
    : tensor<?x2048xf16, #encoding_mmt> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf16, #pad_encoding>>{%dynamic_sz}
  return
}

// CHECK-LABEL: @dynamic_set_zero_pad_encoding_and_store
// CHECK:         %[[CST:.+]] = hal.interface.constant.load {{.+}} ordinal(1) : i32
// CHECK:         %[[SZ:.+]] = arith.index_castui %[[CST]]
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2112xf16>>
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [%[[SZ]], 2048], strides = [1, 1]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>{%[[SZ]]} -> tensor<?x2048xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [%[[SZ]], 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<?x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2112xf16>>{%[[SZ]]}

// -----

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#encoding_mmt_lhs = #iree_encoding.encoding<operand_index = 0 : index, op_type = matmul, element_types = [f16, f16, f16]>
#pad_encoding_lhs = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 64]>]>
#encoding_mmt_rhs = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f16, f16, f16]>
#pad_encoding_rhs = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 128]>]>
#encoding_mmt_out = #iree_encoding.encoding<operand_index = 2 : index, op_type = matmul, element_types = [f16, f16, f16]>
func.func @load_from_padded_and_mmt() {
  %c0 = arith.constant 0 : index
  %c8650752 = arith.constant 8650752 : index
  %c17301504 = arith.constant 17301504 : index
  %cst = arith.constant 0.000000e+00 : f16
  %0 = hal.interface.binding.subspan layout(<bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect")
         : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16, #pad_encoding_lhs>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%c8650752) flags("ReadOnly|Indirect")
         : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16, #pad_encoding_rhs>>
  %2 = hal.interface.binding.subspan layout(<bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c17301504) flags(Indirect)
         : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>
  %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
         : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16, #pad_encoding_lhs>> -> tensor<2048x2048xf16, #encoding_mmt_lhs>
  %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
         : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16, #pad_encoding_rhs>> -> tensor<2048x2048xf16, #encoding_mmt_rhs>
  %5 = tensor.empty() : tensor<2048x2048xf16, #encoding_mmt_out>
  %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2048x2048xf16, #encoding_mmt_out>) -> tensor<2048x2048xf16, #encoding_mmt_out>
  %7 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%3, %4 : tensor<2048x2048xf16, #encoding_mmt_lhs>, tensor<2048x2048xf16, #encoding_mmt_rhs>)
    outs(%6 : tensor<2048x2048xf16, #encoding_mmt_out>) {
  ^bb0(%in: f16, %in_0: f16, %out: f16):
    %9 = arith.mulf %in, %in_0 : f16
    %10 = arith.addf %out, %9 : f16
    linalg.yield %10 : f16
  } -> tensor<2048x2048xf16, #encoding_mmt_out>
  %8 = iree_encoding.unset_encoding %7 : tensor<2048x2048xf16, #encoding_mmt_out> -> tensor<2048x2048xf16>
  iree_tensor_ext.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : tensor<2048x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>
  return
}

// CHECK-LABEL: @load_from_padded_and_mmt
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2112xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2176xf16>>
// CHECK:         %[[C:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>
// CHECK:         %[[LD_A:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                    !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2112xf16>> -> tensor<2048x2048xf16>
// CHECK:         %[[LD_B:.+]] = iree_tensor_ext.dispatch.tensor.load %[[B]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                    !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2176xf16>> -> tensor<2048x2048xf16>
//
// CHECK:         tensor.empty() : tensor<2048x2048xf16>
// CHECK:         %[[FILL:.+]] = linalg.fill {{.+}} : tensor<2048x2048xf16>
// CHECK:         %[[MMT:.+]] = linalg.generic
// CHECK-SAME:      ins(%[[LD_A]], %[[LD_B]] : tensor<2048x2048xf16>, tensor<2048x2048xf16>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<2048x2048xf16>)
//
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[MMT]], %[[C]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<2048x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>

// -----

// This test gets the encoding resolver from executable target. The target is intentionally
// set to `gfx1100` so that the output is different from the default target, which ensures that
// the resolver is actually used.

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#pad_encoding = #iree_encoding.pad_encoding_layout<[0, ?]>
#executable_target = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_pad_layout<>,
    iree.gpu.target = #iree_gpu.target<arch = "gfx1100",
                                       features = "",
                                       wgp = <compute =  fp32,
                                              storage =  b32,
                                              subgroup =  arithmetic,
                                              dot =  none, mma = [],
                                              subgroup_size_choices = [32, 64],
                                              max_workgroup_sizes = [1024, 1024, 1024],
                                              max_thread_count_per_workgroup = 1024,
                                              max_workgroup_memory_bytes = 65536,
                                              max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>
  }>
func.func @set_pad_encoding_and_store_with_unresolved_encodings_from_executable() attributes { hal.executable.target = #executable_target } {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect")
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
  %6 = iree_encoding.set_encoding %5 : tensor<2048x2048xf16> -> tensor<2048x2048xf16, #pad_encoding>
  iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : tensor<2048x2048xf16, #pad_encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  return
}

// CHECK-LABEL: @set_pad_encoding_and_store_with_unresolved_encodings_from_executable
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<2048x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16>>

// -----

#binding_ro = #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
#binding = #hal.pipeline.binding<storage_buffer, Indirect>
#pad_encoding = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 64]>]>
func.func @set_pad_encoding_and_store_with_resolved_layout() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %3 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect")
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
  %4 = hal.interface.binding.subspan layout(<constants = 1, bindings = [#binding_ro, #binding], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  %5 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
  %6 = iree_encoding.set_encoding %5 : tensor<2048x2048xf16> -> tensor<2048x2048xf16, #pad_encoding>
  iree_tensor_ext.dispatch.tensor.store %6, %4, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
    : tensor<2048x2048xf16, #pad_encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2048xf16, #pad_encoding>>
  return
}

// CHECK-LABEL: @set_pad_encoding_and_store_with_resolved_layout
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>>
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2112xf16>>
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<2048x2048xf16>> -> tensor<2048x2048xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<2048x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2048x2112xf16>>

// -----

#encoding = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 32]>]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @materialize_pad_encoding_on_partial_dynamic_shape() {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i64
  %1 = arith.index_castui %0 : i64 to index
  %2 = util.assume.int %1<umin = 0, umax = 9007199254740991> : index
  %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x2048xf32>>
  %4 = iree_tensor_ext.dispatch.workload.ordinal %2, 0 : index
  %5 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32, #encoding>>{%4}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%4}
  %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%4, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf32, #encoding>>{%4} -> tensor<?x2048xf32, #iree_encoding.pad_encoding_layout<[0, ?]>>
  %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [4096, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x2048xf32>> -> tensor<4096x2048xf32>
  %9 = tensor.empty(%4) : tensor<?x4096xf32>
  %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x4096xf32>) -> tensor<?x4096xf32>
  %11 = iree_encoding.unset_encoding %7 : tensor<?x2048xf32, #iree_encoding.pad_encoding_layout<[0, ?]>> -> tensor<?x2048xf32>{%4}
  %12 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%11, %8 : tensor<?x2048xf32>, tensor<4096x2048xf32>) outs(%10 : tensor<?x4096xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %13 = arith.mulf %in, %in_0 : f32
    %14 = arith.addf %out, %13 : f32
    linalg.yield %14 : f32
  } -> tensor<?x4096xf32>
  iree_tensor_ext.dispatch.tensor.store %12, %6, offsets = [0, 0], sizes = [%4, 4096], strides = [1, 1] : tensor<?x4096xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x4096xf32>>{%4}
  return
}
// CHECK-LABEL: @materialize_pad_encoding_on_partial_dynamic_shape
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:      !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2080xf32>>
// CHECK:         iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [%{{.+}}, 2048], strides = [1, 1]
// CHECK-SAME:      !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2080xf32>>{%{{.+}}} -> tensor<?x2048xf32>

// -----

#encoding = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 32]>]>
#pipeline_layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#pipeline_layout1 = #hal.pipeline.layout<constants = 1, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @materialize_pad_encoding_dynamic_load_store() {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.constant.load layout(#pipeline_layout) ordinal(0) : i64
  %1 = arith.index_castui %0 : i64 to index
  %2 = util.assume.int %1<umin = 0, umax = 9007199254740991> : index
  %3 = hal.interface.constant.load layout(#pipeline_layout1) ordinal(0) : i32
  %4 = arith.index_castui %3 : i32 to index
  %5 = hal.interface.binding.subspan layout(#pipeline_layout1) binding(0) alignment(64) offset(%4) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>{%2}
  %6 = hal.interface.binding.subspan layout(#pipeline_layout1) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf16, #encoding>>{%2}
  %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%2, 2048], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>{%2} -> tensor<?x2048xf16>
  %8 = iree_encoding.set_encoding %7 : tensor<?x2048xf16> -> tensor<?x2048xf16, #iree_encoding.pad_encoding_layout<[0, ?]>>
  iree_tensor_ext.dispatch.tensor.store %8, %6, offsets = [0, 0], sizes = [%2, 2048], strides = [1, 1] : tensor<?x2048xf16, #iree_encoding.pad_encoding_layout<[0, ?]>> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2048xf16, #encoding>>{%2}
  return
}
// CHECK-LABEL: @materialize_pad_encoding_dynamic_load_store
// CHECK:         %[[A:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(0)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>{%{{.+}}}
// CHECK:         %[[B:.+]] = hal.interface.binding.subspan layout({{.+}}) binding(1)
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2080xf16>>{%{{.+}}}
// CHECK:         %[[LD:.+]] = iree_tensor_ext.dispatch.tensor.load %[[A]], offsets = [0, 0], sizes = [%{{.+}}, 2048], strides = [1, 1]
// CHECK-SAME:                  !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x2048xf16>>{%{{.+}}} -> tensor<?x2048xf16>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[LD]], %[[B]], offsets = [0, 0], sizes = [%{{.+}}, 2048], strides = [1, 1]
// CHECK-SAME:                  tensor<?x2048xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x2080xf16>>{%{{.+}}}
