// RUN: iree-opt --split-input-file --iree-stream-specialize-encodings %s | FileCheck %s

//------------------------------------------------------------------------------
// Stream ops that have TensorPhaseOp trait. This test suite tests that the
// encoding is updated that carries resolved layouts.
//------------------------------------------------------------------------------

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding = #iree_cpu.vmvx_encoding_layout<>}>
#executable_target_x86_64 = #hal.executable.target<"llvm-cpu", "xyz", {encoding = #iree_cpu.cpu_encoding_layout<>, target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>
module {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_

  util.func public @tensor_sizeof(%d0: index, %d1: index) -> (index, index) {
    %size0 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
    util.return %size0, %size1 : index, index
  }
}
// CHECK:       #[[$ENCODING0:.+]] = #iree_encoding.encoding
// CHECK-SAME:    #iree_cpu.vmvx_encoding_layout
// CHECK-SAME:    encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK:       #[[$ENCODING1:.+]] = #iree_encoding.encoding
// CHECK-SAME:    #iree_cpu.cpu_encoding_layout
// CHECK-SAME:    encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK-LABEL: util.func public @tensor_sizeof
// CHECK:         %[[D0_RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING0]]>
// CHECK:         %[[D1_RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING1]]>
// CHECK:         return %[[D0_RES]], %[[D1_RES]]

// -----

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding = #iree_cpu.vmvx_encoding_layout<>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>
module {
  util.global private @device_a = #device_target_local_0_

  util.func public @ops_with_result_encoding_only(%arg0: index, %arg1: index, %scalar_f32 : f32) {
    %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
    %barrier_0 = util.optimization_barrier %0 : !stream.resource<*>
    %1 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
    %2 = stream.tensor.splat on(#hal.device.affinity<@device_a>) %scalar_f32 : f32 -> tensor<?x1x10xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
    %barrier_1 = util.optimization_barrier %2 : !stream.resource<*>
    util.return
  }
}
// CHECK:       #[[$ENCODING:.+]] = #iree_encoding.encoding
// CHECK-SAME:    #iree_cpu.vmvx_encoding_layout
// CHECK-SAME:    encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK:       #[[TARGET:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE:.+]] = #[[TARGET]]
// CHECK-LABEL: util.func public @ops_with_result_encoding_only
// CHECK:         stream.tensor.empty on(#hal.device.affinity<@[[$DEVICE]]>) : tensor<?x0xf32, #[[$ENCODING]]>
// CHECK:         stream.tensor.constant {{.+}} : tensor<1x5x64xf32>
// CHECK:         stream.tensor.splat on(#hal.device.affinity<@[[$DEVICE]]>) {{.+}} -> tensor<?x1x10xf32, #[[$ENCODING]]>
// CHECK:         return
