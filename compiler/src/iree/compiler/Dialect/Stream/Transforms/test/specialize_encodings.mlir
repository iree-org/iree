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

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "none"}>
#map = affine_map<(d0) -> (d0)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_
  stream.executable private @ex {
    stream.executable.export public @dispatch
  }
  util.func public @multi_device(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c4]) type(%element_type_f32) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@device_a>) %arg0 : !hal.buffer_view -> tensor<4xf32> in !stream.resource<external>{%c16}
    %1 = stream.timepoint.import on(#hal.device.affinity<@device_a>) %arg1 : (!hal.fence) => !stream.timepoint
    %2 = stream.timepoint.await %1 => %0 : !stream.resource<external>{%c16}
    %3 = stream.async.transfer %2 : !stream.resource<external>{%c16} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%c16}
    %4 = stream.async.dispatch on(#hal.device.affinity<@device_a>) @ex::@dispatch(%3[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
    %5 = stream.async.transfer %4 : !stream.resource<*>{%c16} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%c16}
    %6 = stream.async.dispatch on(#hal.device.affinity<@device_b>) @ex::@dispatch(%5[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
    %7 = stream.async.transfer %6 : !stream.resource<*>{%c16} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%c16}
    %result, %result_timepoint = stream.timepoint.barrier on(#hal.device.affinity<@device_a>) %7 : !stream.resource<*>{%c16} => !stream.timepoint
    stream.timepoint.chain_external on(#hal.device.affinity<@device_a>) %result_timepoint => (%arg2 : !hal.fence)
    %8 = stream.async.transfer %result : !stream.resource<*>{%c16} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<external>{%c16}
    %9 = stream.tensor.export on(#hal.device.affinity<@device_a>) %8 : tensor<4xf32> in !stream.resource<external>{%c16} -> !hal.buffer_view
    util.return %9 : !hal.buffer_view
  }
}

// CHECK:       #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK:       #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK-LABEL: util.func public @multi_device
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@dispatch
