// RUN: iree-opt --split-input-file --iree-stream-specialize-encodings --verify-diagnostics %s | FileCheck %s

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
    %1 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
    %2 = stream.tensor.splat on(#hal.device.affinity<@device_a>) %scalar_f32 : f32 -> tensor<?x1x10xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
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

// -----

// Checks that the stream.tensor.constant op with encoding is not supported.

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding = #iree_cpu.vmvx_encoding_layout<>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>
module {
  util.global private @device_a = #device_target_local_0_

  // expected-error @+1 {{failed on adding layouts to Stream::TensorPhaseOp with encodings}}
  util.func public @ops_with_result_encoding_only(%arg0: index) {
    %0 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32, #encoding>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
    util.return
  }
}

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "none"}>
#map = affine_map<(d0) -> (d0)>
#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_
  stream.executable private @ex {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
        %c0 = arith.constant 0 : index
        %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
        return
      }
    }
  }
  util.func public @multi_device_with_same_executable_targets(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view {
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
// CHECK-NOT:   stream.executable private
// CHECK-LABEL: util.func public @multi_device_with_same_executable_targets
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX0]]::@dispatch

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {ukernels = "none"}>
#executable_target_x86_64 = #hal.executable.target<"llvm-cpu", "xyz", {encoding = #iree_cpu.cpu_encoding_layout<>, target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64]> : !hal.device
#map = affine_map<(d0) -> (d0)>
#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_
  stream.executable private @ex {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
        %c0 = arith.constant 0 : index
        %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
        %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
        return
      }
    }
  }
  util.func public @multi_device_with_different_executable_targets(%arg0: !hal.buffer_view, %arg1: !hal.fence, %arg2: !hal.fence) -> !hal.buffer_view {
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
// CHECK-LABEL: util.func public @multi_device_with_different_executable_targets
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@dispatch

// -----

// This tests the set_encoding, where the destination tensor type is encoded.
// The program has two external stream.resource. It imports transfer one to
// the device_a and the other to the device_b. Then it launches the set_encoding
// executable on both devices. We check that the executable is duplicated and
// the encodings on bindings are updated.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding = #iree_cpu.vmvx_encoding_layout<>}>
#executable_target_x86_64_ = #hal.executable.target<"llvm-cpu", "x86_64", {cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>, target_triple = "x86_64-xyz-xyz"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64_]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_
  stream.executable private @ex {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
        return
      }
    }
  }
  util.func public @multi_device_set_encoding(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = stream.async.transfer %arg0 : !stream.resource<external>{%c16} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%c16}
    %1 = stream.async.dispatch on(#hal.device.affinity<@device_a>) @ex::@dispatch(%0[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
    %2 = util.optimization_barrier %1 : !stream.resource<*>
    %3 = stream.async.transfer %arg1 : !stream.resource<external>{%c16} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%c16}
    %4 = stream.async.dispatch on(#hal.device.affinity<@device_b>) @ex::@dispatch(%3[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
    %5 = util.optimization_barrier %4 : !stream.resource<*>
    util.return
  }
}
// CHECK-DAG:   #[[DEVICE_A_ENCODING:.+]] = #iree_encoding.encoding{{.+}} layouts = [#iree_cpu.vmvx_encoding_layout
// CHECK-DAG:   #[[DEVICE_B_ENCODING:.+]] = #iree_encoding.encoding{{.+}} layouts = [#iree_cpu.cpu_encoding_layout
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
//
// Explicitly capture the last `>` symbol because it makes sure that the
// `layouts` is not attached in the ORIG_ENCODING.
//
// CHECK-DAG:   #[[ORIG_ENCODING:.+]] = #iree_encoding.encoding<{{.+}} user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         func.func @dispatch(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #[[DEVICE_A_ENCODING]]>
// CHECK:           %[[SRC:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK:           %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32> -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           flow.dispatch.tensor.store %[[SET_ENCODING]], %[[DEST_BINDING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         func.func @dispatch(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #[[DEVICE_B_ENCODING]]>
// CHECK:           %[[SRC:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK:           %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32> -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           flow.dispatch.tensor.store %[[SET_ENCODING]], %[[DEST_BINDING]]
// CHECK-LABEL: util.func public @multi_device_set_encoding
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@dispatch

// -----

// This test is simliar to the set_encoding test, but with unset_encoding ops.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding = #iree_cpu.vmvx_encoding_layout<>}>
#executable_target_x86_64_ = #hal.executable.target<"llvm-cpu", "x86_64", {cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>, target_triple = "x86_64-xyz-xyz"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64_]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_
  stream.executable private @ex {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%0, %1} -> tensor<?x?xf32, #encoding>
        %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%0, %1}
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
  util.func public @multi_device_unset_encoding(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>) {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0 = stream.async.transfer %arg0 : !stream.resource<external>{%c16} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%c16}
    %1 = stream.async.dispatch on(#hal.device.affinity<@device_a>) @ex::@dispatch(%0[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
    %2 = util.optimization_barrier %1 : !stream.resource<*>
    %3 = stream.async.transfer %arg1 : !stream.resource<external>{%c16} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%c16}
    %4 = stream.async.dispatch on(#hal.device.affinity<@device_b>) @ex::@dispatch(%3[%c0 to %c16 for %c16]) : (!stream.resource<*>{%c16}) -> !stream.resource<*>{%c16}
    %5 = util.optimization_barrier %4 : !stream.resource<*>
    util.return
  }
}
// CHECK-DAG:   #[[DEVICE_A_ENCODING:.+]] = #iree_encoding.encoding{{.+}} layouts = [#iree_cpu.vmvx_encoding_layout
// CHECK-DAG:   #[[DEVICE_B_ENCODING:.+]] = #iree_encoding.encoding{{.+}} layouts = [#iree_cpu.cpu_encoding_layout
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[ORIG_ENCODING:.+]] = #iree_encoding.encoding<{{.+}} user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         func.func @dispatch(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_ENCODING]]>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
// CHECK:           %[[SRC:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32, #[[ORIG_ENCODING]]> -> tensor<?x?xf32>
// CHECK:           flow.dispatch.tensor.store %[[UNSET_ENCODING]], %[[DEST_BINDING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         func.func @dispatch(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_ENCODING]]>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>
// CHECK:           %[[SRC:.+]] = flow.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32, #[[ORIG_ENCODING]]> -> tensor<?x?xf32>
// CHECK:           flow.dispatch.tensor.store %[[UNSET_ENCODING]], %[[DEST_BINDING]]
// CHECK-LABEL: util.func public @multi_device_unset_encoding
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@dispatch

// -----

// This tests the computation ops on tensor encodings, where all the tensor
// types are encoded. The computation body is fill + matmul.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {encoding = #iree_cpu.vmvx_encoding_layout<>}>
#executable_target_x86_64_ = #hal.executable.target<"llvm-cpu", "x86_64", {cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>, target_triple = "x86_64-xyz-xyz"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64_]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
module attributes {stream.affinity.default = #hal.device.affinity<@device_a>} {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_
  stream.executable private @ex {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !stream.binding) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
        %4 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0}
        %5 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3}
        %6 = stream.binding.subspan %arg6[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0} -> tensor<?x?xf32, #encoding>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3} -> tensor<?x?xf32, #encoding1>
        %9 = tensor.empty(%2, %3) : tensor<?x?xf32, #encoding2>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
        %11 = linalg.matmul ins(%7, %8 : tensor<?x?xf32, #encoding>, tensor<?x?xf32, #encoding1>) outs(%10 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #encoding2> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        return
      }
    }
  }
  util.func public @multi_device_gemm(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>, %arg2: !stream.resource<external>, %arg3: !stream.resource<external>) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %cst_M = arith.constant 1024 : index
    %cst_N = arith.constant 2048 : index
    %cst_K = arith.constant 512 : index
    %cst_MK = arith.muli %cst_M, %cst_K : index
    %cst_NK = arith.muli %cst_N, %cst_K : index
    %cst_MN = arith.muli %cst_M, %cst_N : index
    %M = util.optimization_barrier %cst_M : index
    %N = util.optimization_barrier %cst_N : index
    %K = util.optimization_barrier %cst_K : index
    %MK = util.optimization_barrier %cst_MK : index
    %NK = util.optimization_barrier %cst_NK : index
    %MN = util.optimization_barrier %cst_MN : index
    %LHS_A = stream.async.transfer %arg0 : !stream.resource<external>{%MK} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%MK}
    %RHS_A = stream.async.transfer %arg1 : !stream.resource<external>{%NK} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%NK}
    %RES_A = stream.async.dispatch on(#hal.device.affinity<@device_a>)
      @ex::@dispatch[%K, %K, %M, %N](%LHS_A[%c0 to %MK for %MK],
                                     %RHS_A[%c0 to %NK for %NK],
                                     %K, %K, %M, %N)
      : (!stream.resource<*>{%MK}, !stream.resource<*>{%NK}, index, index, index, index)
      -> !stream.resource<*>{%MN}
    %barrier_0 = util.optimization_barrier %RES_A : !stream.resource<*>
    %LHS_B = stream.async.transfer %arg2 : !stream.resource<external>{%MK} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%MK}
    %RHS_B = stream.async.transfer %arg3 : !stream.resource<external>{%NK} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%NK}
    %RES_B = stream.async.dispatch on(#hal.device.affinity<@device_b>)
      @ex::@dispatch[%K, %K, %M, %N](%LHS_B[%c0 to %MK for %MK],
                                     %RHS_B[%c0 to %NK for %NK],
                                     %K, %K, %M, %N)
      : (!stream.resource<*>{%MK}, !stream.resource<*>{%NK}, index, index, index, index)
      -> !stream.resource<*>{%MN}
    %barrier_1 = util.optimization_barrier %RES_B : !stream.resource<*>
    util.return
  }
}
// CHECK-DAG:   #[[DEVICE_A_LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0{{.+}} layouts = [#iree_cpu.vmvx_encoding_layout
// CHECK-DAG:   #[[DEVICE_A_RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1{{.+}} layouts = [#iree_cpu.vmvx_encoding_layout
// CHECK-DAG:   #[[DEVICE_A_OUT_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2{{.+}} layouts = [#iree_cpu.vmvx_encoding_layout
// CHECK-DAG:   #[[DEVICE_B_LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0{{.+}} layouts = [#iree_cpu.cpu_encoding_layout
// CHECK-DAG:   #[[DEVICE_B_RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1{{.+}} layouts = [#iree_cpu.cpu_encoding_layout
// CHECK-DAG:   #[[DEVICE_B_OUT_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2{{.+}} layouts = [#iree_cpu.cpu_encoding_layout
// CHECK-DAG:   #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:   #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:   #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[ORIG_LHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 0{{.+}} user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[ORIG_RHS_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 1{{.+}} user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[ORIG_OUT_ENCODING:.+]] = #iree_encoding.encoding<operand_index = 2{{.+}} user_indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]>
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         func.func @dispatch(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG4:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG5:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG6:[a-zA-Z0-9]+]]
// CHECK:           %[[LHS_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_LHS_ENCODING]]>>
// CHECK:           %[[RHS_BINDING:.+]] = stream.binding.subspan %[[ARG1]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_RHS_ENCODING]]>>
// CHECK:           %[[OUT_BINDING:.+]] = stream.binding.subspan %[[ARG6]]
// CHECK-SAME:        !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #[[DEVICE_A_OUT_ENCODING]]>>
// CHECK:           %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_LHS_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_LHS_ENCODING]]>
// CHECK:           %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_RHS_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_RHS_ENCODING]]>
// CHECK:           %[[INIT:.+]] = tensor.empty({{.+}}) : tensor<?x?xf32, #[[ORIG_OUT_ENCODING]]>
// CHECK:           %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:           %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:        outs(%[[FILL]]
// CHECK:           flow.dispatch.tensor.store %[[MATMUL]], %[[OUT_BINDING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         func.func @dispatch(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG4:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG5:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG6:[a-zA-Z0-9]+]]
// CHECK:           %[[LHS_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_LHS_ENCODING]]>>
// CHECK:           %[[RHS_BINDING:.+]] = stream.binding.subspan %[[ARG1]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_RHS_ENCODING]]>>
// CHECK:           %[[OUT_BINDING:.+]] = stream.binding.subspan %[[ARG6]]
// CHECK-SAME:        !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #[[DEVICE_B_OUT_ENCODING]]>>
// CHECK:           %[[LHS:.+]] = flow.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_LHS_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_LHS_ENCODING]]>
// CHECK:           %[[RHS:.+]] = flow.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:        !flow.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_RHS_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_RHS_ENCODING]]>
// CHECK:           %[[INIT:.+]] = tensor.empty({{.+}}) : tensor<?x?xf32, #[[ORIG_OUT_ENCODING]]>
// CHECK:           %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:           %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:        outs(%[[FILL]]
// CHECK:           flow.dispatch.tensor.store %[[MATMUL]], %[[OUT_BINDING]]
// CHECK-LABEL: util.func public @multi_device_gemm
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK:         stream.async.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@dispatch
