// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-specialize-encodings)' --verify-diagnostics %s | FileCheck %s

//------------------------------------------------------------------------------
// IREE::CPU encoding layout specialization tests.
// These get serialized to the layout attributes.
//------------------------------------------------------------------------------

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_layout<>}>
#executable_target_x86_64 = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>, target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>

util.global private @device_a = #device_target_local_0_
util.global private @device_b = #device_target_local_1_
util.func public @tensor_sizeof(%d0: index, %d1: index) -> (index, index) {
  %size0 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
  util.return %size0, %size1 : index, index
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.layout<[#iree_cpu.vmvx_encoding_layout{{.+}}encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.layout<[#iree_cpu.cpu_encoding_layout{{.+}}encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK-LABEL: util.func public @tensor_sizeof
// CHECK:         %[[D0_RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING0]]>
// CHECK:         %[[D1_RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING1]]>
// CHECK:         return %[[D0_RES]], %[[D1_RES]]

// -----

//------------------------------------------------------------------------------
// #iree_gpu.gpu_encoding_layout specialization tests.
// These get serialized to the layout attributes.
//------------------------------------------------------------------------------

#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_encoding_layout<>,
    iree.gpu.target = #iree_gpu.target<arch = "gfx942",
                                       features = "",
                                       wgp = <compute = fp32,
                                              storage =  b32,
                                              subgroup =  none,
                                              dot =  none,
                                              mma = [<MFMA_F32_16x16x4_F32>],
                                              subgroup_size_choices = [64],
                                              max_workgroup_sizes = [1024, 1024, 1024],
                                              max_thread_count_per_workgroup = 1024,
                                              max_workgroup_memory_bytes = 65536,
                                              max_workgroup_counts = [2147483647, 2147483647, 2147483647],
                                              max_load_instruction_bits = 128,
                                              simds_per_wgp = 4,
                                              vgpr_space_bits = 16384>>
  }>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm_hsaco_fb]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map0, #map1, #map2]>

util.global private @device_a = #device_target_local_0_
util.func public @gpu_with_encoding_layout(%d0: index, %d1: index) -> index {
  %size0 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
  util.return %size0 : index
}
// CHECK:       #[[$ENCODING:.+]] = #iree_encoding.layout
// CHECK-SAME:    #iree_gpu.gpu_encoding_layout
// CHECK-SAME:    encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK-LABEL: util.func public @gpu_with_encoding_layout
// CHECK:         %[[RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING]]>
// CHECK:         return %[[RES]]

// -----

//------------------------------------------------------------------------------
// iree_gpu.gpu_pad_encoding specialization tests.
// These get serialized to iree_encoding.pad_encoding_layout attributes.
//------------------------------------------------------------------------------

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip",
  iree.encoding.resolver = #iree_gpu.gpu_pad_layout<>,
  iree.gpu.target = #iree_gpu.target<arch = "gfx942",
                                     features = "",
                                     wgp = <compute = fp32,
                                            storage =  b32,
                                            subgroup =  none,
                                            dot =  none,
                                            mma = [<MFMA_F32_16x16x4_F32>],
                                            subgroup_size_choices = [64],
                                            max_workgroup_sizes = [1024, 1024, 1024],
                                            max_thread_count_per_workgroup = 1024,
                                            max_workgroup_memory_bytes = 65536,
                                            max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm_hsaco_fb]> : !hal.device
#encodingA = #iree_encoding.pad_encoding_layout<[0, ?]>
#encodingB = #iree_encoding.pad_encoding_layout<[0, 64]>
#encodingC = #iree_encoding.pad_encoding_layout<[64, 64]>

util.global private @device_a = #device_target_local_0_
util.func public @with_pad_encoding_using_pad_attr(%arg0: index, %arg1: index) {
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x2048xf16, #encodingA>{%arg0} in !stream.resource<*>{%arg1}
  %1 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x16xf16, #encodingA>{%arg0} in !stream.resource<*>{%arg1}
  %2 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x2048xf16, #encodingB>{%arg0} in !stream.resource<*>{%arg1}
  %3 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x2048xf16, #encodingC>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG: #[[$NO_PAD:.+]] = #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 0]>]
// CHECK-DAG: #[[$PAD_DIM1_64:.+]] =  #iree_encoding.layout<[#iree_encoding.pad_encoding_layout<[0, 64]>]

// CHECK-LABEL: util.func public @with_pad_encoding_using_pad_attr(
// CHECK: stream.tensor.empty {{.*}} : tensor<?x2048xf16, #[[$PAD_DIM1_64]]>
// CHECK: stream.tensor.empty {{.*}} : tensor<?x16xf16, #[[$NO_PAD]]>
// CHECK: stream.tensor.empty {{.*}} : tensor<?x2048xf16, #iree_encoding.pad_encoding_layout<[0, 64]>>
// CHECK: stream.tensor.empty {{.*}} : tensor<?x2048xf16, #iree_encoding.pad_encoding_layout<[64, 64]>>

// -----

// Currently unsupported pad_encoding_layouts.

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip",
  iree.encoding.resolver = #iree_gpu.gpu_pad_layout<>,
  iree.gpu.target = #iree_gpu.target<arch = "gfx942",
                                     features = "",
                                     wgp = <compute = fp32,
                                            storage =  b32,
                                            subgroup =  none,
                                            dot =  none,
                                            mma = [<MFMA_F32_16x16x4_F32>],
                                            subgroup_size_choices = [64],
                                            max_workgroup_sizes = [1024, 1024, 1024],
                                            max_thread_count_per_workgroup = 1024,
                                            max_workgroup_memory_bytes = 65536,
                                            max_workgroup_counts = [2147483647, 2147483647, 2147483647]>>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm_hsaco_fb]> : !hal.device
#encodingA = #iree_encoding.pad_encoding_layout<[0, ?]>
#encodingD = #iree_encoding.pad_encoding_layout<[64, ?]>

// expected-error @+1 {{failed to add layouts to Stream::TensorPhaseOp with encodings}}
module {
util.global private @device_a = #device_target_local_0_
util.func public @error_with_pad_encoding_using_pad_attr(%arg0: index, %arg1: index) {
  // expected-error @+2 {{failed to resolve recognized layout}}
  // expected-error @+1 {{failed to convert unserialized encoding to serialized encoding}}
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<2048x?xf16, #encodingA>{%arg0} in !stream.resource<*>{%arg1}
  %1 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x2048xf16, #encodingD>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
}

// -----

// Creates an nop encoding if no `iree.gpu.target` is provided.

#executable_target_rocm_bytecode_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb", {abi = "hip", iree.encoding.resolver = #iree_gpu.gpu_pad_layout<> }>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.func public @create_pad_identity_encoding(%arg0: index, %arg1: index) {
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK: #[[$IDENTITY_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.pad_encoding_layout<[0, 0]>]>
// CHECK-LABEL: @create_pad_identity_encoding
// CHECK: stream.tensor.empty {{.*}} :  tensor<?x0xf32, #[[$IDENTITY_ENCODING]]>

// -----

//------------------------------------------------------------------------------
// Stream ops that have TensorPhaseOp trait. This test suite tests that the
// encoding is updated that carries resolved layouts.
//------------------------------------------------------------------------------

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.func public @ops_with_result_encoding_only(%arg0: index, %arg1: index, %scalar_f32 : f32) {
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
  %1 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  %2 = stream.tensor.splat on(#hal.device.affinity<@device_a>) %scalar_f32 : f32 -> tensor<?x1x10xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x0xf32>>]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x1x10xf32>>]>
// CHECK:       #[[TARGET:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE:.+]] = #[[TARGET]]
// CHECK-LABEL: util.func public @ops_with_result_encoding_only
// CHECK:         stream.tensor.empty on(#hal.device.affinity<@[[$DEVICE]]>) : tensor<?x0xf32, #[[$ENCODING0]]>
// CHECK:         stream.tensor.constant {{.+}} : tensor<1x5x64xf32>
// CHECK:         stream.tensor.splat on(#hal.device.affinity<@[[$DEVICE]]>) {{.+}} -> tensor<?x1x10xf32, #[[$ENCODING1]]>
// CHECK:         return

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>
util.global private @device_a = #device_target_local_0_
util.func public @tensor_fill_op(%arg0: f32, %arg1: !stream.resource<*>, %arg2: index, %arg3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = stream.tensor.fill on(#hal.device.affinity<@device_a>)
    %arg0, %arg1[%c0, %c0 for %c1, %c1] : f32
    -> tensor<?x4xf32, #encoding>{%arg2} in %arg1 as !stream.resource<*>{%arg3}
  util.return
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x4xf32>>]>
// CHECK:       #[[TARGET:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE:.+]] = #[[TARGET]]
// CHECK-LABEL: util.func public @tensor_fill_op
// CHECK:         stream.tensor.fill on(#hal.device.affinity<@[[$DEVICE]]>)
// CHECK-SAME:      f32 -> tensor<?x4xf32, #[[$ENCODING]]>

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>
util.global private @device_a = #device_target_local_0_
util.func public @tensor_encode_op(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x?xf32>>]>
// CHECK:       #[[TARGET:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE:.+]] = #[[TARGET]]
// CHECK-LABEL: util.func public @tensor_encode_op
// CHECK:         stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE]]>)
// CHECK-SAME:      -> tensor<?x?xf32, #[[$ENCODING]]>

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding0 = #iree_encoding.testing_encoding<>
#encoding1 = #iree_encoding.unknown_encoding
util.global private @device_a = #device_target_local_0_
util.func public @tensor_encode_op_change_encoding(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = stream.tensor.encode on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x?xf32, #encoding0>{%arg2, %arg3} in !stream.resource<*>{%arg1}
    -> tensor<?x?xf32, #encoding1>{%arg2, %arg3} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[$ENCODING0:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x?xf32>>]>
// CHECK-DAG:   #[[$ENCODING1:.+]] = #iree_encoding.unknown_encoding
// CHECK:       #[[TARGET:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE:.+]] = #[[TARGET]]
// CHECK-LABEL: util.func public @tensor_encode_op_change_encoding
// CHECK:         stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE]]>)
// CHECK-SAME:      : tensor<?x?xf32, #[[$ENCODING0]]>
// CHECK-SAME:      -> tensor<?x?xf32, #[[$ENCODING1]]>

// -----

// Checks that the stream.tensor.constant op with unserialized encoding is not
// supported.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

// expected-error @+1 {{failed to add layouts to Stream::TensorPhaseOp with encodings}}
module {
  util.global private @device_a = #device_target_local_0_
  util.func public @tensor_constant_op_with_unserialized_encoding(%arg0: index) {
    // expected-error @+1 {{failed to convert unserialized encoding to serialized encoding}}
    %0 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32, #encoding>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
    util.return
  }
}

// -----

#encoding = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
util.global private @device_a : !hal.device
util.func public @tensor_constant_op_with_serialized_encoding(%arg0: index) {
  %0 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32, #encoding>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  util.return
}
// CHECK:       #[[$SERIALIZED_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK-LABEL: util.func public @tensor_constant_op_with_serialized_encoding(
// CHECK:         stream.tensor.constant
// CHECK-SAME:      tensor<?x5x64xf32, #[[$SERIALIZED_ENCODING]]>

// -----

#encoding = #iree_encoding.unknown_encoding
util.global private @device_a : !hal.device
util.func public @tensor_constant_op_with_unknown_encoding(%arg0: index) {
  %0 = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<?x5x64xf32, #encoding>{%arg0} in !stream.resource<constant> = dense<0.000000e+00> : tensor<1x5x64xf32>
  util.return
}
// CHECK:       #[[$UNKNOWN_ENCODING:.+]] = #iree_encoding.unknown_encoding
// CHECK-LABEL: util.func public @tensor_constant_op_with_unknown_encoding(
// CHECK:         stream.tensor.constant
// CHECK-SAME:      tensor<?x5x64xf32, #[[$UNKNOWN_ENCODING]]>

// -----

// Checks that the stream.tensor.clone op with unserialized encoding is not
// supported.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

// expected-error @+1 {{failed to add layouts to Stream::TensorPhaseOp with encodings}}
module {
  util.global private @device_a = #device_target_local_0_
  util.func public @tensor_clone_op_with_unserialized_encoding(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    // expected-error @+1 {{failed to convert unserialized encoding to serialized encoding}}
    %0 = stream.tensor.clone on(#hal.device.affinity<@device_a>)
      %arg0 : tensor<?x4xf32, #encoding>{%arg1} in !stream.resource<*>{%arg2}
      -> tensor<?x4xf32, #encoding>{%arg1} in !stream.resource<*>{%arg2}
    util.return
  }
}

// -----

#unknown_encoding = #iree_encoding.unknown_encoding
util.global private @device_a : !hal.device
util.func public @tensor_clone_op_with_unknown_encodings(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %0 = stream.tensor.clone on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x4xf32, #unknown_encoding>{%arg1} in !stream.resource<*>{%arg2}
    -> tensor<?x4xf32, #unknown_encoding>{%arg1} in !stream.resource<*>{%arg2}
  util.return
}
// CHECK-DAG:   #[[$UNKNOWN_ENCODING:.+]] = #iree_encoding.unknown_encoding
// CHECK-LABEL: util.func public @tensor_clone_op_with_unknown_encodings(
// CHECK:         stream.tensor.clone
// CHECK-SAME:      tensor<?x4xf32, #[[$UNKNOWN_ENCODING]]>
// CHECK-SAME:      tensor<?x4xf32, #[[$UNKNOWN_ENCODING]]>

// -----

#serialized_encoding = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
util.global private @device_a : !hal.device
util.func public @tensor_clone_op_with_serialized_encodings(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %0 = stream.tensor.clone on(#hal.device.affinity<@device_a>)
    %arg0 : tensor<?x4xf32, #serialized_encoding>{%arg1} in !stream.resource<*>{%arg2}
    -> tensor<?x4xf32, #serialized_encoding>{%arg1} in !stream.resource<*>{%arg2}
  util.return
}
// CHECK-DAG:   #[[$SERIALIZED_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK-LABEL: util.func public @tensor_clone_op_with_serialized_encodings(
// CHECK:         stream.tensor.clone
// CHECK-SAME:      tensor<?x4xf32, #[[$SERIALIZED_ENCODING]]>
// CHECK-SAME:      tensor<?x4xf32, #[[$SERIALIZED_ENCODING]]>

// -----

// Checks that the stream.tensor.slice op with unserialized encoding is not
// supported.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

// expected-error @+1 {{failed to add layouts to Stream::TensorPhaseOp with encodings}}
module {
  util.global private @device_a = #device_target_local_0_
  util.func public @tensor_slice_op_with_encoding(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{failed to convert unserialized encoding to serialized encoding}}
    %1 = stream.tensor.slice on(#hal.device.affinity<@device_a>)
      %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32, #encoding>{%arg1} in !stream.resource<*>{%arg2}
      -> tensor<?x1xf32, #encoding>{%arg3} in !stream.resource<*>{%arg4}
    util.return
  }
}

// -----

#unknown_encoding = #iree_encoding.unknown_encoding
#serialized_encoding = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
util.global private @device_a : !hal.device
util.func public @tensor_slice_op_with_unknown_or_serialized_encodings(%arg0: !stream.resource<*>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = stream.tensor.slice on(#hal.device.affinity<@device_a>)
    %arg0[%c0, %c1 for %arg3, %c1] : tensor<?x4xf32, #unknown_encoding>{%arg1} in !stream.resource<*>{%arg2}
    -> tensor<?x1xf32, #serialized_encoding>{%arg3} in !stream.resource<*>{%arg4}
  util.return
}
// CHECK:       #[[$UNKNOWN_ENCODING:.+]] = #iree_encoding.unknown_encoding
// CHECK:       #[[$SERIALIZED_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK-LABEL: util.func public @tensor_slice_op_with_unknown_or_serialized_encodings(
// CHECK:         stream.tensor.slice
// CHECK-SAME:      tensor<?x4xf32, #[[$UNKNOWN_ENCODING]]>
// CHECK-SAME:      tensor<?x1xf32, #[[$SERIALIZED_ENCODING]]>

// -----

// Checks that the stream.tensor.update op with unserialized encoding is not
// supported.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

// expected-error @+1 {{failed to add layouts to Stream::TensorPhaseOp with encodings}}
module {
  util.global private @device_a = #device_target_local_0_
  util.func public @tensor_update_op_with_unserialized_encodings(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{failed to convert unserialized encoding to serialized encoding}}
    %0 = stream.tensor.update on(#hal.device.affinity<@device_a>)
      %arg0, %arg2[%c0, %c0] : tensor<2x2xf32, #encoding> in !stream.resource<*>{%arg1}
      -> tensor<?x4xf32, #encoding>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
    util.return
  }
}

// -----

#unknown_encoding = #iree_encoding.unknown_encoding
#serialized_encoding = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
util.global private @device_a : !hal.device
util.func public @tensor_update_op_with_unknown_or_serialized_encodings(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = stream.tensor.update on(#hal.device.affinity<@device_a>)
    %arg0, %arg2[%c0, %c0] : tensor<2x2xf32, #unknown_encoding> in !stream.resource<*>{%arg1}
    -> tensor<?x4xf32, #serialized_encoding>{%arg3} in %arg2 as !stream.resource<*>{%arg4}
  util.return
}
// CHECK:       #[[$UNKNOWN_ENCODING:.+]] = #iree_encoding.unknown_encoding
// CHECK:       #[[$SERIALIZED_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK-LABEL: util.func public @tensor_update_op_with_unknown_or_serialized_encodings(
// CHECK:         stream.tensor.update
// CHECK-SAME:      tensor<2x2xf32, #[[$UNKNOWN_ENCODING]]>
// CHECK-SAME:      tensor<?x4xf32, #[[$SERIALIZED_ENCODING]]>

// -----

// Creates an identity encoding if encoding attribute is not available in the
// target configuration.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.func public @drop_encoding(%arg0: index, %arg1: index, %scalar_f32 : f32) {
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[$IDENTITY_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.pad_encoding_layout<[0, 0]>]>
// CHECK-LABEL: util.func public @drop_encoding
// CHECK:         stream.tensor.empty {{.+}} : tensor<?x0xf32, #[[$IDENTITY_ENCODING]]>

// -----

// Creates an identity encoding if iree_encoding.identity_encoding is used.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", { iree.encoding.resolver = #iree_encoding.identity_encoding }>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.func public @ignore_encoding_by_identity_encoding(%arg0: index, %arg1: index, %scalar_f32 : f32) {
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[$IDENTITY_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.pad_encoding_layout<[0, 0]>]>
// CHECK-LABEL: util.func public @ignore_encoding_by_identity_encoding
// CHECK:         stream.tensor.empty {{.+}} : tensor<?x0xf32, #[[$IDENTITY_ENCODING]]>

// -----

// Do not update encodings if they are already serialized.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
util.global private @device_a = #device_target_local_0_
util.func public @keep_encoding_if_serialized(%arg0: index, %arg1: index, %scalar_f32 : f32) {
  %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
  util.return
}
// CHECK:       #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK-LABEL: util.func public @keep_encoding_if_serialized
// CHECK:         stream.tensor.empty {{.+}} : tensor<?x0xf32, #[[$ENCODING]]>

// -----

// Check that a failure is signaled if we are not able to resolve a recognized
// encoding.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", { iree.encoding.resolver = #iree_encoding.unsupported_encoding }>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>
// expected-error @+1 {{failed to add layouts to Stream::TensorPhaseOp with encodings}}
module {
  util.global private @device_a = #device_target_local_0_
  util.func public @fail_to_get_recognized_layout(%arg0: index, %arg1: index, %scalar_f32 : f32) {
    // expected-error @+2 {{failed to resolve recognized layout}}
    // expected-error @+1 {{failed to convert unserialized encoding to serialized encoding}}
    %0 = stream.tensor.empty on(#hal.device.affinity<@device_a>) : tensor<?x0xf32, #encoding>{%arg0} in !stream.resource<*>{%arg1}
    util.return
  }
}

// -----

//------------------------------------------------------------------------------
// This test suite tests the executable duplication logic.
//------------------------------------------------------------------------------

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
stream.executable private @executable {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: index) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16xf32, #encoding>>
      return
    }
  }
}

util.func public @tensor_dispatch_with_tied_operands(%arg0: !stream.resource<external>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg2}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @executable::@dispatch(%0, %arg3) : (tensor<4x?xf32, #encoding>{%arg2} in !stream.resource<*>{%arg1}, index) -> tensor<4x?xf32, #encoding>{%arg2} in %0{%arg1}
  util.return %1 : !stream.resource<*>
}
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<4x?xf32>>]>
// CHECK:       #[[TARGET:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE:.+]] = #[[TARGET]]
// CHECK-LABEL: util.func public @tensor_dispatch_with_tied_operands
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9]+]]
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE]]>)
// CHECK-SAME:      tensor<4x?xf32, #[[$ENCODING]]>{%[[ARG2]]}
// CHECK-SAME:      tensor<4x?xf32, #[[$ENCODING]]>{%[[ARG2]]}

// -----

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.global private @device_b = #device_target_local_1_
stream.executable private @ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
      %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
      return
    }
  }
}
util.func public @multi_device_with_same_executable_targets(%arg0: !stream.resource<external>, %arg1: index) {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg1} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg1}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@dispatch(%0) : (tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}) -> tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}
  %2 = stream.async.transfer %1 : !stream.resource<*>{%arg1} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%arg1}
  %3 = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex::@dispatch(%2) : (tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}) -> tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK-DAG:   #[[$ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<16xf32>>]>
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         stream.binding.subspan{{.+}}#[[$ENCODING]]
// CHECK:         stream.binding.subspan{{.+}}#[[$ENCODING]]
// CHECK-NOT:   stream.executable private
// CHECK-LABEL: util.func public @multi_device_with_same_executable_targets
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK-SAME:      #[[$ENCODING]]
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX0]]::@dispatch
// CHECK-SAME:      #[[$ENCODING]]

// -----

// Tests that launch the executable on device_a, pass the result to device_b and
// launch it on device_b. Thus, the incoming layout of second tensor dispatch op
// has device_a layout, and it produces device_b layout.

#executable_target_a = #hal.executable.target<"target_a", "abc", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#executable_target_b = #hal.executable.target<"target_b", "xyz", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<456>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_a]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_b]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.global private @device_b = #device_target_local_1_
stream.executable private @ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
      %c0 = arith.constant 0 : index
      %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
      %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32, #encoding>>
      return
    }
  }
}
util.func public @multi_device_with_different_executable_targets(%arg0: !stream.resource<external>, %arg1: index) {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg1} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg1}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@dispatch(%0) : (tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}) -> tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}
  %2 = stream.async.transfer %1 : !stream.resource<*>{%arg1} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%arg1}
  %3 = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex::@dispatch(%2) : (tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}) -> tensor<16xf32, #encoding> in !stream.resource<*>{%arg1}
  util.return
}
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK-DAG:   #[[$DEVICE_A_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<16xf32>>]>
// CHECK-DAG:   #[[$DEVICE_B_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<456, tensor<16xf32>>]>
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         stream.binding.subspan{{.+}}#[[$DEVICE_A_ENCODING]]
// CHECK:         stream.binding.subspan{{.+}}#[[$DEVICE_A_ENCODING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         stream.binding.subspan{{.+}}#[[$DEVICE_A_ENCODING]]
// CHECK:         stream.binding.subspan{{.+}}#[[$DEVICE_B_ENCODING]]
// CHECK-LABEL: util.func public @multi_device_with_different_executable_targets
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@dispatch
// CHECK-SAME:      #[[$DEVICE_A_ENCODING]]
// CHECK-SAME:      #[[$DEVICE_A_ENCODING]]
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@dispatch
// CHECK-SAME:      #[[$DEVICE_A_ENCODING]]
// CHECK-SAME:      #[[$DEVICE_B_ENCODING]]

// -----

// This tests the set_encoding, where the destination tensor type is encoded.
// The program has two external stream.resource. It imports transfer one to
// the device_a and the other to the device_b. Then it launches the set_encoding
// executable on both devices. We check that the executable is duplicated and
// the encodings on bindings are updated.

#executable_target_a = #hal.executable.target<"target_a", "abc", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#executable_target_b = #hal.executable.target<"target_b", "xyz", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<456>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_a]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_b]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.global private @device_b = #device_target_local_1_
stream.executable private @ex {
  stream.executable.export public @set_encoding
  builtin.module {
    func.func @set_encoding(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
      %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
      %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
      %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
      iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
      return
    }
  }
}
util.func public @multi_device_set_encoding(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>, %arg2: index, %N : index, %K : index) {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg2}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@set_encoding(%0, %N, %K) : (tensor<?x?xf32>{%N, %K} in !stream.resource<*>{%arg2}, index, index) -> (tensor<?x?xf32, #encoding>{%N, %K} in !stream.resource<*>{%arg2})
  %2 = stream.async.transfer %arg1 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%arg2}
  %3 = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex::@set_encoding(%2, %N, %K) : (tensor<?x?xf32>{%N, %K} in !stream.resource<*>{%arg2}, index, index) -> (tensor<?x?xf32, #encoding>{%N, %K} in !stream.resource<*>{%arg2})
  util.return
}

// CHECK-DAG:   #[[DEVICE_A_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x?xf32>>]>
// CHECK-DAG:   #[[DEVICE_B_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<456, tensor<?x?xf32>>]>
// CHECK-DAG:   #[[ORIG_ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         func.func @set_encoding(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #[[DEVICE_A_ENCODING]]>
// CHECK:           %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK:           %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32> -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[SET_ENCODING]], %[[DEST_BINDING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         func.func @set_encoding(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #[[DEVICE_B_ENCODING]]>
// CHECK:           %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK:           %[[SET_ENCODING:.+]] = iree_encoding.set_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32> -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[SET_ENCODING]], %[[DEST_BINDING]]
// CHECK-LABEL: util.func public @multi_device_set_encoding
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@set_encoding
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@set_encoding

// -----

// This test is simliar to the set_encoding test, but with unset_encoding ops.

#executable_target_a = #hal.executable.target<"target_a", "abc", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#executable_target_b = #hal.executable.target<"target_b", "xyz", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<456>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_a]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_b]> : !hal.device
#encoding = #iree_encoding.testing_encoding<>

util.global private @device_a = #device_target_local_0_
util.global private @device_b = #device_target_local_1_
stream.executable private @ex {
  stream.executable.export public @unset_encoding
  builtin.module {
    func.func @unset_encoding(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = iree_tensor_ext.dispatch.workload.ordinal %arg1, 0 : index
      %1 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 1 : index
      %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%0, %1}
      %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%0, %1} -> tensor<?x?xf32, #encoding>
      %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%0, %1}
      iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
      return
    }
  }
}
util.func public @multi_device_unset_encoding(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>, %arg2: index, %M: index, %N: index) {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg2}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@unset_encoding(%0, %M, %N) : (tensor<?x?xf32, #encoding>{%M, %N} in !stream.resource<*>{%arg2}, index, index) -> (tensor<?x?xf32>{%M, %N} in !stream.resource<*>{%arg2})
  %2 = stream.async.transfer %arg1 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%arg2}
  %3 = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex::@unset_encoding(%2, %M, %N) : (tensor<?x?xf32, #encoding>{%M, %N} in !stream.resource<*>{%arg2}, index, index) -> (tensor<?x?xf32>{%M, %N} in !stream.resource<*>{%arg2})
  util.return
}
// CHECK-DAG:   #[[DEVICE_A_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<?x?xf32>>]>
// CHECK-DAG:   #[[DEVICE_B_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<456, tensor<?x?xf32>>]>
// CHECK-DAG:   #[[ORIG_ENCODING:.+]] = #iree_encoding.testing_encoding<>
// CHECK-DAG:   #[[DEVICE_LOCAL_0:.+]] = #hal.device.target
// CHECK-DAG:   #[[DEVICE_LOCAL_1:.+]] = #hal.device.target
// CHECK:       util.global private @[[$DEVICE_A:.+]] = #[[DEVICE_LOCAL_0]]
// CHECK:       util.global private @[[$DEVICE_B:.+]] = #[[DEVICE_LOCAL_1]]
// CHECK:       stream.executable private @[[$EX0:.+]] {
// CHECK:         func.func @unset_encoding(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_ENCODING]]>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>
// CHECK:           %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_A_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32, #[[ORIG_ENCODING]]> -> tensor<?x?xf32>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[UNSET_ENCODING]], %[[DEST_BINDING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         func.func @unset_encoding(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK:           %[[SRC_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_ENCODING]]>>
// CHECK:           %[[DEST_BINDING:.+]] = stream.binding.subspan %[[ARG3]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32>>
// CHECK:           %[[SRC:.+]] = iree_tensor_ext.dispatch.tensor.load %[[SRC_BINDING]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[DEVICE_B_ENCODING]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_ENCODING]]>
// CHECK:           %[[UNSET_ENCODING:.+]] = iree_encoding.unset_encoding %[[SRC]]
// CHECK-SAME:         tensor<?x?xf32, #[[ORIG_ENCODING]]> -> tensor<?x?xf32>
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[UNSET_ENCODING]], %[[DEST_BINDING]]
// CHECK-LABEL: util.func public @multi_device_unset_encoding
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@unset_encoding
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@unset_encoding

// -----

// This tests the computation ops on tensor encodings, where all the tensor
// types are encoded. The computation body is fill + matmul.

#executable_target_a = #hal.executable.target<"target_a", "abc", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#executable_target_b = #hal.executable.target<"target_b", "xyz", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<456>}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_a]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_b]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>

util.global private @device_a = #device_target_local_0_
util.global private @device_b = #device_target_local_1_
stream.executable private @ex {
  stream.executable.export public @gemm
  builtin.module {
    func.func @gemm(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !stream.binding) {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.000000e+00 : f32
      %0 = iree_tensor_ext.dispatch.workload.ordinal %arg2, 0 : index
      %1 = iree_tensor_ext.dispatch.workload.ordinal %arg3, 1 : index
      %2 = iree_tensor_ext.dispatch.workload.ordinal %arg4, 2 : index
      %3 = iree_tensor_ext.dispatch.workload.ordinal %arg5, 3 : index
      %4 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0}
      %5 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3}
      %6 = stream.binding.subspan %arg6[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
      %7 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %0], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0} -> tensor<?x?xf32, #encoding>
      %8 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, %3], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3} -> tensor<?x?xf32, #encoding1>
      %9 = tensor.empty(%2, %3) : tensor<?x?xf32, #encoding2>
      %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
      %11 = linalg.matmul ins(%7, %8 : tensor<?x?xf32, #encoding>, tensor<?x?xf32, #encoding1>) outs(%10 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
      iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #encoding2> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
      return
    }
  }
}
util.func public @multi_device_gemm(%arg0: !stream.resource<external>, %arg1: !stream.resource<external>, %arg2: !stream.resource<external>, %arg3: !stream.resource<external>, %M: index, %N: index, %K: index) {
  %MK = arith.muli %M, %K : index
  %NK = arith.muli %N, %K : index
  %MN = arith.muli %M, %N : index
  %LHS_A = stream.async.transfer %arg0 : !stream.resource<external>{%MK} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%MK}
  %RHS_A = stream.async.transfer %arg1 : !stream.resource<external>{%NK} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%NK}
  %RES_A = stream.tensor.dispatch on(#hal.device.affinity<@device_a>)
    @ex::@gemm(%LHS_A, %RHS_A, %K, %K, %M, %N)
    : (tensor<?x?xf32, #encoding>{%M, %K} in !stream.resource<*>{%MK}, tensor<?x?xf32, #encoding1>{%N, %K} in !stream.resource<*>{%NK}, index, index, index, index)
    -> (tensor<?x?xf32, #encoding2>{%M, %N} in !stream.resource<*>{%MN})
  %barrier_0 = util.optimization_barrier %RES_A : !stream.resource<*>
  %LHS_B = stream.async.transfer %arg2 : !stream.resource<external>{%MK} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%MK}
  %RHS_B = stream.async.transfer %arg3 : !stream.resource<external>{%NK} from(#hal.device.affinity<@device_b>) -> to(#hal.device.affinity<@device_b>) !stream.resource<*>{%NK}
  %RES_B = stream.tensor.dispatch on(#hal.device.affinity<@device_b>)
    @ex::@gemm(%LHS_B, %RHS_B, %K, %K, %M, %N)
    : (tensor<?x?xf32, #encoding>{%M, %K} in !stream.resource<*>{%MK}, tensor<?x?xf32, #encoding1>{%N, %K} in !stream.resource<*>{%NK}, index, index, index, index)
    -> (tensor<?x?xf32, #encoding2>{%M, %N} in !stream.resource<*>{%MN})
  %barrier_1 = util.optimization_barrier %RES_B : !stream.resource<*>
  util.return
}

// CHECK-DAG:   #[[ENCODING_123:.+]] = #iree_encoding.layout<[#iree_encoding.specialized_encoding<123, tensor<?x?xf32>>]
// CHECK-DAG:   #[[ENCODING_456:.+]] = #iree_encoding.layout<[#iree_encoding.specialized_encoding<456, tensor<?x?xf32>>]
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
// CHECK:         func.func @gemm(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG4:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG5:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG6:[a-zA-Z0-9]+]]
// CHECK:           %[[LHS_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_123]]>>
// CHECK:           %[[RHS_BINDING:.+]] = stream.binding.subspan %[[ARG1]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_123]]>>
// CHECK:           %[[OUT_BINDING:.+]] = stream.binding.subspan %[[ARG6]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #[[ENCODING_123]]>>
// CHECK:           %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_123]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_LHS_ENCODING]]>
// CHECK:           %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_123]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_RHS_ENCODING]]>
// CHECK:           %[[INIT:.+]] = tensor.empty({{.+}}) : tensor<?x?xf32, #[[ORIG_OUT_ENCODING]]>
// CHECK:           %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:           %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:        outs(%[[FILL]]
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[MATMUL]], %[[OUT_BINDING]]
// CHECK:       stream.executable private @[[$EX1:.+]] {
// CHECK:         func.func @gemm(
// CHECK-SAME:        %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG3:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG4:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG5:[a-zA-Z0-9]+]]
// CHECK-SAME:        %[[ARG6:[a-zA-Z0-9]+]]
// CHECK:           %[[LHS_BINDING:.+]] = stream.binding.subspan %[[ARG0]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_456]]>>
// CHECK:           %[[RHS_BINDING:.+]] = stream.binding.subspan %[[ARG1]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_456]]>>
// CHECK:           %[[OUT_BINDING:.+]] = stream.binding.subspan %[[ARG6]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x?xf32, #[[ENCODING_456]]>>
// CHECK:           %[[LHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[LHS_BINDING]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_456]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_LHS_ENCODING]]>
// CHECK:           %[[RHS:.+]] = iree_tensor_ext.dispatch.tensor.load %[[RHS_BINDING]]
// CHECK-SAME:        !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x?xf32, #[[ENCODING_456]]>>
// CHECK-SAME:        -> tensor<?x?xf32, #[[ORIG_RHS_ENCODING]]>
// CHECK:           %[[INIT:.+]] = tensor.empty({{.+}}) : tensor<?x?xf32, #[[ORIG_OUT_ENCODING]]>
// CHECK:           %[[FILL:.+]] = linalg.fill ins({{.+}}) outs(%[[INIT]]
// CHECK:           %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME:        ins(%[[LHS]], %[[RHS]]
// CHECK-SAME:        outs(%[[FILL]]
// CHECK:           iree_tensor_ext.dispatch.tensor.store %[[MATMUL]], %[[OUT_BINDING]]
// CHECK-LABEL: util.func public @multi_device_gemm
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX0]]::@gemm
// CHECK:         stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX1]]::@gemm

// -----

// A test for unknown encodings and already serialized encodings. It does
// nothing if the encoding is not recognized. It updates the subspan binding, if
// the encoding is already serialized.

#unknown_encoding = #iree_encoding.unknown_encoding
#serialized_encoding = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
util.global private @device_a : !hal.device
stream.executable private @executable {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: index, %arg2: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xf32>>{%arg1}
      %1 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32>>{%arg1}
      return
    }
  }
}
util.func public @tensor_dispatch_with_unknown_and_serialized_encodings(%arg0: !stream.resource<external>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg2}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @executable::@dispatch(%0, %arg3) : (tensor<4x?xf32, #unknown_encoding>{%arg2} in !stream.resource<*>{%arg1}, index) -> tensor<4x?xf32, #serialized_encoding>{%arg2} in !stream.resource<*>{%arg1}
  util.return %1 : !stream.resource<*>
}
// CHECK-DAG:   #[[$UNKNOWN_ENCODING:.+]] = #iree_encoding.unknown_encoding
// CHECK-DAG:   #[[$SERIALIZED_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123>]>
// CHECK:       stream.executable
// CHECK:         func.func
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:       stream.binding.subspan %[[ARG0]]{{.+}} !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xf32>>
// CHECK-DAG:       stream.binding.subspan %[[ARG2]]{{.+}} !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32, #[[$SERIALIZED_ENCODING]]>>
// CHECK-LABEL: util.func public @tensor_dispatch_with_unknown_and_serialized_encodings(
// CHECK:         stream.tensor.dispatch
// CHECK:           tensor<4x?xf32, #[[$UNKNOWN_ENCODING]]>
// CHECK:           tensor<4x?xf32, #[[$SERIALIZED_ENCODING]]>

// -----

// Test that the unserialized encoding is serialized, and the unknown encoding
// is the same.

#unknown_encoding = #iree_encoding.unknown_encoding
#encoding = #iree_encoding.testing_encoding<>
#executable_target_a = #hal.executable.target<"target_a", "abc", {iree.encoding.resolver = #iree_encoding.unspecialized_encoding<123>}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_a]> : !hal.device
util.global private @device_a = #device_target_local_0_
stream.executable private @executable {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: index, %arg2: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xf32>>{%arg1}
      %1 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32>>{%arg1}
      return
    }
  }
}
util.func public @tensor_dispatch_with_unknown_and_unserialized_encodings(%arg0: !stream.resource<external>, %arg1: index, %arg2: index, %arg3: index) -> !stream.resource<*> {
  %0 = stream.async.transfer %arg0 : !stream.resource<external>{%arg2} from(#hal.device.affinity<@device_a>) -> to(#hal.device.affinity<@device_a>) !stream.resource<*>{%arg2}
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @executable::@dispatch(%0, %arg3) : (tensor<4x?xf32, #unknown_encoding>{%arg2} in !stream.resource<*>{%arg1}, index) -> tensor<4x?xf32, #encoding>{%arg2} in !stream.resource<*>{%arg1}
  util.return %1 : !stream.resource<*>
}
// CHECK-DAG:   #[[$UNKNOWN_ENCODING:.+]] = #iree_encoding.unknown_encoding
// CHECK-DAG:   #[[$SERIALIZED_ENCODING:.+]] = #iree_encoding.testing_encoding<[#iree_encoding.specialized_encoding<123, tensor<4x?xf32>>]>
// CHECK:       stream.executable
// CHECK:         func.func
// CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG1:[a-zA-Z0-9]+]]
// CHECK-SAME:      %[[ARG2:[a-zA-Z0-9]+]]
// CHECK-DAG:       stream.binding.subspan %[[ARG0]]{{.+}} !iree_tensor_ext.dispatch.tensor<readonly:tensor<4x?xf32>>
// CHECK-DAG:       stream.binding.subspan %[[ARG2]]{{.+}} !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4x?xf32, #[[$SERIALIZED_ENCODING]]>>
// CHECK-LABEL: util.func public @tensor_dispatch_with_unknown_and_unserialized_encodings(
// CHECK:         stream.tensor.dispatch
// CHECK:           tensor<4x?xf32, #[[$UNKNOWN_ENCODING]]>
// CHECK:           tensor<4x?xf32, #[[$SERIALIZED_ENCODING]]>

// -----

//------------------------------------------------------------------------------
// Negative tests. The pass should do nothing for the cases.
//------------------------------------------------------------------------------

// It does not fail because there are no encodings on stream.tensor.dispatch
// ops.

hal.executable.source public @executable {
  hal.executable.export public @dispatch ordinal(0) layout(#hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>
  ]>)
}
util.func public @dispatch_hal_executable(%arg0: !stream.resource<*>, %arg1: index, %arg2: index) -> !stream.resource<*> {
  %0 = stream.tensor.dispatch @executable::@dispatch(%arg0) : (tensor<4x?xf32>{%arg2} in !stream.resource<*>{%arg1}) -> tensor<4x?xf32>{%arg2} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-LABEL: util.func public @dispatch_hal_executable(

// -----

// It does not fail because the executable does not match the requirements.

#encoding = #iree_encoding.unknown_encoding
util.global private @device : !hal.device
hal.executable.source public @executable {
  hal.executable.export public @dispatch ordinal(0) layout(#hal.pipeline.layout<constants = 0, bindings = [
    #hal.pipeline.binding<storage_buffer>
  ]>)
}
util.func public @dispatch_hal_executable_with_encodings(%arg0: !stream.resource<*>, %arg1: index, %arg2: index) -> !stream.resource<*> {
  %0 = stream.tensor.dispatch on(#hal.device.affinity<@device>) @executable::@dispatch(%arg0) : (tensor<4x?xf32, #encoding>{%arg2} in !stream.resource<*>{%arg1}) -> tensor<4x?xf32, #encoding>{%arg2} in !stream.resource<*>{%arg1}
  util.return %0 : !stream.resource<*>
}
// CHECK-LABEL: util.func public @dispatch_hal_executable_with_encodings(
