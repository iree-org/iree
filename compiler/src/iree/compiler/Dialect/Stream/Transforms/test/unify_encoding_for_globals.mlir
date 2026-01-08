// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-unify-encoding-for-globals)' %s | FileCheck %s

// Test: immutable source global (with initial value) with two encodings -
// should unify to identity encoding.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.identity_resolver}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>

// CHECK-LABEL: module @immutable_source_with_initial_value
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @immutable_source_with_initial_value {
  util.global private @device_a = #device_target_local
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v2 : !stream.resource<constant>

  // CHECK: util.initializer
  util.initializer {
    // CHECK: %[[SOURCE:.+]] = util.global.load @source
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
    %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_v2 : !stream.resource<constant>

    util.return
  }
}

// -----

// Checks that the identity encoding is generated if no resolver is specified.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<789>]>

// CHECK: util.global private @[[$DEVICE_A:.+]] =
util.global private @device_a = #device_target_local
util.global private @weight : !stream.resource<constant>
util.global private @weight_size : index
util.global private @encoded_v1 : !stream.resource<constant>
util.global private @encoded_v1_size : index
util.global private @encoded_v2 : !stream.resource<constant>
util.global private @encoded_v2_size : index

// CHECK: util.initializer
util.initializer {
  %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf32>
  %0 = stream.resource.size %cst : !stream.resource<constant>
  util.global.store %cst, @weight : !stream.resource<constant>
  util.global.store %0, @weight_size : index
  // CHECK: %[[SOURCE:.+]] = util.global.load @weight
  %source = util.global.load @weight : !stream.resource<constant>
  %source_size = util.global.load @weight_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.identity>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
  %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size1}
  util.global.store %enc1, @encoded_v1 : !stream.resource<constant>
  util.global.store %size1, @encoded_v1_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.identity>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
  %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
  %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size2}
  util.global.store %enc2, @encoded_v2 : !stream.resource<constant>
  util.global.store %size2, @encoded_v2_size : index

  util.return
}

// -----

// Test: multiple devices with different resolvers encoding the same source global.
// Since different resolvers produce different encodings and they share the same source,
// there's no common encoding - should fall back to identity encoding.

#executable_target_0 = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<123>}>
#executable_target_1 = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<456>}>
#device_target_local_0 = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_0]> : !hal.device
#device_target_local_1 = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_1]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<111>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<222>]>

// CHECK: util.global private @[[$DEVICE_A:.+]] =
// CHECK: util.global private @[[$DEVICE_B:.+]] =
util.global private @device_a = #device_target_local_0
util.global private @device_b = #device_target_local_1
// Single source global shared by both devices.
util.global private @weight : !stream.resource<constant>
util.global private @weight_size : index
util.global private @encoded_a_v1 : !stream.resource<constant>
util.global private @encoded_a_v2 : !stream.resource<constant>
util.global private @encoded_b_v1 : !stream.resource<constant>
util.global private @encoded_b_v2 : !stream.resource<constant>

// CHECK: util.initializer
util.initializer {
  %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf32>
  %0 = stream.resource.size %cst : !stream.resource<constant>
  util.global.store %cst, @weight : !stream.resource<constant>
  util.global.store %0, @weight_size : index

  // CHECK: %[[SOURCE:.+]] = util.global.load @weight
  %source = util.global.load @weight : !stream.resource<constant>
  %source_size = util.global.load @weight_size : index

  // Device A encodes the shared source - should get identity encoding.
  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.identity>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
  %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size1}
  util.global.store %enc1, @encoded_a_v1 : !stream.resource<constant>

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.identity>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
  %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
  %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size2}
  util.global.store %enc2, @encoded_a_v2 : !stream.resource<constant>

  // Device B encodes the same shared source - should also get identity encoding.
  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_B]]>) tensor<4096x4096xf32, #iree_encoding.identity>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_B]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
  %size3 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<4096x4096xf32, #encoding1> : index
  %enc3 = stream.tensor.encode on(#hal.device.affinity<@device_b>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size3}
  util.global.store %enc3, @encoded_b_v1 : !stream.resource<constant>

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_B]]>) tensor<4096x4096xf32, #iree_encoding.identity>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_B]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
  %size4 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<4096x4096xf32, #encoding2> : index
  %enc4 = stream.tensor.encode on(#hal.device.affinity<@device_b>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size4}
  util.global.store %enc4, @encoded_b_v2 : !stream.resource<constant>

  util.return
}

// -----

// Test: multiple devices with different resolvers - each device should use its own resolver.
// Also tests executable duplication when the same executable is used by dispatch sites
// targeting different devices with different encodings.

#executable_target_0 = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<123>}>
#executable_target_1 = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<456>}>
#device_target_local_0 = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_0]> : !hal.device
#device_target_local_1 = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_1]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<111>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<222>]>

// CHECK: util.global private @[[$DEVICE_A:.+]] =
// CHECK: util.global private @[[$DEVICE_B:.+]] =
util.global private @device_a = #device_target_local_0
util.global private @device_b = #device_target_local_1
// Two separate source globals - one for each device.
util.global private @weight_a : !stream.resource<constant>
util.global private @weight_a_size : index
util.global private @weight_b : !stream.resource<constant>
util.global private @weight_b_size : index
util.global private @encoded_a_v1 : !stream.resource<constant>
util.global private @encoded_a_v1_size : index
util.global private @encoded_a_v2 : !stream.resource<constant>
util.global private @encoded_a_v2_size : index
util.global private @encoded_b_v1 : !stream.resource<constant>
util.global private @encoded_b_v1_size : index
util.global private @encoded_b_v2 : !stream.resource<constant>
util.global private @encoded_b_v2_size : index

// CHECK: util.initializer
util.initializer {
  // Initialize weight_a for device_a.
  %cst_a = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight_a"> : tensor<4096x4096xf32>
  %size_a = stream.resource.size %cst_a : !stream.resource<constant>
  util.global.store %cst_a, @weight_a : !stream.resource<constant>
  util.global.store %size_a, @weight_a_size : index

  // Initialize weight_b for device_b.
  %cst_b = stream.tensor.constant on(#hal.device.affinity<@device_b>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight_b"> : tensor<4096x4096xf32>
  %size_b = stream.resource.size %cst_b : !stream.resource<constant>
  util.global.store %cst_b, @weight_b : !stream.resource<constant>
  util.global.store %size_b, @weight_b_size : index

  // CHECK: %[[SOURCE_A:.+]] = util.global.load @weight_a
  %source_a = util.global.load @weight_a : !stream.resource<constant>
  %source_a_size = util.global.load @weight_a_size : index

  // CHECK: %[[SOURCE_B:.+]] = util.global.load @weight_b
  %source_b = util.global.load @weight_b : !stream.resource<constant>
  %source_b_size = util.global.load @weight_b_size : index

  // Device A encodes weight_a with specialization_resolver<123>.
  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE_A]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
  %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source_a : tensor<4096x4096xf32> in !stream.resource<constant>{%source_a_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size1}
  util.global.store %enc1, @encoded_a_v1 : !stream.resource<constant>
  util.global.store %size1, @encoded_a_v1_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE_A]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
  %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source_a : tensor<4096x4096xf32> in !stream.resource<constant>{%source_a_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size2}
  util.global.store %enc2, @encoded_a_v2 : !stream.resource<constant>
  util.global.store %size2, @encoded_a_v2_size : index

  // Device B encodes weight_b with specialization_resolver<456>.
  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_B]]>) tensor<4096x4096xf32, #iree_encoding.specialized<456>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_B]]>) %[[SOURCE_B]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<456>>
  %size3 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<4096x4096xf32, #encoding1> : index
  %enc3 = stream.tensor.encode on(#hal.device.affinity<@device_b>) %source_b : tensor<4096x4096xf32> in !stream.resource<constant>{%source_b_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size3}
  util.global.store %enc3, @encoded_b_v1 : !stream.resource<constant>
  util.global.store %size3, @encoded_b_v1_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_B]]>) tensor<4096x4096xf32, #iree_encoding.specialized<456>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_B]]>) %[[SOURCE_B]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<456>>
  %size4 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<4096x4096xf32, #encoding2> : index
  %enc4 = stream.tensor.encode on(#hal.device.affinity<@device_b>) %source_b : tensor<4096x4096xf32> in !stream.resource<constant>{%source_b_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size4}
  util.global.store %enc4, @encoded_b_v2 : !stream.resource<constant>
  util.global.store %size4, @encoded_b_v2_size : index

  util.return
}

// Original executable - updated for device_a (sorted first alphabetically).
// CHECK: stream.executable private @[[$EX:.+]] {
// CHECK:   stream.binding.subspan %arg0{{.*}} -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #iree_encoding.specialized<123>>>
// CHECK:   stream.binding.subspan %arg1{{.*}} -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #iree_encoding.specialized<123>>>
stream.executable private @ex {
  stream.executable.export public @dispatch
  builtin.module {
    func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) {
      %c0 = arith.constant 0 : index
      %1 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding1>>
      %2 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding2>>
      %3 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xf32>>
      return
    }
  }
}

// Duplicated executable for device_b (sorted second alphabetically).
// CHECK: stream.executable private @[[$EX_DUP:.+]] {
// CHECK:   stream.binding.subspan %arg0{{.*}} -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #iree_encoding.specialized<456>>>
// CHECK:   stream.binding.subspan %arg1{{.*}} -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #iree_encoding.specialized<456>>>

// CHECK-LABEL: util.func public @multi_device_with_executable_duplication
util.func public @multi_device_with_executable_duplication(%arg0: index) {
  %encoded_a_v1 = util.global.load immutable @encoded_a_v1 : !stream.resource<constant>
  %encoded_a_v1_size = util.global.load immutable @encoded_a_v1_size : index
  %operand_a_0 = stream.async.clone on(#hal.device.affinity<@device_a>)
    %encoded_a_v1 : !stream.resource<constant>{%encoded_a_v1_size}
    -> !stream.resource<*>{%encoded_a_v1_size}
  %encoded_a_v2 = util.global.load immutable @encoded_a_v2 : !stream.resource<constant>
  %encoded_a_v2_size = util.global.load immutable @encoded_a_v2_size : index
  %operand_a_1 = stream.async.clone on(#hal.device.affinity<@device_a>)
    %encoded_a_v2 : !stream.resource<constant>{%encoded_a_v2_size}
    -> !stream.resource<*>{%encoded_a_v2_size}
  // CHECK: stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @[[$EX]]::@dispatch
  // CHECK-SAME: tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK-SAME: tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %dispatch_a = stream.tensor.dispatch on(#hal.device.affinity<@device_a>)
    @ex::@dispatch(%operand_a_0, %operand_a_1) : (tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%encoded_a_v1_size},
                                                  tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%encoded_a_v2_size}
    ) -> tensor<4096x4096xf32> in !stream.resource<*>{%arg0}

  %encoded_b_v1 = util.global.load immutable @encoded_b_v1 : !stream.resource<constant>
  %encoded_b_v1_size = util.global.load immutable @encoded_b_v1_size : index
  %operand_b_0 = stream.async.clone on(#hal.device.affinity<@device_b>)
    %encoded_b_v1 : !stream.resource<constant>{%encoded_b_v1_size}
    -> !stream.resource<*>{%encoded_b_v1_size}
  %encoded_b_v2 = util.global.load immutable @encoded_b_v2 : !stream.resource<constant>
  %encoded_b_v2_size = util.global.load immutable @encoded_b_v2_size : index
  %operand_b_1 = stream.async.clone on(#hal.device.affinity<@device_b>)
    %encoded_b_v2 : !stream.resource<constant>{%encoded_b_v2_size}
    -> !stream.resource<*>{%encoded_b_v2_size}
  // CHECK: stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_B]]>) @[[$EX_DUP]]::@dispatch
  // CHECK-SAME: tensor<4096x4096xf32, #iree_encoding.specialized<456>>
  // CHECK-SAME: tensor<4096x4096xf32, #iree_encoding.specialized<456>>
  %dispatch_b = stream.tensor.dispatch on(#hal.device.affinity<@device_b>)
    @ex::@dispatch(%operand_b_0, %operand_b_1) : (tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%encoded_b_v1_size},
                                                  tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%encoded_b_v2_size}
    ) -> tensor<4096x4096xf32> in !stream.resource<*>{%arg0}

  util.return
}

// -----

// Test: immutable source global (initialized from parameter in initializer) with
// two encodings - should unify to identity encoding.
// This test also verifies that stream.async.clone between load and encode is
// properly traced through (matching real-world patterns from input pipelines).

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<123>}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<789>]>

// CHECK-LABEL: module @immutable_source_initialized_from_parameter
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @immutable_source_initialized_from_parameter {
  util.global private @device_a = #device_target_local
  util.global private @weight : !stream.resource<constant>
  util.global private @weight_size : index
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v1_size : index
  util.global private @encoded_v2 : !stream.resource<constant>
  util.global private @encoded_v2_size : index

  // CHECK: util.initializer
  util.initializer {
    %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf32>
    %0 = stream.resource.size %cst : !stream.resource<constant>
    util.global.store %cst, @weight : !stream.resource<constant>
    util.global.store %0, @weight_size : index
    // CHECK: %[[SOURCE:.+]] = util.global.load @weight
    %source = util.global.load @weight : !stream.resource<constant>
    %source_size = util.global.load @weight_size : index

    // Clone before encode (common pattern in real pipelines).
    // CHECK: %[[CLONE1:.+]] = stream.async.clone on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]]
    %cloned1 = stream.async.clone on(#hal.device.affinity<@device_a>) %source : !stream.resource<constant>{%source_size} -> !stream.resource<*>{%source_size}

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[CLONE1]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %cloned1 : tensor<4096x4096xf32> in !stream.resource<*>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>
    util.global.store %size1, @encoded_v1_size : index

    // CHECK: %[[CLONE2:.+]] = stream.async.clone on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]]
    %cloned2 = stream.async.clone on(#hal.device.affinity<@device_a>) %source : !stream.resource<constant>{%source_size} -> !stream.resource<*>{%source_size}

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[CLONE2]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %cloned2 : tensor<4096x4096xf32> in !stream.resource<*>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_v2 : !stream.resource<constant>
    util.global.store %size2, @encoded_v2_size : index

    util.return
  }

  // Executable that uses encoding1.
  stream.executable private @executable_v1 {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding1>>
        return
      }
    }
  }

  // Executable that uses encoding2.
  stream.executable private @executable_v2 {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xf32>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding2>>
        return
      }
    }
  }

  // Executable that uses both encodings with index operands in between.
  stream.executable private @executable_both {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: index, %arg2: !stream.binding, %arg3: index, %arg4: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding1>>
        %1 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding2>>
        %2 = stream.binding.subspan %arg4[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
        return
      }
    }
  }

  // CHECK-LABEL: util.func public @use_encoded_v1
  util.func public @use_encoded_v1(%arg0: index) -> !stream.resource<*> {
    %encoded = util.global.load @encoded_v1 : !stream.resource<constant>
    %encoded_size = util.global.load @encoded_v1_size : index
    // CHECK:      stream.tensor.dispatch
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %result = stream.tensor.dispatch @executable_v1::@dispatch(%encoded)
      : (tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%encoded_size})
      -> tensor<16xf32> in !stream.resource<*>{%arg0}
    util.return %result : !stream.resource<*>
  }

  // CHECK-LABEL: util.func public @use_encoded_v2
  util.func public @use_encoded_v2(%arg0: index) -> !stream.resource<*> {
    %encoded = util.global.load @encoded_v2 : !stream.resource<constant>
    %encoded_size = util.global.load @encoded_v2_size : index
    // CHECK:      stream.tensor.dispatch
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %result = stream.tensor.dispatch @executable_v2::@dispatch(%encoded)
      : (tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%encoded_size})
      -> tensor<16xf32> in !stream.resource<*>{%arg0}
    util.return %result : !stream.resource<*>
  }

  // Function that uses both encoded globals in the same dispatch with index
  // operands in between. This tests that the operand index to encoding index
  // mapping is correct.
  // CHECK-LABEL: util.func public @use_both_encodings
  util.func public @use_both_encodings(%arg0: index) -> !stream.resource<*> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %encoded_v1 = util.global.load @encoded_v1 : !stream.resource<constant>
    %encoded_v1_size = util.global.load @encoded_v1_size : index
    %encoded_v2 = util.global.load @encoded_v2 : !stream.resource<constant>
    %encoded_v2_size = util.global.load @encoded_v2_size : index

    %cloned_v1 = stream.async.clone %encoded_v1 : !stream.resource<constant>{%encoded_v1_size} -> !stream.resource<*>{%encoded_v1_size}
    %cloned_v2 = stream.async.clone %encoded_v2 : !stream.resource<constant>{%encoded_v2_size} -> !stream.resource<*>{%encoded_v2_size}

    // CHECK:      stream.tensor.dispatch @executable_both::@dispatch
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %result = stream.tensor.dispatch @executable_both::@dispatch[%c16, %c32](%cloned_v1, %c0, %cloned_v2, %c1) : (tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%encoded_v1_size}, index, tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%encoded_v2_size}, index) -> tensor<16xf32> in !stream.resource<*>{%arg0}

    util.return %result : !stream.resource<*>
  }
}

// -----

// Test: cross-function tracking - load encoded global, pass to callee via
// util.call, and verify dispatch site encoding is updated in callee.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<123>}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<789>]>

// CHECK-LABEL: module @cross_function_tracking
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @cross_function_tracking {
  util.global private @device_a = #device_target_local
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v1_size : index
  util.global private @encoded_v2 : !stream.resource<constant>
  util.global private @encoded_v2_size : index

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>
    util.global.store %size1, @encoded_v1_size : index

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_v2 : !stream.resource<constant>
    util.global.store %size2, @encoded_v2_size : index

    util.return
  }

  stream.executable private @executable_both {
    stream.executable.export public @dispatch
    builtin.module {
      func.func @dispatch(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding1>>
        %1 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xf32, #encoding2>>
        %2 = stream.binding.subspan %arg2[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xf32>>
        return
      }
    }
  }

  // Helper function called from cross_function_call test.
  // CHECK-LABEL: util.func private @dispatch_helper
  util.func private @dispatch_helper(%arg0: !stream.resource<*>, %arg1: index, %arg2: !stream.resource<*>, %arg3: index, %arg4: index) -> !stream.resource<*> {
    // CHECK:      stream.tensor.dispatch @executable_both::@dispatch
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.specialized<123>>
    %result = stream.tensor.dispatch @executable_both::@dispatch(%arg0, %arg2) : (tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%arg1}, tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%arg3}) -> tensor<16xf32> in !stream.resource<*>{%arg4}
    util.return %result : !stream.resource<*>
  }

  util.func public @cross_function_call(%arg0: index) -> !stream.resource<*> {
    %encoded_v1 = util.global.load @encoded_v1 : !stream.resource<constant>
    %encoded_v1_size = util.global.load @encoded_v1_size : index
    %encoded_v2 = util.global.load @encoded_v2 : !stream.resource<constant>
    %encoded_v2_size = util.global.load @encoded_v2_size : index
    %cloned_v1 = stream.async.clone %encoded_v1 : !stream.resource<constant>{%encoded_v1_size} -> !stream.resource<*>{%encoded_v1_size}
    %cloned_v2 = stream.async.clone %encoded_v2 : !stream.resource<constant>{%encoded_v2_size} -> !stream.resource<*>{%encoded_v2_size}
    %result = util.call @dispatch_helper(%cloned_v1, %encoded_v1_size, %cloned_v2, %encoded_v2_size, %arg0) : (!stream.resource<*>, index, !stream.resource<*>, index, index) -> !stream.resource<*>
    util.return %result : !stream.resource<*>
  }
}

// -----

// Test tied operand: dispatch result tied to input operand.
// When encoding changes, both operand and result encoding must change.
// A re-encode op should be inserted after dispatch.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<123>}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<789>]>

// CHECK: util.global private @[[$DEVICE_A:.+]] =
util.global private @device_a = #device_target_local
util.global private @weight : !stream.resource<constant>
util.global private @weight_size : index
util.global private @encoded_v1 : !stream.resource<constant>
util.global private @encoded_v1_size : index
util.global private @encoded_v2 : !stream.resource<constant>
util.global private @encoded_v2_size : index

// CHECK: util.initializer
util.initializer {
  %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf32>
  %0 = stream.resource.size %cst : !stream.resource<constant>
  util.global.store %cst, @weight : !stream.resource<constant>
  util.global.store %0, @weight_size : index
  // CHECK: %[[SOURCE:.+]] = util.global.load @weight
  %source = util.global.load @weight : !stream.resource<constant>
  %source_size = util.global.load @weight_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
  %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size1}
  util.global.store %enc1, @encoded_v1 : !stream.resource<constant>
  util.global.store %size1, @encoded_v1_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
  %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size2}
  util.global.store %enc2, @encoded_v2 : !stream.resource<constant>
  util.global.store %size2, @encoded_v2_size : index

  util.return
}

// Executable with tied operand for in-place update.
stream.executable private @executable_tied {
  stream.executable.export public @dispatch_inplace
  builtin.module {
    func.func @dispatch_inplace(%arg0: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<4096x4096xf32, #encoding1>>
      return
    }
  }
}

// CHECK-LABEL: util.func public @dispatch_with_tied_operand
// CHECK-SAME:    %[[N:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:    %[[M:[a-zA-Z0-9_]+]]: index
util.func public @dispatch_with_tied_operand(%N: index, %M: index) -> !stream.resource<*> {
  %encoded = util.global.load @encoded_v1 : !stream.resource<constant>
  %encoded_size = util.global.load @encoded_v1_size : index
  %cloned = stream.async.clone %encoded : !stream.resource<constant>{%encoded_size} -> !stream.resource<*>{%encoded_size}

  // The dispatch has a tied result (result -> %cloned).
  // CHECK:      %[[DISPATCH:.+]] = stream.tensor.dispatch on(#hal.device.affinity<@[[$DEVICE_A]]>) @executable_tied::@dispatch_inplace
  // CHECK-SAME:   tensor<?x?xf32, #iree_encoding.specialized<123>>{%[[N]], %[[M]]}
  // CHECK-SAME:   -> tensor<?x?xf32, #iree_encoding.specialized<123>>{%[[N]], %[[M]]}
  // Re-encode sizeof and encode ops are inserted after dispatch.
  // CHECK:      stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<?x?xf32, #[[$ENC]]>{%[[N]], %[[M]]}
  // CHECK:      stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[DISPATCH]] :
  // CHECK-SAME:   tensor<?x?xf32, #iree_encoding.specialized<123>>
  // CHECK-SAME:   -> tensor<?x?xf32, #[[$ENC]]>{%[[N]], %[[M]]}
  %result = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @executable_tied::@dispatch_inplace(%cloned)
    : (tensor<?x?xf32, #encoding1>{%N, %M} in !stream.resource<*>{%encoded_size})
    -> tensor<?x?xf32, #encoding1>{%N, %M} in %cloned{%encoded_size}

  util.return %result : !stream.resource<*>
}

// -----

// Test: mutable source global - should be skipped, encoding unchanged.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.identity_resolver}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>

// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK-LABEL: module @mutable_source_skipped
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @mutable_source_skipped {
  util.global private @device_a = #device_target_local
  util.global private mutable @mutable_source : !stream.resource<constant>
  util.global private @encoded : !stream.resource<constant>

  util.initializer {
    %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = dense<0.0> : tensor<4096x4096xf32>
    %cst_size = stream.resource.size %cst : !stream.resource<constant>
    util.global.store %cst, @mutable_source : !stream.resource<constant>

    %source = util.global.load @mutable_source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded : !stream.resource<constant>

    util.return
  }
}

// -----

// Test: mutable encoded global - should be skipped, encoding unchanged.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.identity_resolver}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>

// CHECK: #[[$ENC1:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK: #[[$ENC2:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
// CHECK-LABEL: module @mutable_encoded_global_skipped
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @mutable_encoded_global_skipped {
  util.global private @device_a = #device_target_local
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private mutable @encoded_mutable_v1 : !stream.resource<constant>
  util.global private mutable @encoded_mutable_v2 : !stream.resource<constant>

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #[[$ENC1]]>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #[[$ENC1]]>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_mutable_v1 : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #[[$ENC2]]>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #[[$ENC2]]>
    %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_mutable_v2 : !stream.resource<constant>
    util.return
  }
}

// -----

// Test: single encoding - not a candidate for unification, encoding unchanged.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.identity_resolver}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>

// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK-LABEL: module @single_encoding_no_unification
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @single_encoding_no_unification {
  util.global private @device_a = #device_target_local
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded : !stream.resource<constant>

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded : !stream.resource<constant>

    util.return
  }
}

// -----

// Test: same encoding used twice - not a candidate (only one unique encoding).

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.identity_resolver}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>

// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK-LABEL: module @same_encoding_twice_no_unification
//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
module @same_encoding_twice_no_unification {
  util.global private @device_a = #device_target_local
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v2 : !stream.resource<constant>

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
    %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone on(#hal.device.affinity<@device_a>) %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_v2 : !stream.resource<constant>

    util.return
  }
}

// -----

// The constant is created via stream.tensor.constant with NamedParameterAttr
// and directly encoded without being stored to a source global first.
// Should unify based on the parameter name as the source key.

#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_encoding.specialization_resolver<123>}>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<789>]>

//       CHECK:   util.global private @[[$DEVICE_A:.+]] =
util.global private @device_a = #device_target_local
util.global private @weight : !stream.resource<constant>
util.global private @encoded_v1 : !stream.resource<constant>
util.global private @encoded_v2 : !stream.resource<constant>

// CHECK: util.initializer
util.initializer {
  // The constant is created with parameter attribute, not loaded from a global.
  // CHECK: %[[CST:.+]] = stream.tensor.constant
  %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf32> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf32>
  %cst_size = stream.resource.size %cst : !stream.resource<constant>

  util.global.store %cst, @weight : !stream.resource<constant>

  // CHECK: %[[SOURCE:.+]] = util.global.load @weight
  %weight = util.global.load @weight : !stream.resource<constant>

  // One source is from global load, and the other is from the constant directly.
  // The unification happens based on the parameter name.
  // Note that this is not common if a full global cleanup is performed, since
  // the source will be `stream.tensor.constant` directly. It is mostly for
  // demonstrating that the unification prioritizes parameter names.

  // Should unify to specialized<123> from the resolver.
  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding1> : index
  %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %weight : tensor<4096x4096xf32> in !stream.resource<constant>{%cst_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<constant>{%size1}
  util.global.store %enc1, @encoded_v1 : !stream.resource<constant>

  // Should also unify to specialized<123>.
  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[CST]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.specialized<123>>
  %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf32, #encoding2> : index
  %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %cst : tensor<4096x4096xf32> in !stream.resource<constant>{%cst_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<constant>{%size2}
  util.global.store %enc2, @encoded_v2 : !stream.resource<constant>

  util.return
}

// -----

//------------------------------------------------------------------------------
// #iree_gpu.gpu_encoding_resolver encoding unification tests.
//------------------------------------------------------------------------------

// TODO(hanchung): Add a testing pass in Codegen/ if the test cases expand
// further. We can have basic tests like specialize_encoding.mlir, but we should
// not have too many backend-specific tests in Stream/.
// This is a simplified test case of https://github.com/iree-org/iree/issues/21659

#executable_target_rocm_hsaco_fb = #hal.executable.target<"rocm", "rocm-hsaco-fb",
  {
    abi = "hip",
    iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>
  }>
#device_target_local = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_rocm_hsaco_fb]> : !hal.device
#map = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [?, 4096, 4096]>
#encoding2 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [#map, #map1, #map2], iteration_sizes = [4, 4096, 4096]>

// The GPU resolver's getUnifiedEncoding returns the first encoding, so both
// should be unified to #encoding1 (with iteration_sizes = [?, 4096, 4096]).

// CHECK: #[[$UNIFIED_ENC:.+]] = #iree_encoding.encoding<operand_index = 1 : index, op_type = matmul, element_types = [f8E4M3FNUZ, f8E4M3FNUZ, f32], user_indexing_maps = [{{.+}}], iteration_sizes = [?, 4096, 4096]>
// CHECK: util.global private @[[$DEVICE_A:.+]] =
util.global private @device_a = #device_target_local
util.global private @weight : !stream.resource<constant>
util.global private @weight_size : index
util.global private @encoded_v1 : !stream.resource<constant>
util.global private @encoded_v1_size : index
util.global private @encoded_v2 : !stream.resource<constant>
util.global private @encoded_v2_size : index

// CHECK: util.initializer
util.initializer {
  %cst = stream.tensor.constant on(#hal.device.affinity<@device_a>) : tensor<4096x4096xf8E4M3FNUZ> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf8E4M3FNUZ>
  %0 = stream.resource.size %cst : !stream.resource<constant>
  util.global.store %cst, @weight : !stream.resource<constant>
  util.global.store %0, @weight_size : index
  // CHECK: %[[SOURCE:.+]] = util.global.load @weight
  %source = util.global.load @weight : !stream.resource<constant>
  %source_size = util.global.load @weight_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf8E4M3FNUZ, #[[$UNIFIED_ENC]]>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf8E4M3FNUZ, #[[$UNIFIED_ENC]]>
  %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf8E4M3FNUZ, #encoding1> : index
  %enc1 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf8E4M3FNUZ> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf8E4M3FNUZ, #encoding1> in !stream.resource<constant>{%size1}
  util.global.store %enc1, @encoded_v1 : !stream.resource<constant>
  util.global.store %size1, @encoded_v1_size : index

  // CHECK: stream.tensor.sizeof on(#hal.device.affinity<@[[$DEVICE_A]]>) tensor<4096x4096xf8E4M3FNUZ, #[[$UNIFIED_ENC]]>
  // CHECK: stream.tensor.encode on(#hal.device.affinity<@[[$DEVICE_A]]>) %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf8E4M3FNUZ, #[[$UNIFIED_ENC]]>
  %size2 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<4096x4096xf8E4M3FNUZ, #encoding2> : index
  %enc2 = stream.tensor.encode on(#hal.device.affinity<@device_a>) %source : tensor<4096x4096xf8E4M3FNUZ> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf8E4M3FNUZ, #encoding2> in !stream.resource<constant>{%size2}
  util.global.store %enc2, @encoded_v2 : !stream.resource<constant>
  util.global.store %size2, @encoded_v2_size : index

  util.return
}
