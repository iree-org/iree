// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-stream-unify-encoding-for-globals)' %s | FileCheck %s

// Test: immutable source global (with initial value) with two encodings -
// should unify to identity encoding.

#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>

// CHECK-LABEL: module @immutable_source_with_initial_value
module @immutable_source_with_initial_value {
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v2 : !stream.resource<constant>

  // CHECK: util.initializer
  util.initializer {
    // CHECK: %[[SOURCE:.+]] = util.global.load @source
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK: stream.tensor.encode %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
    %size1 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK: stream.tensor.encode %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
    %size2 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_v2 : !stream.resource<constant>

    util.return
  }
}

// -----

// Test: immutable source global (initialized from parameter in initializer) with
// two encodings - should unify to identity encoding.

#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>

module @immutable_source_initialized_from_parameter {
  util.global private @weight : !stream.resource<constant>
  util.global private @weight_size : index
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v1_size : index
  util.global private @encoded_v2 : !stream.resource<constant>
  util.global private @encoded_v2_size : index

  // CHECK: util.initializer
  util.initializer {
    %cst = stream.tensor.constant : tensor<4096x4096xf8E4M3FNUZ> in !stream.resource<constant> = #stream.parameter.named<"model"::"weight"> : tensor<4096x4096xf32>
    %0 = stream.resource.size %cst : !stream.resource<constant>
    util.global.store %cst, @weight : !stream.resource<constant>
    util.global.store %0, @weight_size : index
    // CHECK: %[[SOURCE:.+]] = util.global.load @weight
    %source = util.global.load @weight : !stream.resource<constant>
    %source_size = util.global.load @weight_size : index

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK: stream.tensor.encode %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
    %size1 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>
    util.global.store %size1, @encoded_v1_size : index

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK: stream.tensor.encode %[[SOURCE]] : {{.*}} -> tensor<4096x4096xf32, #iree_encoding.identity>
    %size2 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
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

  // Executable that uses both encodings.
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

  // CHECK-LABEL: util.func public @use_encoded_v1
  util.func public @use_encoded_v1(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
    %encoded = util.global.load @encoded_v1 : !stream.resource<constant>
    %encoded_size = util.global.load @encoded_v1_size : index
    %cloned = stream.async.clone %encoded : !stream.resource<constant>{%encoded_size} -> !stream.resource<*>{%encoded_size}

    // CHECK: stream.tensor.dispatch {{.*}} tensor<4096x4096xf32, #iree_encoding.identity>
    %result = stream.tensor.dispatch @executable_v1::@dispatch(%arg0, %cloned) : (tensor<16xf32> in !stream.resource<*>{%arg1}, tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%encoded_size}) -> tensor<16xf32> in !stream.resource<*>{%arg1}

    util.return %result : !stream.resource<*>
  }

  // CHECK-LABEL: util.func public @use_encoded_v2
  util.func public @use_encoded_v2(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
    %encoded = util.global.load @encoded_v2 : !stream.resource<constant>
    %encoded_size = util.global.load @encoded_v2_size : index
    %cloned = stream.async.clone %encoded : !stream.resource<constant>{%encoded_size} -> !stream.resource<*>{%encoded_size}

    // CHECK: stream.tensor.dispatch {{.*}} tensor<4096x4096xf32, #iree_encoding.identity>
    %result = stream.tensor.dispatch @executable_v2::@dispatch(%arg0, %cloned) : (tensor<16xf32> in !stream.resource<*>{%arg1}, tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%encoded_size}) -> tensor<16xf32> in !stream.resource<*>{%arg1}

    util.return %result : !stream.resource<*>
  }

  // Function that uses both encoded globals in the same dispatch.
  // CHECK-LABEL: util.func public @use_both_encodings
  util.func public @use_both_encodings(%arg0: index) -> !stream.resource<*> {
    %encoded_v1 = util.global.load @encoded_v1 : !stream.resource<constant>
    %encoded_v1_size = util.global.load @encoded_v1_size : index
    %encoded_v2 = util.global.load @encoded_v2 : !stream.resource<constant>
    %encoded_v2_size = util.global.load @encoded_v2_size : index

    %cloned_v1 = stream.async.clone %encoded_v1 : !stream.resource<constant>{%encoded_v1_size} -> !stream.resource<*>{%encoded_v1_size}
    %cloned_v2 = stream.async.clone %encoded_v2 : !stream.resource<constant>{%encoded_v2_size} -> !stream.resource<*>{%encoded_v2_size}

    // CHECK:      stream.tensor.dispatch @executable_both::@dispatch
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.identity>
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.identity>
    %result = stream.tensor.dispatch @executable_both::@dispatch(%cloned_v1, %cloned_v2) : (tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%encoded_v1_size}, tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%encoded_v2_size}) -> tensor<16xf32> in !stream.resource<*>{%arg0}

    util.return %result : !stream.resource<*>
  }

  // Helper function called from cross_function_call test.
  // CHECK-LABEL: util.func private @dispatch_helper
  util.func private @dispatch_helper(%arg0: !stream.resource<*>, %arg1: index) -> !stream.resource<*> {
    // CHECK:      stream.tensor.dispatch @executable_v1::@dispatch
    // CHECK-SAME:   tensor<4096x4096xf32, #iree_encoding.identity>
    %result = stream.tensor.dispatch @executable_v1::@dispatch(%arg0, %arg0) : (tensor<16xf32> in !stream.resource<*>{%arg1}, tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%arg1}) -> tensor<16xf32> in !stream.resource<*>{%arg1}
    util.return %result : !stream.resource<*>
  }

  // Test cross-function tracking: load encoded global, pass to callee via util.call.
  // CHECK-LABEL: util.func public @cross_function_call
  util.func public @cross_function_call() -> !stream.resource<*> {
    %encoded = util.global.load @encoded_v1 : !stream.resource<constant>
    %encoded_size = util.global.load @encoded_v1_size : index
    %cloned = stream.async.clone %encoded : !stream.resource<constant>{%encoded_size} -> !stream.resource<*>{%encoded_size}
    %result = util.call @dispatch_helper(%cloned, %encoded_size) : (!stream.resource<*>, index) -> !stream.resource<*>
    util.return %result : !stream.resource<*>
  }
}

// -----

// Test: mutable source global - should be skipped, encoding unchanged.

#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>

// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK-LABEL: module @mutable_source_skipped
module @mutable_source_skipped {
  util.global private mutable @mutable_source : !stream.resource<constant>
  util.global private @encoded : !stream.resource<constant>

  util.initializer {
    %cst = stream.tensor.constant : tensor<4096x4096xf32> in !stream.resource<constant> = dense<0.0> : tensor<4096x4096xf32>
    %cst_size = stream.resource.size %cst : !stream.resource<constant>
    util.global.store %cst, @mutable_source : !stream.resource<constant>

    %source = util.global.load @mutable_source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size1 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded : !stream.resource<constant>

    util.return
  }
}

// -----

// Test: mutable encoded global - should be skipped, encoding unchanged.

#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
#encoding2 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>

// CHECK: #[[$ENC1:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK: #[[$ENC2:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<456>]>
// CHECK-LABEL: module @mutable_encoded_global_skipped
module @mutable_encoded_global_skipped {
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private mutable @encoded_mutable_v1 : !stream.resource<constant>
  util.global private mutable @encoded_mutable_v2 : !stream.resource<constant>

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #[[$ENC1]]>
    // CHECK: stream.tensor.encode {{.*}} -> tensor<4096x4096xf32, #[[$ENC1]]>
    %size1 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_mutable_v1 : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #[[$ENC2]]>
    // CHECK: stream.tensor.encode {{.*}} -> tensor<4096x4096xf32, #[[$ENC2]]>
    %size2 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding2> : index
    %enc2 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding2> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_mutable_v2 : !stream.resource<constant>
    util.return
  }
}

// -----

// Test: single encoding - not a candidate for unification, encoding unchanged.

#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>

// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK-LABEL: module @single_encoding_no_unification
module @single_encoding_no_unification {
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded : !stream.resource<constant>

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size1 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded : !stream.resource<constant>

    util.return
  }
}

// -----

// Test: same encoding used twice - not a candidate (only one unique encoding).

#encoding1 = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>

// CHECK: #[[$ENC:.+]] = #iree_encoding.testing<layouts = [#iree_encoding.specialized<123>]>
// CHECK-LABEL: module @same_encoding_twice_no_unification
module @same_encoding_twice_no_unification {
  util.global private @source = #stream.parameter.named<"model"::"weight"> : !stream.resource<constant>
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v2 : !stream.resource<constant>

  util.initializer {
    %source = util.global.load @source : !stream.resource<constant>
    %source_size = stream.resource.size %source : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size1 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc1 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size1}
    %const1 = stream.async.clone %enc1 : !stream.resource<*>{%size1} -> !stream.resource<constant>{%size1}
    util.global.store %const1, @encoded_v1 : !stream.resource<constant>

    // CHECK: stream.tensor.sizeof tensor<4096x4096xf32, #[[$ENC]]>
    // CHECK: stream.tensor.encode {{.*}} -> tensor<4096x4096xf32, #[[$ENC]]>
    %size2 = stream.tensor.sizeof tensor<4096x4096xf32, #encoding1> : index
    %enc2 = stream.tensor.encode %source : tensor<4096x4096xf32> in !stream.resource<constant>{%source_size} -> tensor<4096x4096xf32, #encoding1> in !stream.resource<*>{%size2}
    %const2 = stream.async.clone %enc2 : !stream.resource<*>{%size2} -> !stream.resource<constant>{%size2}
    util.global.store %const2, @encoded_v2 : !stream.resource<constant>

    util.return
  }
}
