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

// CHECK-LABEL: module @immutable_source_initialized_from_parameter
module @immutable_source_initialized_from_parameter {
  util.global private @weight : !stream.resource<constant>
  util.global private @weight_size : index
  util.global private @encoded_v1 : !stream.resource<constant>
  util.global private @encoded_v2 : !stream.resource<constant>

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
