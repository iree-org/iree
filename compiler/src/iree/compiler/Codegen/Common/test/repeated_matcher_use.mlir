// RUN: iree-opt %s \
// RUN: --iree-transform-dialect-interpreter \
// RUN: --split-input-file --verify-diagnostics

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_repeated_matcher_use_callback"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.iree.emit_remark "first" at %first : !transform.any_op
    transform.iree.emit_remark "second" at %second : !transform.any_op
    transform.yield
  } // @__transform_main
} // module

module {
  func.func private @f1(f32) -> f32
  func.func private @f2(f32, f32) -> f32

  func.func @foo() -> tensor<10xf32> {
    %dummy1 = tensor.empty() : tensor<10xf32>
    %dummy2 = tensor.empty() : tensor<10xf32>
    %dummy3 = tensor.empty() : tensor<10xf32>
    %c0 = arith.constant 0.0 : f32
    %operand = linalg.fill ins(%c0 : f32) outs(%dummy1 : tensor<10xf32>) -> tensor<10xf32>

    // expected-remark @below {{first}}
    %first = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand : tensor<10xf32>)
      outs(%dummy2 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %0 = func.call @f1(%arg0) : (f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>

    // expected-remark @below {{second}}
    %second = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand, %first : tensor<10xf32>, tensor<10xf32>)
      outs(%dummy3 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = func.call @f2(%arg0, %arg1) : (f32, f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>
    return %second : tensor<10xf32>
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    // expected-error @+2 {{failed to match}}
    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_repeated_matcher_use_callback"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.iree.emit_remark "first" at %first : !transform.any_op
    transform.iree.emit_remark "second" at %second : !transform.any_op
    transform.yield
  } // @__transform_main
} // module

module {
  func.func private @f1(f32) -> f32
  func.func private @f2(f32, f32) -> f32

  func.func @foo() -> tensor<10xf32> {
    %dummy1 = tensor.empty() : tensor<10xf32>
    %dummy2 = tensor.empty() : tensor<10xf32>
    %dummy3 = tensor.empty() : tensor<10xf32>
    %dummy5 = tensor.empty() : tensor<10xf32>
    %c0 = arith.constant 0.0 : f32
    %c5 = arith.constant 5.0 : f32
    %operand5 = linalg.fill ins(%c5 : f32) outs(%dummy5 : tensor<10xf32>) -> tensor<10xf32>
    %operand = linalg.fill ins(%c0 : f32) outs(%dummy1 : tensor<10xf32>) -> tensor<10xf32>

    %first = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand : tensor<10xf32>)
      outs(%dummy2 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %0 = func.call @f1(%arg0) : (f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>

    %second = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand5, %first : tensor<10xf32>, tensor<10xf32>)
      outs(%dummy3 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = func.call @f2(%arg0, %arg1) : (f32, f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>
    return %second : tensor<10xf32>
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    // expected-error @+2 {{failed to match}}
    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_repeated_matcher_use_callback"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.iree.emit_remark "first" at %first : !transform.any_op
    transform.iree.emit_remark "second" at %second : !transform.any_op
    transform.yield
  } // @__transform_main
} // module

module {
  func.func private @f1(f32) -> f32
  func.func private @f2(f32, f32) -> f32

  func.func @foo() -> tensor<10xf32> {
    %dummy1 = tensor.empty() : tensor<10xf32>
    %dummy2 = tensor.empty() : tensor<10xf32>
    %dummy3 = tensor.empty() : tensor<10xf32>
    %dummy5 = tensor.empty() : tensor<10xf32>
    %c0 = arith.constant 0.0 : f32
    %operand = linalg.fill ins(%c0 : f32) outs(%dummy1 : tensor<10xf32>) -> tensor<10xf32>

    %first = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand : tensor<10xf32>)
      outs(%dummy2 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %0 = func.call @f1(%arg0) : (f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>

    %second = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%first, %first : tensor<10xf32>, tensor<10xf32>)
      outs(%dummy3 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = func.call @f2(%arg0, %arg1) : (f32, f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>
    return %second : tensor<10xf32>
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_value_matcher_callback"(%root)
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.iree.emit_remark "first" at %first : !transform.any_op
    transform.iree.emit_remark "second" at %second : !transform.any_op
    transform.yield
  } // @__transform_main
} // module

module {
  func.func private @f1(f32) -> f32
  func.func private @f2(f32, f32) -> f32

  func.func @foo() -> tensor<10xf32> {
    %dummy1 = tensor.empty() : tensor<10xf32>
    %dummy2 = tensor.empty() : tensor<10xf32>
    %dummy3 = tensor.empty() : tensor<10xf32>
    %operand = tensor.empty() : tensor<10xf32>

    // expected-remark @below {{first}}
    %first = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand : tensor<10xf32>)
      outs(%dummy2 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %0 = func.call @f1(%arg0) : (f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>

    // expected-remark @below {{second}}
    %second = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%operand, %first : tensor<10xf32>, tensor<10xf32>)
      outs(%dummy3 : tensor<10xf32>) {
    ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
      %0 = func.call @f2(%arg0, %arg1) : (f32, f32) -> f32
      linalg.yield %0 : f32
    } -> tensor<10xf32>
    return %second : tensor<10xf32>
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    transform.iree.register_match_callbacks

    %0 = transform.iree.match_callback failures(propagate) "_test_shaped_value_matcher_callback"(%root)
      : (!transform.any_op) -> !transform.any_op
    transform.iree.emit_remark "matched" at %0 : !transform.any_op
    transform.yield
  } // @__transform_main
} // module

module {
  func.func @foo(%arg0: tensor<42x10xf32>) -> tensor<10x42xf32> {
    %init = tensor.empty() : tensor<10x42xf32>
    // expected-remark @below {{rank: 2}}
    // expected-remark @below {{dimensions: 10, 42}}
    // expected-remark @below {{matched}}
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0: tensor<42x10xf32>)
      outs(%init: tensor<10x42xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      linalg.yield %arg1 : f32
    } -> tensor<10x42xf32>
    return %0 : tensor<10x42xf32>
  }
}
