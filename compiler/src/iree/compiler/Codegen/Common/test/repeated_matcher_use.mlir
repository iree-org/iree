// RUN: iree-opt %s --iree-transform-dialect-interpreter --verify-diagnostics --split-input-file

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.register_match_callbacks

    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_repeated_matcher_use_callback"(%arg0)
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    transform.iree.emit_remark "first" at %first : !pdl.operation
    transform.iree.emit_remark "second" at %second : !pdl.operation
  }

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
}

// -----

// expected-error @below {{transform dialect interpreter failed}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.register_match_callbacks

    // expected-error @+2 {{failed to match}}
    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_repeated_matcher_use_callback"(%arg0)
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    transform.iree.emit_remark "first" at %first : !pdl.operation
    transform.iree.emit_remark "second" at %second : !pdl.operation
  }

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
}

// -----

// expected-error @below {{transform dialect interpreter failed}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.register_match_callbacks

    // expected-error @+2 {{failed to match}}
    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_repeated_matcher_use_callback"(%arg0)
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    transform.iree.emit_remark "first" at %first : !pdl.operation
    transform.iree.emit_remark "second" at %second : !pdl.operation
  }

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
}

// -----

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    transform.iree.register_match_callbacks

    %first, %second =
      transform.iree.match_callback failures(propagate) "_test_value_matcher_callback"(%arg0)
      : (!pdl.operation) -> (!pdl.operation, !pdl.operation)

    transform.iree.emit_remark "first" at %first : !pdl.operation
    transform.iree.emit_remark "second" at %second : !pdl.operation
  }

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
}
