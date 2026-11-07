// RUN: iree-opt --iree-stablehlo-to-iree-input --verify-diagnostics \
// RUN:   --split-input-file %s

func.func @unknown_custom_call(%input: tensor<i32>) {
  // expected-error @+1 {{no applicable lowering for custom call target 'test.unknown'}}
  stablehlo.custom_call @test.unknown(%input) {has_side_effect = true} : (tensor<i32>) -> ()
  return
}

// -----

func.func @malformed_shape_assertion(%predicate: tensor<i32>) {
  // expected-error @+1 {{no applicable lowering for custom call target 'shape_assertion'}}
  stablehlo.custom_call @shape_assertion(%predicate) {
    error_message = "predicate must be i1", has_side_effect = true
  } : (tensor<i32>) -> ()
  return
}

// -----

// Diagnose each target in the same module, including a pure call whose result
// could otherwise be discarded during conversion.
func.func @multiple_unknown_targets(%input: tensor<i32>) {
  // expected-error @+1 {{no applicable lowering for custom call target 'test.first'}}
  stablehlo.custom_call @test.first(%input) {has_side_effect = true} : (tensor<i32>) -> ()
  // expected-error @+1 {{no applicable lowering for custom call target 'test.unused'}}
  %unused = stablehlo.custom_call @test.unused(%input) : (tensor<i32>) -> tensor<i32>
  return
}
