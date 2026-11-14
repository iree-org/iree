// RUN: iree-opt --iree-stablehlo-to-iree-input --verify-diagnostics \
// RUN:   --split-input-file %s
// RUN: split-file %s %t
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls),iree-stablehlo-to-iree-input)" --verify-diagnostics %t/shape-assertion-result.mlir

//--- unsupported-calls.mlir

func.func @unknown_custom_call(%input: tensor<i32>) {
  // expected-error @+1 {{no applicable lowering for custom call target 'test.unknown'}}
  stablehlo.custom_call @test.unknown(%input) {has_side_effect = true} : (tensor<i32>) -> ()
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

// -----
//--- shape-assertion-result.mlir

// Assertions with live results cannot be erased as runtime-only preconditions.
func.func @shape_assertion_with_result(%predicate: tensor<i1>) -> tensor<i1> {
  // expected-error @+1 {{no applicable lowering for custom call target 'shape_assertion'}}
  %result = stablehlo.custom_call @shape_assertion(%predicate) {
    error_message = "unexpected assertion result", has_side_effect = true
  } : (tensor<i1>) -> tensor<i1>
  return %result : tensor<i1>
}
