// RUN: iree-opt --iree-stablehlo-to-iree-input --verify-diagnostics %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-stablehlo-legalize-custom-calls),iree-stablehlo-to-iree-input)" \
// RUN:   --verify-diagnostics %s

// Assertions with live results cannot be erased as runtime-only preconditions.
func.func @shape_assertion_with_result(%predicate: tensor<i1>) -> tensor<i1> {
  // expected-error @+1 {{no applicable lowering for custom call target 'shape_assertion'}}
  %result = stablehlo.custom_call @shape_assertion(%predicate) {
    error_message = "unexpected assertion result", has_side_effect = true
  } : (tensor<i1>) -> tensor<i1>
  return %result : tensor<i1>
}
