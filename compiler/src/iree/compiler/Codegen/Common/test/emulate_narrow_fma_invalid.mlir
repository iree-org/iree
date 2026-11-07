// RUN: iree-opt --split-input-file --verify-diagnostics --pass-pipeline='builtin.module(func.func(iree-convert-unsupported-float-arith{include-f16=true include-bf16=true include-math=true}))' %s

// Widening this operation does not preserve one final rounding. For example,
// f16 fma(1.5, 1.0029296875, 2^-24) differs from trunc_f16(fma_f32(...)).
func.func @negative_half_fma(%a: f16, %b: f16, %c: f16) -> f16 {
  // expected-error @+1 {{cannot preserve exact narrow FMA semantics}}
  %result = math.fma %a, %b, %c : f16
  return %result : f16
}

// -----

func.func @negative_bfloat_fma(%a: bf16, %b: bf16, %c: bf16) -> bf16 {
  // expected-error @+1 {{cannot preserve exact narrow FMA semantics}}
  %result = math.fma %a, %b, %c : bf16
  return %result : bf16
}
