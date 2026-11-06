// RUN: iree-opt --split-input-file --verify-diagnostics --iree-convert-bf16-arith-to-f32 %s

// f32 evaluation followed by a BF16 cast can double-round a fused result.
func.func @negative_scalar_fma(%a: bf16, %b: bf16, %c: bf16) -> bf16 {
  // expected-error @+1 {{failed to legalize operation 'math.fma'}}
  %result = math.fma %a, %b, %c : bf16
  return %result : bf16
}

// -----

func.func @negative_vector_fma(%a: vector<4xbf16>, %b: vector<4xbf16>,
                               %c: vector<4xbf16>) -> vector<4xbf16> {
  // expected-error @+1 {{failed to legalize operation 'vector.fma'}}
  %result = vector.fma %a, %b, %c : vector<4xbf16>
  return %result : vector<4xbf16>
}
