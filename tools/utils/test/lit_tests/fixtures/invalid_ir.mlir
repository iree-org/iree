// RUN: iree-opt --verify-diagnostics %s

// This test intentionally contains invalid IR to demonstrate error detection.
// The arith.addi operation expects integer types but receives tensor<f32>.

func.func @invalid_addi() {
  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  // expected-error @+1 {{'arith.addi' op operand #0 must be signless-integer-like}}
  %sum = arith.addi %c0, %c1 : f32
  return
}
