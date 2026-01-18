// RUN: rm -f %t_main.vmfb %t_encoder.mlir %t_encoder.vmfb %t_input.irpa %t_output.irpa
//
// Compile main module with encoder MLIR output and splat parameter export.
// RUN: iree-compile %s \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --iree-parameter-encoder-output-file=%t_encoder.mlir \
// RUN:   --iree-parameter-splat=%t_input.irpa \
// RUN:   -o %t_main.vmfb
//
// Compile the encoder module separately.
// RUN: iree-compile %t_encoder.mlir \
// RUN:   --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   -o %t_encoder.vmfb
//
// Run the encoder to transform parameters.
// RUN: iree-encode-parameters \
// RUN:   --module=%t_encoder.vmfb \
// RUN:   --parameters=model=%t_input.irpa \
// RUN:   --output=encoded=%t_output.irpa \
// RUN:   --quiet
//
// Run the main module with both input and encoded parameters.
// The encoded parameters contain the pre-computed transformed values.
// RUN: iree-run-module \
// RUN:   --device=local-sync \
// RUN:   --module=%t_main.vmfb \
// RUN:   --function=main \
// RUN:   --parameters=model=%t_input.irpa \
// RUN:   --parameters=encoded=%t_output.irpa | \
// RUN: FileCheck %s

// Test parameter transformation with encoder.
// The global loads a parameter and applies an add operation to transform it.
// The encoder runs the add offline, and the main module loads the
// pre-computed result from the encoded parameter scope.

// CHECK-LABEL: EXEC @main
// CHECK: 256xi32=42 42 42 42

// Parameter loaded from input archive (model scope).
// The splat export creates this with all zeros.
util.global private @raw_param = #flow.parameter.named<"model"::"param_global"> : tensor<256xi32>

// This global holds the transformed value.
util.global private @transformed : tensor<256xi32>

util.initializer {
  // Load the raw parameter (all zeros from splat).
  %raw = util.global.load @raw_param : tensor<256xi32>
  // Add 42 to each element - this uses the parameter values and can be encoded.
  // With input of 0s, result is 42s.
  %c42 = arith.constant 42 : i32
  %init = tensor.empty() : tensor<256xi32>
  %c42_tensor = linalg.fill ins(%c42 : i32) outs(%init : tensor<256xi32>) -> tensor<256xi32>
  %added = linalg.add ins(%raw, %c42_tensor : tensor<256xi32>, tensor<256xi32>) outs(%init : tensor<256xi32>) -> tensor<256xi32>
  util.global.store %added, @transformed : tensor<256xi32>
  util.return
}

func.func @main() -> tensor<256xi32> {
  // Load and return the full transformed tensor.
  // If encoding worked, all elements should be 42 (0 + 42).
  // If encoding didn't work, all elements would be 0 (splat init).
  %tensor = util.global.load @transformed : tensor<256xi32>
  return %tensor : tensor<256xi32>
}
