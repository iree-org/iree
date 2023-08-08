// RUN: iree-compile --iree-hal-target-backends=vmvx %s | iree-run-module --device=local-task --module=- --function=passthrough_scalar --input=-1.2 | FileCheck %s --check-prefix=EXEC
// RUN: iree-compile --iree-hal-target-backends=webgpu %s

// EXEC-LABEL: EXEC @passthrough_scalar
func.func @passthrough_scalar(%input_0 : f32) -> f32 {
  return %input_0 : f32
}
// EXEC-NEXT: result[0]: f32=-1.2
