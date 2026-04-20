// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-llvmgpu-vector-flattening))" \
// RUN:   --split-input-file %s | FileCheck %s

// CHECK-LABEL: @vector_flattening
func.func @vector_flattening() {
  return
}
