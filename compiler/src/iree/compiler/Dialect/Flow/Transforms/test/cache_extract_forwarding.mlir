// RUN: iree-opt --iree-flow-canonicalize %s --split-input-file --mlir-print-local-scope | FileCheck %s

// A scalar extraction of a functional cache update should read from exactly
// one of the patch or the previous cache, never both unconditionally.
util.func public @forward_static_slice(%base: tensor<16xf32>,
                                       %patch: tensor<2xf32>,
                                       %index: index) -> f32 {
  %updated = tensor.insert_slice %patch into %base [5] [2] [1] : tensor<2xf32> into tensor<16xf32>
  %value = tensor.extract %updated[%index] : tensor<16xf32>
  util.return %value : f32
}
// CHECK-LABEL: util.func public @forward_static_slice
// CHECK: arith.cmpi
// CHECK: scf.if
// CHECK: tensor.extract
// CHECK: scf.yield
// CHECK: else
// CHECK: tensor.extract
// CHECK: scf.yield
// CHECK: util.return

// -----

// A dynamic insertion offset is intentionally not matched by this first patch.
util.func public @do_not_forward_dynamic_offset(%base: tensor<16xf32>,
                                                %patch: tensor<2xf32>,
                                                %offset: index,
                                                %index: index) -> f32 {
  %updated = tensor.insert_slice %patch into %base [%offset] [2] [1] : tensor<2xf32> into tensor<16xf32>
  %value = tensor.extract %updated[%index] : tensor<16xf32>
  util.return %value : f32
}
// CHECK-LABEL: util.func public @do_not_forward_dynamic_offset
// CHECK: tensor.insert_slice
// CHECK: tensor.extract
