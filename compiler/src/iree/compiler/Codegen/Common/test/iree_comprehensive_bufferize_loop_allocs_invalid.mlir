// RUN: iree-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-eliminate-empty-tensors))"
// RUN: iree-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-eliminate-empty-tensors{allow-return-allocs-from-loops=false}))"
// RUN: iree-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize))"
// RUN: iree-opt %s --split-input-file --verify-diagnostics --pass-pipeline="builtin.module(func.func(iree-codegen-iree-comprehensive-bufferize{allow-return-allocs-from-loops=false}))"

// Default and explicit false preserve the loop equivalence requirement.
func.func @for_yields_allocation(%input: memref<4xi32>, %count: index) -> memref<4xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %result = scf.for %i = %c0 to %count step %c1 iter_args(%iter = %tensor) -> tensor<4xi32> {
    %copy = bufferization.alloc_tensor() copy(%iter) : tensor<4xi32>
    // expected-error @+1 {{Yield operand #0 is not equivalent to the corresponding iter bbArg}}
    scf.yield %copy : tensor<4xi32>
  }
  %buffer = bufferization.to_buffer %result : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}

// -----

func.func @while_yields_allocation(%input: memref<4xi32>, %condition: i1) -> memref<4xi32> {
  %tensor = bufferization.to_tensor %input restrict writable : memref<4xi32> to tensor<4xi32>
  %result = scf.while (%iter = %tensor) : (tensor<4xi32>) -> tensor<4xi32> {
    scf.condition(%condition) %iter : tensor<4xi32>
  } do {
  ^bb0(%iter: tensor<4xi32>):
    %copy = bufferization.alloc_tensor() copy(%iter) : tensor<4xi32>
    // expected-error @+1 {{Yield operand #0 is not equivalent to the corresponding iter bbArg}}
    scf.yield %copy : tensor<4xi32>
  }
  %buffer = bufferization.to_buffer %result : tensor<4xi32> to memref<4xi32>
  return %buffer : memref<4xi32>
}
