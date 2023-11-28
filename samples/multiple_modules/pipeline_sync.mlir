// RUN: (iree-compile --iree-hal-target-backends=vmvx %p/module_a.mlir -o=%t.module_a.vmfb && \
// RUN:  iree-compile --iree-hal-target-backends=vmvx %p/module_b.mlir -o=%t.module_b.vmfb && \
// RUN:  iree-compile --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-sync \
// RUN:    --module=%t.module_a.vmfb \
// RUN:    --module=%t.module_b.vmfb \
// RUN:    --module=- --function=run \
// RUN:    --input=4096xf32=-2.0 \
// RUN:    --expected_output=4096xf32=4.0) | \
// RUN:  FileCheck %s
// CHECK: [SUCCESS]

// Functions declared in external modules - note `module_name.func_name`.
// `abs` will allocate transient memory to pass back the result.
// `mul` will use the provided output memory to produce the result in-place.
// Note that though the returned SSA tensor value shares its storage with the
// `%output` arg the returned value *must* be used to reference the produced
// version of its contents.
func.func private @module_a.abs(%input: tensor<4096xf32>) -> tensor<4096xf32>
func.func private @module_b.mul(%lhs: tensor<4096xf32>, %rhs: tensor<4096xf32>, %output: tensor<4096xf32> {iree.abi.output = 0 : index}) -> tensor<4096xf32>

// Top-level pipeline invoked by the command line tool.
func.func @run(%input: tensor<4096xf32>) -> tensor<4096xf32> {
  // Make a simple call that produces a transient result tensor.
  %input_abs = call @module_a.abs(%input) : (tensor<4096xf32>) -> tensor<4096xf32>

  // Allocate output storage for the next call. This isn't needed here and
  // functionally equivalent to `abs` above allocating its own transient memory
  // but demonstrates how in-place operations can be performed across module
  // boundaries.
  %result_storage = tensor.empty() : tensor<4096xf32>

  // Make a call that produces its output in the given `%result_storage`.
  %result = call @module_b.mul(%input_abs, %input_abs, %result_storage) : (tensor<4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>

  // Return the final result value - note that we pass back the result of the
  // `mul` call that aliases the `%result_storage` representing the computed
  // value and not just `%result_storage`.
  return %result : tensor<4096xf32>
}
