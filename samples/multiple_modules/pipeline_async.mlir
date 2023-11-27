// RUN: (iree-compile --iree-execution-model=async-external --iree-hal-target-backends=vmvx %p/module_a.mlir -o=%t.module_a.vmfb && \
// RUN:  iree-compile --iree-execution-model=async-external --iree-hal-target-backends=vmvx %p/module_b.mlir -o=%t.module_b.vmfb && \
// RUN:  iree-compile --iree-execution-model=async-external --iree-hal-target-backends=vmvx %s | \
// RUN:  iree-run-module --device=local-task \
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
//
// In this asynchronous example both functions follow the "coarse-fences" ABI
// model where the compiler inserts a wait and signal fence pair on each call.
// To enable this the modules must compiled with the
// `--iree-execution-model=async-external` and the external declarations must
// be annotated with the `iree.abi.model` attribute so that the compiler knows
// the calls have the fences. Note that it's possible to have any combination of
// asynchronous and synchronous modules and calls in the same program.
func.func private @module_a.abs(%input: tensor<4096xf32>) -> tensor<4096xf32> attributes {
  iree.abi.model = "coarse-fences"
}
func.func private @module_b.mul(%lhs: tensor<4096xf32>, %rhs: tensor<4096xf32>, %output: tensor<4096xf32> {iree.abi.output = 0 : index}) -> tensor<4096xf32> attributes {
  iree.abi.model = "coarse-fences"
}

// Top-level pipeline invoked by the command line tool.
// Since this is compiled with `--iree-execution-model=async-external` this
// export will have a wait and signal fence pair that allows the hosting
// application to execute the entire pipeline asynchronously.
func.func @run(%input: tensor<4096xf32>) -> tensor<4096xf32> {
  // Make a simple call that produces a transient result tensor.
  // Since the call is asynchronous the result is not ready upon return to this
  // function and it'll be passed with the fence down to the consumer call.
  %input_abs = call @module_a.abs(%input) : (tensor<4096xf32>) -> tensor<4096xf32>

  // Allocate output storage for the next call. This isn't needed here and
  // functionally equivalent to `abs` above allocating its own transient memory
  // but demonstrates how in-place operations can be performed across module
  // boundaries. The allocation is asynchronous and will be passed with a fence
  // indicating when it's ready to the consumer call.
  %result_storage = tensor.empty() : tensor<4096xf32>

  // Make a call that produces its output in the given `%result_storage`.
  // The inputs and result storage are passed with their respective fences and
  // no guarantee that they are available at the time the call is made. The
  // `mul` implementation will chain its work with the fences and only signal
  // its fence when all transitive dependencies and its own execution has
  // completed.
  %result = call @module_b.mul(%input_abs, %input_abs, %result_storage) : (tensor<4096xf32>, tensor<4096xf32>, tensor<4096xf32>) -> tensor<4096xf32>

  // Return the final result value - note that we pass back the result of the
  // `mul` call that aliases the `%result_storage` representing the computed
  // value and not just `%result_storage`. This is required as the `%result` has
  // an associated fence indicating when it is available for use and using
  // `%result_storage` would just wait for the storage to be allocated and not
  // for the contents to have been populated by `mul`.
  return %result : tensor<4096xf32>
}
