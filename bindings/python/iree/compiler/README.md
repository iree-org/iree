# IREE Compiler Python Bindings

## Core compiler

```py
import iree.compiler

SIMPLE_MUL_ASM = """
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
"""

# Also see compile_file()
# There are many keyword options available.
# See iree.compiler.CompilerOptions
binary = iree.compiler.compile_str(SIMPLE_MUL_ASM, target_backends=["vulkan-spirv"])
```

### Debugging

A number of optional arguments to the compiler can be useful for debugging:

* `extended_diagnostics=True` - Outputs verbose attached operations to
  diagnostics. Can output a large volume of information.
* `crash_reproducer_path=... some .mlir file path...` - On a crash or error,
  a reproducer will be output at the listed path.
* `extra_args=[...]` - Passes extra arguments to the compiler. Useful for
  various standard features of MLIR based compilers like `-print-ir-after-all`.

In addition, the core compiler and frontend compiler APIs have a unified
mechanism for saving their temporary files, which are often useful for post
mortem debugging. Since the need for this is often as part of a larger system,
it is exposed both via an environment variable and an API.

In order to save all temporaries and reproducers, set the `IREE_SAVE_TEMPS`
environment variable to a directory in which to dump artifacts. For complex
programs that invoke the compiler many times, it will typically be necessary
to further qualify the path, and there are a few placeholders that will be
expanded:

* `{id}` - A per-process monotonically increasing number for each compiler
  invocation. Can be overridden by the API if a better symbolic name is
  available (i.e. test case, etc).
* `{pid}` - Process ID of the current process.
* `{main}` - Basename of `sys.argv[0]`, which is typically the name of the
  Python main file.

For interactive use, the following (on a Unix-like system) should provide
value:

```shell
export IREE_SAVE_TEMPS="/tmp/ireedumps/{main}/{id}"
```

For the context manager based API, refer to the
`iree.compiler.debugging.TempFileSaver` class.


## TensorFlow compiler

```py
import tensorflow as tf
import iree.compiler.tf

class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

# Also see compile_saved_model to directly compile an on-disk saved model.
# There are many keyword options available.
# See: iree.compiler.tf.ImportOptions
binary = iree.compiler.tf.compile_module(
    SimpleArithmeticModule(), target_backends=["vulkan-spirv"])
```
