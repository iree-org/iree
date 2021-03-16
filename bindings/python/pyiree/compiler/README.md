# IREE Compiler Python Bindings

Transitional note: These bindings are not complete yet and will ultimately
replace the `pyiree.compiler` and `pyiree.tf.compiler` packages.

## Core compiler

```py
from pyiree.compiler import *

SIMPLE_MUL_ASM = """
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
      attributes { iree.module.export } {
    %0 = "mhlo.multiply"(%arg0, %arg1) {name = "mul.1"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
}
"""

# Also see compile_file()
# There are many keyword options available.
# See pyiree.compiler.CompilerOptions
binary = compile_str(SIMPLE_MUL_ASM, target_backends=["vulkan-spirv"])
```


## TensorFlow compiler

```py
import tensorflow as tf
from pyiree.compiler.tf import *

class SimpleArithmeticModule(tf.Module):

  @tf.function(input_signature=[
      tf.TensorSpec([4], tf.float32),
      tf.TensorSpec([4], tf.float32)
  ])
  def simple_mul(self, a, b):
    return a * b

# Also see compile_saved_model to directly compile an on-disk saved model.
# There are many keyword options available.
# See: pyiree.compiler.tf.ImportOptions
binary = compile_module(
    SimpleArithmeticModule(), target_backends=["vulkan-spirv"])
```
