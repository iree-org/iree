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
