# "Variables and State" sample

This sample shows how to

1. Create a TensorFlow program that tracks state with an internal variable
2. Import that program into IREE's compiler
3. Compile that program to an IREE VM bytecode module
4. Load the compiled program using IREE's high level runtime C API
5. Call exported functions on the loaded program to interact with the internal
   variable

Steps 1-3 are performed in Python via the
[`variables_and_state.ipynb`](./variables_and_state.ipynb)
[Colab](https://research.google.com/colaboratory/) notebook:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/variables_and_state/variables_and_state.ipynb)

Steps 4-5 are in [`main.c`](./main.c)

The program used to demonstrate tracks a single integer counter and provides
a few functions for interacting with it:

```python
class CounterModule(tf.Module):
  def __init__(self):
    super().__init__()
    self.counter = tf.Variable(0)

  @tf.function(input_signature=[])
  def get_value(self):
    return self.counter

  @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
  def set_value(self, new_value):
    self.counter.assign(new_value)

  @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
  def add_to_value(self, x):
    self.counter.assign(self.counter + x)

  @tf.function(input_signature=[])
  def reset_value(self):
    self.set_value(0)
```

## Background

Just like in other programming models, _variables_ in this context are used to
represent some persistent state that programs may manipulate. See TensorFlow's
[Introduction to Variables](https://www.tensorflow.org/guide/variable) for
more information on using variables in TensorFlow.

Mutable variables and internal program state are modeled natively in IREE along
with the rest of a program's executable code and data. Variables are not given
special treatment - program authors must define explicit exported functions for
interacting with them and then invoke these functions as they would any other
functions in the compiled programs.

## Instructions

1. Run the Colab notebook and download the `counter.mlir` and
   `counter_vmvx.vmfb` files it generates

2. Build the `iree_samples_variables_and_state` CMake target (see
    [here](https://iree.dev/building-from-source/getting-started/)
    for general instructions on building using CMake)

    ```
    cmake -B ../iree-build/ -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo .
    cmake --build ../iree-build/ --target iree_samples_variables_and_state
    ```

3. Run the sample binary:

   ```
   ../iree-build/samples/variables_and_state/variables-and-state \
       /path/to/counter_vmvx.vmfb local-task
   ```

### Changing compilation options

The provided Colab notebook imports the TensorFlow program into MLIR and then
compiles it further to an IREE VM bytecode module for IREE's reference CPU
executable format named "VMVX". To use a different executable format or backend,
set compilation options, or include source code changes to the compiler, you can
compile the imported MLIR file using IREE's tools on your own machine.

For example, to use IREE's `cpu` target, which is optimized for CPU execution
using LLVM, refer to the
[documentation](https://iree.dev/guides/deployment-configurations/cpu/)
and compile the imported `counter.mlir` file using `iree-compile`:

```
../iree-build/tools/iree-compile \
    --iree-hal-target-backends=llvm-cpu \
    --iree-input-type=stablehlo \
    counter.mlir -o counter_cpu.vmfb
```

then run the program with that new VM bytecode module:

```
../iree-build/samples/variables_and_state/variables-and-state \
    /path/to/counter_cpu.vmfb local-task
```
