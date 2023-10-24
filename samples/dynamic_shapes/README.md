# "Dynamic Shapes" sample

This sample shows how to

1. Create a program that includes dynamic shapes in program inputs and outputs
2. Import that program into IREE's compiler
3. Compile that program to an IREE VM bytecode module
4. Load the compiled program using IREE's high level runtime C API
5. Call exported functions on the loaded program

Steps 1-2 are performed in Python via the
[`pytorch_dynamic_shapes.ipynb`](./pytorch_dynamic_shapes.ipynb) or
[`tensorflow_dynamic_shapes.ipynb`](./tensorflow_dynamic_shapes.ipynb)
[Colab](https://colab.google/) notebooks:

| Framework | Notebook |
| --------- | -------- |
PyTorch | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/dynamic_shapes/pytorch_dynamic_shapes.ipynb)
TensorFlow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openxla/iree/blob/main/samples/dynamic_shapes/tensorflow_dynamic_shapes.ipynb)

Step 3 should be performed on your development host machine

Steps 4-5 are in [`main.c`](./main.c)

The program used to demonstrate includes functions with varying uses of
dynamic shapes:

```python
import torch
import shark_turbine.aot as aot

class DynamicShapesModule(aot.CompiledModule, export_name="module"):
  # reduce_sum_1d (dynamic input size, static output size)
  #   tensor<?xi32> -> tensor<i32>
  #   e.g. [1, 2, 3] -> 6
  def reduce_sum_1d(self, values=aot.AbstractTensor(None, dtype=torch.int32)):
    return self.compute_reduce_sum_1d(values)

  @aot.jittable
  def compute_reduce_sum_1d(values):
    return torch.sum(values, dtype=torch.int32)

  # reduce_sum_2d (partially dynamic input size, static output size)
  #   tensor<?x3xi32> -> tensor<3xi32>
  #   e.g. [[1, 2, 3], [10, 20, 30]] -> [11, 22, 33]
  def reduce_sum_2d(self, values=aot.AbstractTensor(None, 3, dtype=torch.int32)):
    return self.compute_reduce_sum_2d(values)

  @aot.jittable
  def compute_reduce_sum_2d(values):
    return torch.sum(values, 0, dtype=torch.int32)

  # add_one (dynamic input size, dynamic output size)
  #   tensor<?xi32>) -> tensor<?xi32>
  #   e.g. [1, 2, 3] -> [2, 3, 4]
  def add_one(self, values=aot.AbstractTensor(None, dtype=torch.int32)):
    return self.compute_add_one(values)

  @aot.jittable
  def compute_add_one(values):
    return values + 1
```

## Background

Tensors are multi-dimensional arrays with a uniform type (e.g. int32, float32)
and a shape. Shapes consist of a rank and a list of dimensions and may be
static (i.e. fully known and fixed) or varying degrees of dynamic. For more
information, see these references:
* PyTorch:
[Compiler dynamic shapes](https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html),
[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)
* TensorFlow: [Introduction to Tensors](https://www.tensorflow.org/guide/tensor)

Dynamic shapes are useful for passing variable sized batches as input,
receiving variable length sentences of text as output, etc.

NOTE: as in other domains, providing more information to a compiler allows it
to generate more efficient code. As a general rule, the slowest varying
dimensions of program data like batch index or timestep are safer to treat as
dynamic than faster varying dimensions like image x/y/channel. See
[this paper](https://arxiv.org/pdf/2006.03031.pdf) for a discussion of the
challenges imposed by dynamic shapes and one project's approach to addressing
them.

## Instructions

1. Run either Colab notebook and download the `dynamic_shapes.mlir` file it
    generates

2. Build the `iree-compile` tool (see
    [here](https://iree.dev/building-from-source/getting-started/)
    for general instructions on building using CMake)

    ```
    cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo .
    cmake --build ../iree-build/ --target iree-compile
    ```

3. Compile the `dynamic_shapes.mlir` file using `iree-compile`. The
    [CPU configuration](https://iree.dev/guides/deployment-configurations/cpu/)
    has the best support for dynamic shapes:

    ```
    ../iree-build/tools/iree-compile \
        --iree-hal-target-backends=llvm-cpu \
        dynamic_shapes.mlir -o dynamic_shapes_cpu.vmfb
    ```

4. Build the `iree_samples_dynamic_shapes` CMake target

    ```
    cmake --build ../iree-build/ --target iree_samples_dynamic_shapes
    ```

    Alternatively if using a non-CMake build system the `Makefile` provided can
    be used as a reference of how to use the IREE runtime in an external
    project.

5. Run the sample binary:

   ```
   ../iree-build/samples/dynamic_shapes/dynamic-shapes \
       /path/to/dynamic_shapes_cpu.vmfb local-task
   ```
