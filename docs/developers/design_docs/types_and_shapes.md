# Types and Shapes

IREE supports compiling programs from a variety of frontend frameworks to a
number of backends and uses a collection of MLIR dialects and passes to connect
between each slice through the system. Each layer of the stack has its its own
views on data types and shapes.

* Data _type_ here refers to an attribute of data which describes its meaning,
  defines operations that can be performed on it, and gives information about
  how it can be stored. Examples of data types are `integer`, `float`, and
  `string`. See [the Wikipedia page on data types](https://en.wikipedia.org/wiki/Data_type)
  for more background.
* Data _shape_ here refers to an attribute of multidimensional data (scalars,
  matrices, tensors) which describes the number of elements in each axis of the
  data. Shapes are comprised of a rank (the number of axes, if defined) and a
  list of dimensions, one element per axis. Some example shapes are `[3, 4]`,
  `[*]` (unranked), and `[?, 2]` (ranked with one unknown dimension). See the
  [MLIR 'shape' Dialect documentation](https://mlir.llvm.org/docs/Dialects/ShapeDialect/)
  for more background.

Frontend references:

* TensorFlow: [Introduction to Tensors](https://www.tensorflow.org/guide/tensor)
* PyTorch: [`torch.Tensor` documentation](https://pytorch.org/docs/stable/tensors.html)
* NumPy: [Data types documentation](https://numpy.org/doc/stable/user/basics.types.html)

Backend references:

* Vulkan: [buffer and image formats](https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#formats)
* SPIR-V: [types](https://www.khronos.org/registry/SPIR-V/specs/1.0/SPIRV.html#_types) and [capabilities](https://www.khronos.org/registry/SPIR-V/specs/1.0/SPIRV.html#_a_id_capability_a_capability)

## Types

Types can roughly be grouped in a few different ways:

* Primitive (`char`, `int`) vs composite (`string`, `array<int>`)
* Signed (`int`, `int32_t`) vs unsigned (`unsigned`, `uint32_t`) vs signless
* Fixed width (`int32_t`) vs variable width (`int`, `index`, `uintptr_t`)
* Real (`float32`) vs complex (`tf.complex64`)
* Concrete vs opaque (`void*`, API internal structs, hardware image formats)
* Quantized data types (`bfloat16`)

Types are least constrained in user code within high level frameworks, where
composite types such as Python classes, media files, Protocol Buffers, JSON
objects, and other data structures can be freely created and transformed.
Meanwhile, types are most constrained by hardware and device APIs, where only
specific low level primitives are defined or where certain operations are
supported by efficient hardware implementations.

### Conversion process

IREE lowers programs from representations produced by high level frondends down
to low level host code with scheduling logic and device code containing fused
kernels of dense computation. The phases of compilation can be segmented by
which MLIR dialects are primarily being transformed:

```
Frontends (PyTorch, JAX, TensorFlow, TOSA, etc.)
  * Includes user code, serialized ML models / programs, and other libraries

                                    ↓

Import dialects (`iree`, `tensor`, `linalg`, etc.)

                                    ↓

`flow` dialect (tensor program modeling and compute workload partitioning)

                                    ↓

               `stream` dialect                    |      code generation
  (device placement and asynchronous scheduling)   |    (SPIR-V, LLVM, etc.)

                                    ↓

`hal` dialect (Hardware Abstraction Layer for buffer and execution management)

                                    ↓

`vm` dialect (Virtual Machine for setting up and dispatching workloads)
```

See also https://google.github.io/iree/#project-architecture.

#### Requirements for import dialects

#### Requirements for `flow` dialect

#### Requirements for `stream` dialect

#### Requirements for code generation

TODO: LLVM / SPIR-V emulation of types?

#### Requirements for `hal` dialect

The Hardware Abstraction Layer maps nearly directly to underlying hardware APIs
such as Vulkan, Metal, and CUDA.

* No tensor types. Buffers of primitives or explicitly supported opaque data
  types.
* Supported primitives vary per target backend and may be optionally available.
  Generally expect for int32 and float32 to be well supported for mobile to
  desktop -scale devices and for lower or higher bit depth types (e.g. float16,
  int64) to be optionally available. On embedded systems or certain
  accelerators there may be no floating support at all.

#### Requirements for `vm` dialect

IREE's Virtual Machine aims to be maximally portable, so it implements support
for i64, f32, and f64 behind extensions. See
[iree/base/config.h](https://github.com/google/iree/blob/main/iree/base/config.h)
for the specifics of each extension.

### Strategies for converting between types

#### Emulating

#### Truncating / Demotion

#### Extending / Promotion

#### Packing

TODO: pack i1 into i8/i32 (vectorization)

## Shapes

TODO: static vs dynamic
TODO: ranked vs unranked
TODO: shape inference, https://mlir.llvm.org/docs/ShapeInference/

## Layouts and tiling

TODO: dense vs sparse
TODO: dispatch grids
