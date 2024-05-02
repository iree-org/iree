---
date: 2024-01-29
authors:
  - bjacob
categories:
  - Performance
tags:
  - CPU
slug: iree-mlir-linalg-tutorial  # Fix extra `-` in e.g. `iree--mlir`
---

# IREE / MLIR / Linalg tutorial

## Introduction

This tutorial is simultaneously about IREE, MLIR, and specifically the MLIR
Linalg dialect.

### What is MLIR?

[MLIR](https://mlir.llvm.org/) is a programming language, but MLIR in itself is
almost just an empty shell. What it really provides is a framework allowing to
define [MLIR dialects](https://mlir.llvm.org/docs/Dialects/) which are where the
features come from.

The "IR" part of the MLIR name stands for "intermediate representation". It
means that MLIR is meant to be primarily for compiler-internal representations
of code. But MLIR is actually fairly nice for humans to work with, and it's not
hard to hand-author some MLIR programs from scratch. That is exactly the topic
of this tutorial.

<!-- more -->

The "ML" part of the MLIR name stands for "multi-level" (not machine learning!).
It means that MLIR allows for multiple dialects to be freely mixed in the same
MLIR programs. Each dialect can define operations, types and attributes, and
each single MLIR statement can mix ops, types and attributes coming from
different dialects.

### What is the Linalg dialect?

[Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) is a MLIR dialect that
essentially consists of a single op, `linalg.generic`, with most other ops in
this dialect being just convenience aliases for special cases of
`linalg.generic`. So, to describe Linalg dialect is essentially to describe
`linalg.generic`.

The point of this is that this single op, `linalg.generic`, is:

* General enough to express the entirety of usual machine learning workloads in
  any quantization scheme at all.
* High-level enough to be lowered to efficient code for any target (CPU, GPU,
  ...)
* Designed to be a good fit for compiler IR-to-IR transformations.

These traits make the Linalg dialect an ideal "middle-end" IR for a machine
learning compiler.

### What is IREE?

[IREE](https://iree.dev/) is a MLIR compiler and runtime that can lower MLIR
programs through successive, ever lower-level dialects, ultimately producing
machine code for various CPU, GPU and other hardware targets. Check out the
[Developer overview docs](../../../developers/general/developer-overview.md)
and the [ML frameworks docs](../../../guides/ml-frameworks/index.md).

Front-ends can ingest source programs from various machine-learning frameworks
into MLIR Linalg dialect. Boundaries are in flux, but it is a good enough mental
model to think of anything up to Linalg as "front-end". One example is, for
[ingesting PyTorch](../../../guides/ml-frameworks/pytorch.md) programs,
the front-end is [torch-mlir](https://github.com/llvm/torch-mlir) and end-users
are encouraged to use [iree-turbine](https://github.com/iree-org/iree-turbine),
which integrates IREE, torch-mlir and PyTorch.

This tutorial is only concerned about the Linalg dialect, and we are going to
learn to hand-author some Linalg programs. The point of the above tangent about
front-ends is to make it clear that no matter which way you feed a program into
IREE, it will internally be rewritten into a Linalg program, because that really
is the intermediate representation in this compiler.

### Getting IREE binaries

IREE builds can be [downloaded](https://github.com/iree-org/iree/releases) or
installed as Python [packages](../../../reference/bindings/python.md) or
[built](../../../building-from-source/index.md) from sources.

## First linalg programs

Before we start: there is also an official
[Linalg tutorial](https://mlir.llvm.org/docs/Tutorials/transform/Ch0/). It takes
a different approach compared to the present tutorial, so the two are complementary.

### Static-shape, element-wise addition of two 1D arrays

Here is our first Linalg function. The scalar type used in this program, `f32`,
is 32-bit floating-point.

Notice some elements of [MLIR syntax](https://mlir.llvm.org/docs/LangRef):

* The `%` prefix on an identifier indicates a
  [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) value, like
  here `%result`.
* The `@` prefix on an identifier indicates a function, like here `@foo`.
* The `^` prefix on an identifier indicates a
  [block](https://mlir.llvm.org/docs/LangRef/#blocks), like here `^bb0`.
* The `#` prefix on an identifier indicates an
  [attribute alias](https://mlir.llvm.org/docs/LangRef/#attribute-value-aliases),
  like here `#map_1d_identity`.
* The `x` letter is used as delimiter in shapes, and between the shape and the
  element type, like here `10xf32` meaning a 1D shape of size 10 with element
  type `f32`.
* Operations have the form `dialect.name`. For example, `tensor.empty` is the
  `empty` operation within the `tensor` dialect, and `func.func` is the `func`
  operation within the `func` dialect.

```mlir
// The 1D identity map, used below.
#map_1d_identity = affine_map<(m) -> (m)>

// Define a function @foo taking two tensor arguments `%lhs` and `%rhs` and returning a tensor.
func.func @foo(
      %lhs : tensor<10xf32>,
      %rhs : tensor<10xf32>
    ) -> tensor<10xf32> {
  // A constant used below.
  %c0f32 = arith.constant 0.0 : f32
  // Create a result "init value". Think of it as an abstract "allocation",
  // creating a tensor but not giving its elements any particular value. It would be
  // undefined behavior to read any element from this tensor.
  %result_empty =  tensor.empty() : tensor<10xf32>

  // Perform the computation. The following is all a single linalg.generic op.

  %result = linalg.generic {
    // This {...} section is the "attributes" - some compile-time settings for this op.
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_identity
    ],
    // There is one tensor dimension, and it's a parallel-iteration dimension,
    // meaning that it occurs also as a result tensor dimension. The alternative
    // would be "reduction", for dimensions that do not occur in the result tensor.
    iterator_types=["parallel"]
  } // End of the attributes for this linalg.generic. Next come the parameters:
    // `ins` is where we pass regular input-parameters
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    // `outs` is where we pass the "outputs", but that term has a subtle meaning
    // in linalg. Here we are passing a tensor.empty, meaning just a placeholder
    // for the output with no preexisting element values. In other examples with
    // an accumulator, this is where the accumulator would be passed.
    outs(%result_empty : tensor<10xf32>)
    // End of parameters. The next {...} part is the "code block".
  {
    // bb0 is a code block taking one scalar from each input tensor as argument, and
    // computing and "yielding" (ie returning) the corresponding output tensor element.
    ^bb0(%lhs_entry : f32, %rhs_entry : f32, %unused_result_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  } // End of the basic block. Finally, we describe the return type.
  -> tensor<10xf32>

  // End of the linalg.generic op.

  // Return the function's return value.
  return %result : tensor<10xf32>
}
```

Compile it like this:

```bash
iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
```

!!! note

    These are just minimalist `iree-compile` flags for running on CPU without
    trying to maximize performance.

    * To run on GPU or other non-CPU targets, explore other values for
      `--iree-hal-target-backends=`. You will then need to pass a matching
      `--device=` to `iree-run-module` below.
    * To cross-compile, explore `--iree-llvmcpu-target-triple=`.
    * To enable higher CPU performance by enabling CPU features:
        * On x86, explore `--iree-llvmcpu-target-cpu=` (e.g.
          `--iree-llvmcpu-target-cpu=znver4` to target AMD Zen4).
        * On other architectures, explore `--iree-llvmcpu-target-cpu-features=`.
        * To optimize for running on the same machine that the compilation ran
          on, pass  `--iree-llvmcpu-target-cpu=host`. That works regardless of
          CPU architecture.
    * Check out
      [these docs](../../../developers/general/developer-tips.md) for
      more useful `iree-compile` flags.

Run it like this:

```console
$ iree-run-module --module=/tmp/prog.vmfb \
  --input=10xf32=[0,1,2,3,4,5,6,7,8,9] \
  --input=10xf32=[90,80,70,60,50,40,30,20,10,0]

EXEC @foo
result[0]: hal.buffer_view
10xf32=90 81 72 63 54 45 36 27 18 9
```

Here, each `--input` parameter specifies one input. First its shape and element
type, `10xf32`, then the example array elements in `[...]` brackets. The output
of `iree-run-module` above shows the contents of the result.

### Dynamic-shape, element-wise addition of two 1D arrays

While we are going to mostly focus on static shapes for simplicity in the rest
of this tutorial, let us give one dynamic-shape example to at least show that
that's not a problem. Here is the dynamic-shape equivalent of the previous
example.

```mlir
#map_1d_identity = affine_map<(m) -> (m)>

func.func @foo(
      %lhs : tensor<?xf32>,
      %rhs : tensor<?xf32>
    ) -> tensor<?xf32> {
  %c0f32 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %size = tensor.dim %lhs, %c0 : tensor<?xf32>
  %result_empty =  tensor.empty(%size) : tensor<?xf32>

  %result = linalg.generic {
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_identity
    ],
    iterator_types=["parallel"]
  } ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
    outs(%result_empty : tensor<?xf32>)
  {
    ^bb0(%lhs_entry : f32, %rhs_entry : f32, %unused_result_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  }
  -> tensor<?xf32>

  return %result : tensor<?xf32>
}
```

This program can be compiled and run exactly like the previous one, except that
now the `iree-run-module` command may specify inputs of arbitrary length. The
only requirement is that both inputs have the same length, otherwise the
`linalg.generic` will have undefined behavior.

```console
$ iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
$ iree-run-module --module=/tmp/prog.vmfb \
  --input=10xf32=[0,1,2,3,4,5,6,7,8,9] \
  --input=10xf32=[90,80,70,60,50,40,30,20,10,0]

EXEC @foo
result[0]: hal.buffer_view
10xf32=90 81 72 63 54 45 36 27 18 9
```

### Passing one of the inputs in `outs`

Here is a more concise variant achieving the same result in fewer lines of code,
and giving us a first taste of that that `outs(...)` parameters list can do. We
didn't want to show it first, because it's less idiomatic. `outs` will only
become really necessary (and idiomatic) when we will look at `reduction`
iterators. In the previous examples, we had only passed a `tensor.empty`
placeholder for `outs`. This new example shows that we can actually pass there
any of the inputs that are shaped like the result.

```mlir
#map_1d_identity = affine_map<(m) -> (m)>

func.func @foo(
      %lhs : tensor<10xf32>,
      %rhs : tensor<10xf32>
    ) -> tensor<10xf32> {

  %result = linalg.generic {
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_identity
    ],
    iterator_types=["parallel"]
  } ins(%lhs : tensor<10xf32>)
    outs(%rhs : tensor<10xf32>)
  {
    ^bb0(%lhs_entry : f32, %rhs_entry : f32):
      %add = arith.addf %lhs_entry, %rhs_entry : f32
      linalg.yield %add : f32
  }
  -> tensor<10xf32>

  return %result : tensor<10xf32>
}
```

```console
$ iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
$ iree-run-module --module=/tmp/prog.vmfb \
  --input=10xf32=[0,1,2,3,4,5,6,7,8,9] \
  --input=10xf32=[90,80,70,60,50,40,30,20,10,0]

EXEC @foo
result[0]: hal.buffer_view
10xf32=90 81 72 63 54 45 36 27 18 9
```

### A first `reduction` example: summing a 1D array

This function takes a 1D array of floats and returns their sum. `tensor<f32>`
is a 0-dimensional tensor type. We could as well extract the single `f32`
element and return that, but we wanted to make this example as simple as
possible.

What's subtle here is how the `bb0` block in the `linalg.generic` now actively
uses the `%result_entry` as an operand to `arith.addf`, yielding the result of
this addition on every iteration. Implicitly, this stores the result of that
addition to the destination, from where it is re-loaded on the next iteration
again as `%result_entry`. So the SSA value `%result_entry` has a different
value on each iteration.

Because the values from the `outs` parameter are now actually used, we can't
directly pass there the `tensor.empty`, whose elements are uninitialized. We
have to initialize the result entries as zeroes, which is achieved by the
`linalg.fill`.

```mlir
#map_1d_identity = affine_map<(m) -> (m)>
#map_1d_proj_0d = affine_map<(m) -> ()>

func.func @foo(
      %input : tensor<10xf32>) -> tensor<f32> {
  %result_empty = tensor.empty() : tensor<f32>
  %cst_0 = arith.constant 0.0 : f32
  %result_init = linalg.fill ins(%cst_0 : f32) outs(%result_empty : tensor<f32>) -> tensor<f32>
  %result = linalg.generic {
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_1d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_1d_proj_0d
    ],
    iterator_types=["reduction"]
  } ins(%input : tensor<10xf32>)
    outs(%result_init : tensor<f32>)
  {
    ^bb0(%input_entry : f32, %result_entry : f32):
      %add = arith.addf %input_entry, %result_entry : f32
      linalg.yield %add : f32
  }
  -> tensor<f32>

  return %result : tensor<f32>
}
```

```console
$ iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
$ iree-run-module --module=/tmp/prog.vmfb --input=10xf32=[0,1,2,3,4,5,6,7,8,9]

EXEC @foo
result[0]: hal.buffer_view
f32=45
```

### Combining `parallel` and `reduction` iterators: summing each row of a 2D array.

This is our first 2D example so for the first time we have to start explaining
how the `iterator_types` are enumerated and we start seeing some more
interesting examples of `affine_map`.

```mlir
#map_2d_identity = affine_map<(m, n) -> (m, n)>
#map_2d_proj_first = affine_map<(m, n) -> (m)>

func.func @foo(
      %input : tensor<3x5xf32>) -> tensor<3xf32> {
  %result_empty = tensor.empty() : tensor<3xf32>
  %cst_0 = arith.constant 0.0 : f32
  %result_init = linalg.fill ins(%cst_0 : f32) outs(%result_empty : tensor<3xf32>) -> tensor<3xf32>
  %result = linalg.generic {
    indexing_maps=[
      // Indexing maps for the parameters listed in `ins(...)`
      #map_2d_identity,
      // Indexing maps for the parameters listed in `outs(...)`
      #map_2d_proj_first
    ],
    iterator_types=[
      // Rule: the i-th iterator_type corresponds to the i-th coordinate in the
      // source space of the affine_maps defined above, (m, n). So:
      "parallel",  // This refers to the `m` coordinate in the affine-maps.
                   // This is the coordinate that is preserved in the result,
                   // see the map_2d_proj_first map given above.
      "reduction" // This refers to the `n` coordinate in the affine-maps.
                  // This is the coordinate that is dropped by the map_2d_proj_first
                  // given above and thus not present in the 1D result.
    ]
  } ins(%input : tensor<3x5xf32>)
    outs(%result_init : tensor<3xf32>)
  {
    ^bb0(%input_entry : f32, %result_entry : f32):
      %add = arith.addf %input_entry, %result_entry : f32
      linalg.yield %add : f32
  }
  -> tensor<3xf32>

  return %result : tensor<3xf32>
}
```

```console
$ iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
$ iree-run-module --module=/tmp/prog.vmfb \
  --input=3x5xf32=[[0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]]

EXEC @foo
result[0]: hal.buffer_view
3xf32=10 35 60
```

## Matrix multiplication as a `linalg.matmul` and as a `linalg.generic`

We are now ready to see how to express matrix multiplication as a
`linalg.generic`. But actually, rather than just writing that by hand, we are
going to let Linalg do it for us. Indeed, in addition to `linalg.generic`,
Linalg contains a number of "named ops", which are essentially just short-hand
notation for special cases of `linalg.generic`. One of them is `linalg.matmul`,
doing matrix multiplication accumulating into an existing accumulator. Here is
a simple function performing a matrix-multiplication-with-accumulation using
`linalg.matmul`. Also in this example, we use dynamic shapes (the `?` in the
shapes, see the above section where we encountered that), but we could just as
well use static shapes.

```mlir
func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>, %acc: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %result = linalg.matmul
    ins(%lhs, %rhs: tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%acc: tensor<?x?xf32>)
  -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
```

```console
$ iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
$ iree-run-module --module=/tmp/prog.vmfb \
  --input=2x2xf32=[[1,2][3,4]] \
  --input=2x2xf32=[[1,4][3,2]] \
  --input=2x2xf32=[[0,0][0,0]]

EXEC @matmul_dynamic
result[0]: hal.buffer_view
2x2xf32=[7 8][15 20]
```

Now we encounter another IREE tool: `iree-opt`. Unlike `iree-compile` which
compiles a MLIR program all the way down to a `.vmfb` that's ready to run on the
target device, `iree-opt` only applies selected transformations.

We run:

```console
iree-opt --linalg-generalize-named-ops prog.mlir
```

And that prints:

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @foo(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
  }
}
```

So that's the `linalg.generic` implementing matrix multiplication equivalently
to the above `linalg.matmul` form. We can  compile and run that like the above
program and it will have exactly the same result.

Here the 3 listed `iterator_types`, `["parallel", "parallel", "reduction"]`,
correspond to the 3 listed coordinates in the `affine_map`'s, `(d0, d1, d2)`.
So, `d0` and `d1` are parallel dimensions and `d2` is the reduction dimension.
That's why the first two `affine_map`'s results involve `d2` (they are
respectively for the LHS `%arg0` and RHS `%arg1`) and the last `affine_map`'s
result only involves the parallel `d0` and `d1`, as it refers to the result
matrix.

!!! note

    Some current IREE compiler optimizations are only triggering on named ops
    like `linalg.matmul`, not on the equivalent `linalg.generic` form. Think of
    that as a non-essential current limitation, and the intent is over time to
    overcome these, but in the near term do use `linalg.matmul` when performance
    matters.

### Integer element types

MLIR defines integer types for absolutely any bit-width, including
non-power-of-two bit-widths, and in three signedness flavors:

* Signed integers, indicated by the letters `si`.
* Unsigned integers, indicated by the letters `ui`.
* Sign-less integers indicated by the letter `i`. "Sign-less" means that the
  integer type does not convey signedness; the integer value may be used as
  either a signed or an unsigned value but that's a property of the *operation*
  using that value as an operand, that's not encoded in the *type*.

So for instance, `si16` is the 16-bit signed integer type, `ui24` is the 24-bit
unsigned integer type, and `i8` is the sign-less 8-bit integer type.

Now here is a very important principle of how the MLIR dialects that are
relevant to us in IREE operate:

!!! note

    Only use sign-less types. Always encode signedness in operations, not in types.

For example, here is how we perform a matrix multiplication where the LHS is
signed 8-bit integers, the RHS is unsigned 8-bit integers, and the accumulator
is signed 32-bit integers. Notice how the fact that LHS is signed and the RHS is
unsigned is encoded only in the implementation of the `linalg.generic` basic
block, where the LHS and RHS entries are extended, respectively as signed
(`arith.extsi`) and unsigned (`arith.extui`):

```mlir
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @foo(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>, %acc: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %result = linalg.generic
      {indexing_maps = [#map, #map1, #map2],
       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%lhs, %rhs : tensor<?x?xi8>, tensor<?x?xi8>)
      outs(%acc : tensor<?x?xi32>) {
    ^bb0(%lhs_entry: i8, %rhs_entry: i8, %acc_entry: i32):
      %lhs_extended = arith.extsi %lhs_entry : i8 to i32
      %rhs_extended = arith.extui %rhs_entry : i8 to i32
      %mul = arith.muli %lhs_extended, %rhs_extended : i32
      %add = arith.addi %acc_entry, %mul : i32
      linalg.yield %add : i32
    } -> tensor<?x?xi32>
    return %result : tensor<?x?xi32>
  }
}
```

```console
$ iree-compile --iree-hal-target-backends=llvm-cpu prog.mlir -o /tmp/prog.vmfb
$ iree-run-module --module=/tmp/prog.vmfb \
  --input=2x2xi8=[[-1,-2][-3,-4]] \
  --input=2x2xi8=[[1,4][3,2]] \
  --input=2x2xi32=[[0,0][0,0]]

EXEC @foo
result[0]: hal.buffer_view
2x2xi32=[-7 -8][-15 -20]
```

!!! note

    A current runtime limitation,
    <https://github.com/iree-org/iree/issues/16241>,
    prevents passing sub-byte-bit-width integers on the `iree-run-module`
    command line.
