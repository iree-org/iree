# IREE TFLite C API Compatibility Shim

**EXPERIMENTAL**: we are working towards making this a stable API but it has a
ways to go still. Progress is being tracked in https://github.com/openxla/iree/projects/17.

Provides a (mostly) tflite-compatible API that allows loading compiled IREE
modules, managing tensors, and invoking functions with the same conventions as
[TFLite's C API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/c).

The intent is not to have a faithful reproduction of all tflite features as
IREE obviates the need for many of them (such as builtin ops). This is meant as
a way for applications that may currently be using the tflite API to quickly
onboard with IREE. We still expect applications to later migrate to proper IREE
APIs to gain the most from IREE as many of the features in IREE are not possible
to expse from simple single-shot invocation APIs like tflite.

## Quickstart

### Compiling TFLite FlatBuffers to IREE Modules

```sh
# TODO(#3970): example command line converting the fb (iree-compile-tflite).
```

**TODO(benvanik)**: tosa flow, iree-compile (whenever ready)

### Retargeting to the IREE TFLite Binding Library

```sh
# TODO(benvanik): example command line with include path and IREE static lib.
```

The bindings supply copies of the tflite include files that expose **only the
public API and portions that IREE supports**. All unsupported features are
guarded by a `IREE_BINDINGS_TFLITE_INCLUDE_UNSUPPORTED_APIS` define which can be
enabled if compatibility with code that may reference the calls or types is
required. Needing that is a strong indication, though, that the application is
not following the public API and will corrupt its memory touching random
structure fields.

Always **prefer static linkage** when possible: because the IREE runtime is so
small the overhead of linking it in its own library can double the size.
Instead, ensure you are linking it into your main application so the use of
things like the C runtime are amortized across both your code and IREE's. You
can force static linkage via the `TFL_COMPILE_LIBRARY` define (as with normal
tflite).

### Loading and Executing Models

```c
// TODO(benvanik): tflite usage example.
```

Use `TfLiteModelCreateFromFile` if loading your model from the filesystem so
that IREE can directly map it into memory. Even if the model is fetched
dynamically it is still worth it to write it to disk and then map it so that
the infrequently used mapped pages can be discarded by the OS to reclaim memory.
Only use `TfLiteModelCreate` if you have the model embedded in your binary or
are sure you can accept the wired pages for the convenience like in a
REPL/notebook. When used in conjunction with the IREE C API the compiler
arranges the module memory to be able to perform things like data prefetching
(for constants prior to when they are used), page eviction (for initializer
constants that won't be needed again), and when compiling for native targets the
ability to directly execute pages from the memory.

**TODO(benvanik)**: stock example

## Support

|     | Glossary
| --- | --------
|  ✔️  | Supported and expected to match tflite semantics
|  ⚠️  | Supported with compatibility caveats (avoid if possible)
|  🐢 | Supported with performance caveats (prefer the IREE C API)
|  🚫 | Unimplemented but supportable if needed
|  ⛔ | Unsupported and unlikely to ever be (see notes below)
|  🔒 | Not part of the tflite public API
|  ❔ | Unknown; not yet studied

### Op Coverage

**TODO(benvanik)**: note about TOSA, link to tflite->tosa documentation

### API

Only the public C API functions are supported. The contents of internal
structures like `TfLiteTensor` and `TfLiteContext` are undefined.

|     | TFLite API                                 | Notes
| --- | ------------------------------------------ | -----
|  ✔️  | `TfLiteVersion`                            | returns an IREE version string
|     |                                            |
|  🔒 | `TfLiteModel struct`                       | _implementation detail_
|  ✔️  | `TfLiteModelCreate`                        |
|  ✔️  | `TfLiteModelCreateFromFile`                |
|  ✔️  | `TfLiteModelDelete`                        |
|     |                                            |
|  🔒 | `TfLiteInterpreterOptions struct`          | _implementation detail_
|  ✔️  | `TfLiteInterpreterOptionsCreate`           |
|  ✔️  | `TfLiteInterpreterOptionsDelete`           |
|  🐢 | `TfLiteInterpreterOptionsSetNumThreads`    | interpreters will not share thread pools; see [external contexts](#-external-contexts)
|  ✔️  | `TfLiteInterpreterOptionsSetErrorReporter` |
|  ⛔ | `TfLiteInterpreterOptionsAddBuiltinOp`     | IREE's compiler generates code
|  🚫 | `TfLiteInterpreterOptionsAddCustomOp`      | [not yet implemented](#-custom-ops)
|  🚫 | `TfLiteInterpreterOptionsSetOpResolver`    | [not yet implemented](#-custom-ops)
|  ⚠️  | `TfLiteInterpreterOptionsAddDelegate`      | available but a no-op; [not needed in IREE](#-delegates)
|  ⚠️  | `TfLiteInterpreterOptionsSetUseNNAPI`      | available but a no-op; NNAPI not supported
|     |                                            |
|  🔒 | `TfLiteInterpreter struct`                 | _implementation detail_
|  ✔️  | `TfLiteInterpreterCreate`                  |
|  ✔️  | `TfLiteInterpreterCreateWithSelectedOps`   | alias to `TfLiteInterpreterCreate`
|  ✔️  | `TfLiteInterpreterDelete`                  |
|  ✔️  | `TfLiteInterpreterResetVariableTensors`    |
|  ✔️  | `TfLiteInterpreterGetInputTensorCount`     |
|  ✔️  | `TfLiteInterpreterGetInputTensor`          |
|  ✔️  | `TfLiteInterpreterResizeInputTensor`       |
|  ✔️  | `TfLiteInterpreterAllocateTensors`         |
|  ✔️  | `TfLiteInterpreterInvoke`                  |
|  ✔️  | `TfLiteInterpreterGetOutputTensorCount`    |
|  ✔️  | `TfLiteInterpreterGetOutputTensor`         |
|     |                                            |
|  🚫 | `TfLiteTensor struct`                      | currently opaque; could be exposed with caveats
|  ✔️  | `TfLiteTensorType`                         |
|  ✔️  | `TfLiteTensorNumDims`                      |
|  ✔️  | `TfLiteTensorDim`                          |
|  ✔️  | `TfLiteTensorByteSize`                     |
|  ✔️  | `TfLiteTensorData`                         |
|  ✔️  | `TfLiteTensorName`                         |
|  ✔️  | `TfLiteTensorQuantizationParams`           |
|  ✔️  | `TfLiteTensorCopyFromBuffer`               |
|  ✔️  | `TfLiteTensorCopyToBuffer`                 |

### Features

|     | TFLite Feature         | Notes
| --- | ---------------------- | ------
| 🔒  | Sparsity               | **API not public**; likely possible
| 🔒  | Complex Numbers        | **API not public**; likely possible
| 🔒  | External Contexts      | **API not public**; support possible but API inadequate for performance sensitive applications
| 🚫  | Custom Ops             | [not yet implemented](#-custom-ops); can be supported with performance caveats
| 🚫  | Dynamic Model Creation | [avoid doing this and use a compiler](#-dynamic-model-creation); almost all use cases besides specialized tools like REPLs can compile their models offline
| ⛔  | Delegates              | concept mismatch; [not needed in IREE](#-delegates) due to its hardware abstraction layer (HAL)
| ⛔  | TFLite Micro           | concept mismatch; [compilers are much better at this scale](#-tflite-micro)

#### 🧪 External Contexts

**CURRENTLY UNSUPPORTED**: tflite has
[experimental support](https://github.com/tensorflow/tensorflow/blob/4827424ac32433075bf1ec885aa4b38b1ede2d65/tensorflow/lite/c/common.h#L735-L743) for
["external contexts"](https://github.com/tensorflow/tensorflow/blob/4827424ac32433075bf1ec885aa4b38b1ede2d65/tensorflow/lite/c/common.h#L63-L89)
but they are not exposed via the public API yet.

Though it's possible to use multiple `TfLiteInterpreter` instances in the same
process in the real tflite it is strongly discouraged: each interpreter will create its own thread and memory pools and device handles to accelerators and
assume it owns all resources exclusively. The experimental external contexts
API is present to try to allow for something better than that and IREE would be
able to make use of it to the extent the feature allows.

But IREE is designed to fully support large constellations of models all running
concurrently and passing data between both each other and the application
efficiently pipelined cross-device and cross-process. Though external contexts
would allow IREE to at least share some resources such as the thread pool it
would still require applications to cooperatively schedule model execution to
ensure that predictable latencies and memory consumption would be fixed at the
sum of all models peak memory use regardless of scheduling.

When using more than one simultaneously loaded and execution model it is much
better to use the IREE C API instead.

#### 🤷🏿‍♂️ Custom Ops

**CURRENTLY UNSUPPORTED**: possible to implement if needed; it seems as if there
barely any usage of the custom op C API outside of Google though so we recommend
avoiding it for now. ([1](https://www.google.com/search?q=%22TfLiteInterpreterOptionsAddCustomOp%22), [2](https://www.google.com/search?q=%22TfLiteInterpreterOptionsSetOpResolver%22)).

Custom ops in tflite map to functions imported into compiled IREE modules.
The IREE tflite API shim could provide a wrapper implemented as an
[iree_vm_module_t](https://github.com/openxla/iree/blob/main/iree/vm/module.h)
that resolves and executes the functions as they are called by the VM. Having
real IREE modules, though, provides significant benefits in representation
such as the ability to have asynchronous custom behavior that interacts well
with hardware accelerators and IREE's concurrency and pipelining model. It also
allows for a majority of the kind of operations that previously would have
necessitated custom ops at runtime to instead be done in the compiler as MLIR
dialects and lowered right to native code, SPIR-V, or WebAssembly without the
need for expensive interop and opportunities for the compiler to tightly
optimize the custom behavior with the rest of the model.

Relevant **unsupported** APIs:

* [`TfLiteRegistration`](https://github.com/tensorflow/tensorflow/blob/4827424ac32433075bf1ec885aa4b38b1ede2d65/tensorflow/lite/c/common.h#L827-L884)
* [`TfLiteInterpreterOptionsAddCustomOp`](https://github.com/tensorflow/tensorflow/blob/4827424ac32433075bf1ec885aa4b38b1ede2d65/tensorflow/lite/c/c_api_experimental.h#L51-L68)
* [`TfLiteInterpreterOptionsSetOpResolver`](https://github.com/tensorflow/tensorflow/blob/4827424ac32433075bf1ec885aa4b38b1ede2d65/tensorflow/lite/c/c_api_experimental.h#L70-L91)

#### 🙅‍♀️ Dynamic Model Creation

**ACTIVELY DISCOURAGED**: IREE separates compilation and execution; for
nearly all models that are capable of running on tflite there is no benefit (and
often extreme downsides) to constructing them at runtime. As in any other domain
if you don't need a JIT you should *never* use a JIT. Dynamic model creation in
tflite is most frequently used to work around the lack of dynamism that has been
inherent in the tflite interchange format and interpreter, neither issues of
which IREE suffers from. Though it's possible to ship MLIR to target devices and
construct models on-demand there are almost no situations in which one should do
so beyond tools like REPLs and it is not a supported IREE use case.

Relevant **unsupported** APIs:

* [`TfLite*Params` structures](https://github.com/tensorflow/tensorflow/blob/2d03c32d6299935ea74083c943c8d727ff50d4c8/tensorflow%2Flite%2Fc%2Fbuiltin_op_data.h)

#### 🙅‍♀️ Delegates

**SUPERFLUOUS**: The concept of delegates - something that dissects a model at
runtime and attempts to slice off random parts of it to steal for itself - is
unnecessary in IREE. The extensible hardware abstraction layer (HAL) that IREE
provides achieves cross-device portability in a way that allows for predictable
high-performance heterogeneity and efficient co-optimization with the compiler.
The act of mutating a user's input graph happens in the compiler where
significantly more metadata and resources are available to optimize the model
and the the model is deployed as multi-architecture binaries containing one or
more formats of native machine code or low-level intermediate representations
like SPIR-V or WebAssembly. It's akin to transmitting JPEGs and WebPs to web
browsers and allowing for the client to select the appropriate decoder vs.
shipping source uncompressed PNGs and having to transcode them on the fly. The
problem of distribution for deployable IREE artifacts matches that of the kind
apps must already deal with:
[split APKs](https://developer.android.com/studio/build/configure-apk-splits),
[universal binaries](https://developer.apple.com/documentation/xcode/porting_your_macos_apps_to_apple_silicon?language=objc), etc
and it's easy to map anything you can do with IREE artifacts to that mental
model.

Relevant **unsupported** APIs:

* [`TfLiteDelegate`](https://github.com/tensorflow/tensorflow/blob/2d03c32d6299935ea74083c943c8d727ff50d4c8/tensorflow/lite/c/common.h#L919-L960)
* [`TfLiteInterpreterOptionsAddDelegate`](https://github.com/tensorflow/tensorflow/blob/2d03c32d6299935ea74083c943c8d727ff50d4c8/tensorflow/lite/c/c_api.h#L109-L117)

#### 🙅‍♀️ TFLite Micro

**ACTIVELY DISCOURAGED**: in situations where memory and compute are at a
premium one should always use an ahead-of-time compiler and not waste precious
resources (memory, power, thermals, etc) on things like
[memory planning](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc)
(especially when only static shapes are supported), data conversion like
[endianness swapping](https://github.com/tensorflow/tensorflow/blob/2cad9d750cadd825910b61351a731eb0e8031608/tensorflow/lite/micro/micro_interpreter.cc#L180-L214), executing
[unfused elementwise ops](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/add.cc), including any executable bytes for
[code-paths that will never be reached](https://github.com/tensorflow/tensorflow/blob/2cad9d750cadd825910b61351a731eb0e8031608/tensorflow/lite/micro/kernels/softmax.cc#L103-L129)
in your model, or perform a single superfluous
**[m](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/split.cc)
[e](https://github.com/tensorflow/tensorflow/blob/2cad9d750cadd825910b61351a731eb0e8031608/tensorflow/lite/micro/micro_interpreter.cc#L240-L260)
[m](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/reshape.cc)
[c](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/concatenation.cc)
[p](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/pad.cc)
[y](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/strided_slice.cc)**.

In it's most size-optimized form the IREE VM bytecode is lowered to C or LLVM IR
and compiled along with application code. All tensor operations are aggressively
fused to avoid the need for transient memory, runtime memory planning (even when
using dynamic shapes), or copies. Just as there's an expectation that compiler
toolchains perform basic optimizations to application code (dead code removal,
code deduplication, etc) so too should there be an expectation for ML models and
even more so in environments where every byte (and joule) matters.

IREE will likely never have a shim for the tflite micro API or the
tflite `TF_LITE_STATIC_MEMORY` mode; when operating at that scale the entire
solution needs to change.
