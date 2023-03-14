# Extension mechanisms

!!! Note
    Much of this describes provisions for extension within IREE but until
    the core of the system has settled little work will be done to fully
    flesh-out and document them in detail. A large majority of things that would
    make someone want to extend IREE can instead be accomplished much easier and
    performantly using native MLIR dialects that are then processed by the IREE
    compiler.

## Guidelines

IREE has a compiler and runtime separation, a multi-layered architecture, and
split between execution of "host code" that schedules compute-heavy work and
[SPMD](https://en.wikipedia.org/wiki/SPMD) "device code" that performs the bulk
of compute operations. Each axis has a different set of extension mechanisms
that can be used independently or combined.

### Extension philosophy

Organized below are some of the mechanisms IREE provides for extending the core
compiler and runtime and when they should(n't) be used. The goal of these
progressively lower-level extension mechanisms is to make it easier for users
to fall into [the pit of success](https://ricomariani.medium.com/the-pit-of-success-cfefc6cb64c8):

!!! quote

    "_a well-designed system makes it easy to do the right things and annoying (but not impossible) to do the wrong things._" - [Jeff Atwood](https://blog.codinghorror.com/falling-into-the-pit-of-success/)

The amount of engineering complexity for initial bring-up and maintenance
increases with each subsequently lower-level approach and it is best to start
from the top and exit as fast as possible: this is a choose-your-own-adventure
where you're trying to escape the dungeon with both the loot and your limbs
:dragon:. Avoid the temptation of immediately dropping down to making external
C calls at runtime because that's how it's been done before as it's easier,
more robust, and more performant to use the system as it is intended to be used.

### When to extend

**The primary goal when extending any framework should first be to avoid
extending it at all.** There is no mechanism that is free - whether in terms of
engineering effort to develop and maintain over time, include in compiler
deployments, or include in runtime deployments. As a system scales in
deployment configurations the available mechanisms for extension increase but
so too does the chaos introduced by extensions that do not also scale with that
design. Users are the only ones who can determine the tradeoffs they are
willing to accept: for example, the mechanism to extend device code with a
custom runtime call to a C function does not work on GPUs and gets
significantly more complicated on CPUs as sandboxes/enclaves are used - but if
the user scenario is for local process CPU-only execution that may not matter.

### Where to extend (inputs/compiler/runtime)

Consider in normal software development when one would choose to write more
code (possibly packaging it into a reusable library) vs. changing the
programming language or compiler they are using to compile their code vs.
changing the operating systems their code runs on. The further one gets from
the problem they are trying to solve the more work, coordination, and
maintenance is involved and though there are reasons to make changes across the
stack they should be done only when a simpler solution would not suffice.

An author will retain more control over their logic the closer they sit to the
inputs to the compiler. IREE provides several mechanisms that try to keep
control with the author and robust to changes in IREE or MLIR internals and it
is strongly encouraged that those looking to extend take those routes first.
Contributions that help everyone are very welcome but do have a higher cost and
it's often much easier to design and justify upstream changes with working
examples in forks or at higher levels of the stack.

### Where to extend (host/device)

From a performance perspective the rule is to colocate code with the data it is
acting on: tensor data, for example, should almost exclusively be manipulated by
device code as tensors live on device. Attempting to use tensor data with host
code will result in synchronization points and host/device transfers that can
decimate performance. This can lead to seemingly paradoxical situations where
swapping out compiler-generated code for a human-authored "fast path" can be
slower than even the most naive compiler results. An important thing to keep in
mind with compilers is that it is exceedingly difficult to produce code by hand
that is consistently more performant across a broad range of deployments and
_the first temptation should always be to improve the compiler_ - extending it
via other mechanisms when not required by the task is often just premature
optimization.

## 1. Target IREE input dialects

!!! tldr "TL;DR"

    Convert your custom ops into standard MLIR dialects.

``` text
+------------+      +--------+      +---------------+
| Your input | -+-> |  iree  | -+-> | IREE compiler |
+------------+  |   +--------+  |   +---------------+
                |   +--------+  |
                +-> | linalg | -+
                |   +--------+  |
                |      ....     |
```

The easiest, cleanest, and most robust path to extend IREE is to make use of
what MLIR is designed for: composing dialects and converting between them. IREE
supports several input dialects such as `tosa`, `mhlo`, `linalg`, and the
standard `arith`, `math`, `tensor`, and `scf` dialects. Any source IR that can
be turned into that mix of dialects (directly or transitively) will work with
the whole IREE pipeline for all deployment configurations and targets. If
possible to express the computation in this form it will always be the best
route to getting small deployments without the need to modify or include any
additional code at runtime and run on all device types and execution modes.

This mechanism can also be layered with any of the subsequent lower-level ones:
if some part of the operation runs on the host and some part on device then
decomposing it such that it contains as many standard ops for flow control as
possible and linear algebra/custom ops for the dense math will reduce the
engineering effort required on both sides and lead to an easier to maintain
solution even if lower-level extension is required.

A large majority of classic ML "custom ops" can be accomplished with this
approach. When bringing up projects built on IREE it's best to concisely
describe the operation in more elemental mathematical representations and then
add optimizations where required knowing that things will still work even if
those optimizations never happen.

### Pros

* No IREE compiler or runtime code changes required.
    * Can use standard IREE packaged releases and tools.
    * No versioning issues at runtime.
* IREE's host/device partitioning can partition your code.
* Fusion and other compiler techniques (CSE/DCE/inlining/etc) work on your code.
* All target backends (CPU/GPU/accelerators/enclaves/etc) work.

### Cons

* Input dialects cannot natively represent all possible programs (such as file
  IO and other syscalls).
* Performance-sensitive host code (b-trees and other in-memory databases) will
  run through the slower VM paths if not authored as dense compute.

### When to use

* :material-check: Targeting multiple MLIR toolchains of which IREE is just
  one (as little to no IREE-specific code is required).
* :material-check: Operation represents host code in addition to device code.
* :material-check: All code is known statically or symbolically at
  compile-time (instead of independently versioned libraries at runtime).
* :material-close: Complex high-performance code not representable as linear algebra.
* :material-close: External runtime interactions (file/network/user IO). Use
  custom modules.

### Implementation

To make use of this approach one just needs to follow the standard
[MLIR dialect conversion](https://mlir.llvm.org/docs/DialectConversion/)
behavior: add a dialect with ops, add a conversion pass, and run that pass
before providing the resulting IR to the IREE compiler. See
[Creating a Dialect](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/).

Think of this like authoring C++ sources with templates that you compile into
your application: Clang (and LLVM beyond) don't know about your library details
and instead just process it as it would any other code. You can take the same
source and pass it to GCC and it'll be robust to underlying changes in the
system.

## 2. Extend host code with custom modules

!!! tldr "TL;DR"

    Import MLIR functions in the compiler and custom modules at runtime.

```mlir
// Main user module compiled by IREE:
module @model {
  // Declare a synchronous external function:
  func.func private @my_custom_module.sync_func(%input: tensor<?xf32>) -> i32
  // Declare an asynchronous external function:
  func.func private @my_custom_module.async_func(%input: tensor<?xf32>) -> tensor<?xf32> attributes {
    iree.abi.model = "coarse-fences",
    nosideeffects
  }
  func.func @predict() {
    ...
    // Call a synchronous/blocking external function:
    %sync_result = call @my_custom_module.sync_func(%sync_input) : (tensor<?xf32>) -> i32
    ...
    ...
    // Call an asynchronous/non-blocking external function:
    %async_result = call @my_custom_module.async_func(%async_input) : (tensor<?xf32>) -> tensor<?xf32>
    ...
  }
}
```

IREE provides dynamic linking at runtime via its VM interfaces. For code that
runs on the host and requires syscalls or calling out to existing libraries -
such as file IO, text processing, and JPEG decoding - this is an easy way to
interop without paying attention to the more complex details of device code.
An IREE module compiled using custom modules is portable and dynamically
deployable so long as the custom module is registered at runtime.

This approach conceptually matches what normal native binaries do in an OS:
imports are declared and at runtime they are resolved based on the available
exports of modules in the system. Just as with normal systems engineering
design of the API between modules is up to the user and depending on rigor can
have several pitfalls but these problems and their solutions are not IREE
specific and anyone who has designed a shared library interface can apply the
same rules here in IREE around versioning, performance, etc. One does not add 2
integers via a syscall and the same holds here: custom modules and the functions
within should perform a large amount of work to hide overheads involved in the
cross-module calls and users must be aware that the compiler cannot optimize
across the call boundaries.

See the [synchronous tensor I/O](https://github.com/openxla/iree/tree/main/samples/custom_module/sync/)
and [asynchronous tensor I/O](https://github.com/openxla/iree/tree/main/samples/custom_module/async/)
samples.

### Pros

* No IREE compiler code changes required.
* Produced artifacts are portable across IREE deployment configurations.
* Full system access is allowed - the VM just calls external functions.
* Runtime modules can be implemented (via shims) in other languages/runtimes.

### Cons

* Custom modules must be registered at runtime by the user.
* The VM custom module ABI goo must be authored by the user (such as with JNI or
  pybind to move between java/python and C).
* All custom module code must be compiled and deployed regardless of how much
  any modules use. The granularity of modules and their versioning is up to the
  user.
* Custom module code cannot be optimized by the IREE compiler to avoid
  host/device readbacks and unnecessary data type conversion.

### When to use

* :material-check: Interactions with large libraries or system calls.
* :material-check: Performance-sensitive host code that cannot easily be
  represented as device code (like UTF-8 string transformation using libicu).
* :material-close: Extensively using tensor resources.

### Implementation

The runtime portion requires that the code be exported to the VM system by way
of an `iree_vm_module_t` interface. A low-level native interface exists with
minimal overhead and is used for example [by the IREE HAL itself](https://github.com/openxla/iree/tree/main/iree/modules/hal).
There is also a C++ wrapper that is significantly easier to work with however it
needs some performance improvements.

Full end-to-end examples can be found under [`samples/custom_modules/`](https://github.com/openxla/iree/tree/main/samples/custom_modules):

* The [basic](https://github.com/openxla/iree/tree/main/samples/custom_module/basic/)
sample shows how to add VM modules with custom types and take advantage of ABI
features like fallback functions and optional imports.
* The [synchronous tensor I/O](https://github.com/openxla/iree/tree/main/samples/custom_module/sync/)
sample shows a call taking and returning a tensor and performing blocking work.
* The [asynchronous tensor I/O](https://github.com/openxla/iree/tree/main/samples/custom_module/async/)
sample shows the same thing but with fences for asynchronous scheduling.

## 3. Extend target-specific device conversion patterns

!!! tldr "TL;DR"

    Add patterns to `iree/Compiler/Codegen/` to emit target code.

The easiest and most robust path for specializations of device code is to emit
such code mixed with the IREE compiler generated code at the highest possible
level of abstraction within the target pipeline. For example, if the code can be
represented with the `vector` dialect then inserting conversion patterns between
`linalg` and `vector` enables the emitted code to be specialized further based
on user configuration and optimized with the full set of available passes that
run in the pipeline. For each level lower one goes the more flexibility they
gain such as being able to emit inline assembly blocks that do anything while
trading off generality and multi-targeting applicability.

How much the tradeoff matters is based on the behavior of the extension.
If a pattern changing a transcendental function to an approximation can operate
at the vector level then all IREE deployment targets can benefit from the
pattern and as new targets are made available they will automatically receive
the benefits. In contrast, a pattern at the vector level that turns generic
vector operations into architecture-specific LLVM intrinsics by its nature only
pertains to a single target family and can be done at a lower level. As a rule
of thumb if a particular pattern is going to need ~N implementations for ~N
targets that are all mostly the same it's better to try to move that higher in
the stack.

At this point the complexity of extending things is still fairly constrained: a
C++ pass or pattern is verified with normal lit tests and can be upstreamed
easily either into MLIR or IREE (a large number of IREE patterns are upstreamed,
benefiting all users of MLIR). Cross-compilation and versioning are not a factor
and the IREE artifacts can be considered durable at a coarse level (outside of
major target architectural changes).

Note that depending on the target there are various mechanisms for representing
code in MLIR, up to including inline assembly snippets in IR via
[`llvm.inline_asm`](https://mlir.llvm.org/docs/Dialects/LLVM/#llvminline_asm-mlirllvminlineasmop).

### Pros

* Not limited to what is possible to represent in any particular MLIR dialect.
* Rich target configuration available; multiple passes can contribute info.
* Produced executable binaries are hermetic and no runtime changes are required.
* Specialization can happen in MLIR dialects like `linalg` or `vector` as well
  as target-specific representations like SPIR-V and LLVM IR.
* The compiler can perform deep optimizations across both the generated code and
  the provided code (hoisting/loop invariant code motion/cse/etc).

### Cons

* Requires implementing the patterns as code in the IREE compiler or via TBD
  interfaces.

### When to use

* :material-check: Code that must be emitted during target lowering - such as
  something optimizing for a particular CPU architecture.
* :material-check: Hot code mixed with generated code at a fine granularity
  (within the innermost loop).
* :material-close: External existing hand-authored libraries. Either statically
  or dynamically link instead.

### Implementation

There are several ways to author patterns and passes in MLIR. As examples:

* A majority of patterns are authored in C++ using [PatternRewriter](https://mlir.llvm.org/docs/PatternRewriter/).
* [PDL](https://mlir.llvm.org/docs/Dialects/PDLOps/) is an MLIR-based way to
  express rewrite operations with strong typing, compile-time verification, and
  easily-readable and less-verbose IR.
* `linalg` uses a [python-based DSL](https://mlir.llvm.org/docs/Dialects/Linalg/OpDSL/)
  for defining some of its extended ops.

There are many examples within both MLIR and IREE, one specifically being the
[polynomial approximation expansion patterns](https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/Math/Transforms/PolynomialApproximation.cpp).

## 4. Include external target-specific device code

!!! tldr "TL;DR"

    Statically link external object files into IREE executables.

For large bodies of existing device code or library calls that are available for
static linkage the work involved to reimplement them at higher levels of the
stack can be cost prohibitive even if it leads to better results. In these cases
just as with a normal toolchain one would just want to declare an external
function, call it, and add the object file to the linker command line. In IREE
the same can be performed by way of taking compatible bitcode or native object
files and linking them in with the generated code. An MLIR pattern would declare
and emit the call and the target-specific IREE linker would pull in the objects.

As the linking behavior varies per target (for example, some targets like
SPIR-V don't have traditional linkers) how this is performed is up to the IREE
target backends. The complexity involved in producing the object files to link
will also vary per-backend and the complexity of the deployment: cross-compiling
for multiple architectures or compilation modes (ASAN, etc) will require unique
copies of the object files matching that precise configuration.

At this point generality is largely out as is the ability to cleanly upstream
such files. It should be apparent how a few dozen lines of C++ or PDL that
avoids the need for any of this complexity is more appealing. In extremely
specific cases of a single platform/architecture/version for a single program
deployed via a specific artifact composition it's not so bad but IREE is
designed such that extreme specificity is an optional mode of the more general
solution. This does not mean this mechanism is not useful in some situations and
only that it should be a last-resort when one of the easier to manage solutions
is not viable - not a shortcut to avoid writing some C++ patterns.

### Pros

* Works with hand-authored code in compatible object files from any toolchain.
* No IREE runtime changes required.
    * All deployment modes still work, including multi-targeting.
    * No versioning concerns as custom code is included in artifacts.

### Cons

* Users must provide per-target precompiled object files on disk.
* IREE compiler changes are still needed for generating the external calls.
* Though LTO _may_ be able to optimize across the calls it is not guaranteed.

### When to use

* :material-check: Existing math libraries or architecture-specific functions
  that cannot be ported into a more MLIR-friendly form.
* :material-check: Mixing in hand-authored code written in C/rust/etc with
  generated code from MLIR.
* :material-close: External code can be represented as either `linalg`,
  `vector`, or LLVM IR. Use target-specific conversion patterns instead.
* :material-close: External code size is large and unlikely to benefit from
  link-time optimizations (such as something like libjpeg). Dynamically link
  instead.

### Implementation

As the linking behavior varies per target backend there is no general solution
at this level: if targeting the CPU then the system native linker or lld need
to be provided the object files, while SPIR-V will need to merge the SPIR-V
binaries directly, and Metal shader libraries will need to be constructed with
the Apple-specific `metallib` tooling. Producing these files and performing the
linking is outside the scope of IREE.

If the files can be acquired then compiler changes will be required to emit
calls to them and invoke the linker with the the files.

On the CPU an alternative is to use the static library output mode where IREE
produces an object file and then the user invokes the linker themselves; this
still requires the compiler changes to emit the calls but avoids needing to
teach the compiler how to link the files.

## 5. Dynamically link target-specific device code (CPU only)

!!! tldr "TL;DR"

    Dynamically link external C functions at runtime from device code.

_It is pitch black. You are likely to be eaten by a grue._

This is the lowest-level integration in the system and is designed to act as an
escape hatch and - as with any emergency escape hatch - it's not designed for
ergonomics. Users should try first to come in through the door and attempting to
use this mechanism should trigger alarms about the approach being attempted.

IREE's execution model for device code and native machine binary deployment
mechanisms are designed with several constraints in order to make all of the
above approaches possible and performant. Calling arbitrary C functions from
deep within the system can introduce subtle (and not-so-subtle) bugs that are
extremely difficult to track down and versioning between the compiler emitting
the calls and the runtime providing the implementations can cause skew unless
held carefully. Consider the methods added here like syscalls in that they must
be extremely focused and if they are ever likely to change (including being
removed) then care will be needed just as with versioning or redirecting a
syscall. Designing good stable interfaces is hard and a classic pit of failure.

Some things to note:

* Device code executes in a tiled fashion and single dispatches may invoke the
  same function many times from many threads concurrently to perform
  the larger work.
* Tiles may execute in any order and on any thread; performing fine-grained
  locking within the tile can lead to deadlocks.
* Device code is stateless in order to allow for access restrictions and caching
  across multiple loaded models - any library state required must be externally
  managed via process globals.
* Device code may be running out-of-process (sandbox/enclave) and the library
  functions must be available where the dispatches run and not where they are
  launched (such as being linked into the sandbox binary, if separate from the
  main process binary).
* The stack must be used to pass arguments/results to external calls via a
  single pointer and there is no libffi-like functionality for magically calling
  arbitrary C functions. Users must provide the shims they need.
* Thread-local storage is unavailable in the called code (it may be usable, but
  it is not guaranteed it'll work on all platforms and leaks are likely).
* No heap allocator is provided and the use of libc malloc is unsupported.

Most of the constraints here come from the [SPMD](https://en.wikipedia.org/wiki/SPMD)
parallelism model, platform-agnostic deployment format, and overall
data-oriented design of IREE. Code operating in this fashion has a certain shape
and that is usually not the same as big legacy single-threaded CPU-focused BLAS
libraries that perform their own caching, internal thread and state management,
and other shenanigans. IREE is not designed to wrap such things and if any of
these notes are issues it is more an indicator that the approach needs
adjustment than anything else. Trying to bypass or workaround the constraints is
possible - after all IREE is an open source project and any user is welcome to
fork it - but unsupported by the core IREE team.

### Pros

* Function resolution at runtime is orthogonal to compiler target specification.
* Machine code can be shared between the application and IREE artifacts.

### Cons

* IREE compiler and runtime must both be modified.
* Deeper integration with the IREE codegen compiler infrastructure required.
* ABI versioning complexity between compiler and runtime.
* Runtimes must ship the imports for the lifetime of any artifact compiled to
  use them.
    * Humans are bad at predicting the future.
    * Using the same artifact in different binaries at runtime requires changes
      to each binary - including those that may not be owned by the person
      producing the artifact.
    * Weak imports and conditional usage can help but still leads to bloat.

### When to use

* :material-check: Calling into opaque closed-source BLAS-like microkernel
  libraries.
* :material-close: Any other cases covered above can be used, especially
  microkernels that can be represented in MLIR or as statically linked
  libraries.

### Implementation

The compiler is changed to produce calls to imports via a dynamic import table
provided to each dispatch function. The import table is declared in the
executable library for use at runtime. Runtime applications register an import
provider to resolve named symbols in the import table to C functions that
marshal arguments and results.

The compiler-side needs some additional work but an example is included here:
[Issue 7504](https://github.com/openxla/iree/issues/7504).
The runtime-side is complete and resolution is performed by a user-supplied
`iree_hal_executable_import_provider_t`.
