# Design roadmap

A not-so-concise walkthrough of various IREE features that are in the design
process and planned for future versions. A lot of the questions around how the
IREE IR is designed and why certain components exist (such as the VM) hopefully
become much clearer when seeing where we want to go with the infrastructure we
are building (as opposed to where we currently are with our MVP slice). This
document is not meant to encompass the entire design of any individual feature
and if there's interest please say hi on the
[iree-discuss](https://groups.google.com/forum/#!forum/iree-discuss) mailing
list.

[TOC]

## Input Dialects

### Quantization

It's assumed that any work related to quantization/compression has happened
prior to lowering into IREE dialects. Our plan is to use the proposed
[Quantization Transforms](https://llvm.discourse.group/t/rfc-a-proposal-for-implementing-quantization-transformations-in-mlir/655)
to achieve both training and inference-time quantization of types in a way that
preserves maximum accuracy. IREE will support running with original unquantized
floats in all cases, allowing for a smooth on-ramp to quantization and the gains
in performance and reduction in model size that come from it.

As future work IREE would like to move beyond these transformation-directed
approaches to quantization and interface directly to frontends which have a
defined enough type system to represent accurate quantized (and otherwise
compressed) computations directly, not relying exclusively on compiler-side type
inference transforms.

## `flow`: Data- and Execution-Flow Modeling

The `flow` dialect is designed to allow us to extract as much concurrency as
possible from a program and partition IR into the scheduling and execution
domains. Today we have the IR structure and transformation flow in place but
have not yet got to the most interesting things such an infrastructure enables.
A majority of the largest performance, latency, and memory usage improvements
IREE can offer are determined first here and all following lowerings benefit.
_The fastest code is the code you don't execute and the smallest allocation is
the allocation you don't make_ ;)

### Avoiding Readbacks with `flow.stream`

A majority of the readbacks we have today (manifested as `flow.tensor.load.*`
ops) will be removed when we have an
[HLO tensor->primitive conversion](#xla-hlo-tensor-to-primitive-conversion).
There will still be cases when readbacks are required for correctness but they
usually fall into a small set of usage patterns. For those that don't this is
one place where IREE will warn about performance issues, allowing programs that
perform suboptimally but encouraging authors to adjust their input model to
enable better behavior. The IREE VM also has specific support for hiding
readback latency in an efficient way via
[coroutines](#coroutines-for-batching-and-cooperative-scheduling).

The most common case we are currently seeing in the IR is that of dynamic copies
where the offsets are dependent on the result of previous computations. Source
models may have top-k + gather operations, for example. These appear as a
`flow.stream`, a `flow.tensor.load`, and then another `flow.stream` that uses
the loaded value for a `flow.tensor.update` (or other operation):

```mlir
%index_tensor = flow.ex.stream.fragment(...) -> tensor<i32> { ... }
%index = flow.tensor.load %index_tensor : tensor<i32>
%result = flow.ex.stream.fragment(%arg0 = %index : i32, ...) -> ... {
  %0 = flow.dispatch ...
  %1 = flow.tensor.update %0, %arg2[%index] : tensor<10xf32> -> tensor<1x10xf32>
  ...
}
```

Today the `flow.tensor.update` turns into HAL command buffer transfer operations
that must have their offsets known at recording time. This is a limitation of
`vkCmdCopyBuffer` but not a fundamental limitation of any hardware. In fact
several drivers implement copies as small built-in shader programs meaning that
we could perform the same expansion here with the right primitives. This would
allow, in the above example, both the index to be computed and the tensor to be
updated within the same stream to entirely remove the host round-trip.

### Threading `flow.stream` through the CFG

The current `flow.ex.stream.fragment`, as denoted by the `ex`perimental tag, is
a temporary implementation designed to get the concept of streams lowered to the
HAL dialect. For streams to be effective at modeling larger concurrency scopes
they need to be able to move across branches in the CFG. This intuitively
follows exactly what one would do if recording commands in C:

```c++
vkCmdCopyBuffer(cmd, ...);
if (some_flag) {
  vkCmdBindPipeline(cmd, ..., pipeline_a);
} else {
  vkCmdBindPipeline(cmd, ..., pipeline_b);
}
vkCmdDispatch(cmd, ...);
```

The corresponding `flow` IR:

```mlir
  flow.stream.append[%s0](...) {
    flow.tensor.update ...
  }
  %b = arith.cmpi ne %some_flag, ...
  cond_br %b, ^a(%s0), ^b(%s0)
^a(%s1):
  flow.stream.append[%s1](...) {
    flow.dispatch @pipeline_a, ...
  }
  br ^end(%s1)
^b(%s2):
  flow.stream.append[%s2](...) {
    flow.dispatch @pipeline_b, ...
  }
  br ^end(%s2)
^end(%s3):
  ...
```

This allows the entire stream to be lowered into one command buffer without the
need for any host round-trips. The conversion into the `flow` dialect will walk
the CFG and attempt to thread the `flow.stream` values through so long as there
are no external dependencies.

### Predication of `flow.dispatch`

While the
[`flow.stream` threading through the CFG](#threading-flowstream-through-the-cfg)
can remove many of the simpler conditional dispatches there will always be some
that will have their execution dependent on the result of prior dispatches. For
these a `flow.cond_dispatch` will allow a condition to be provided that must be
true for the dispatch to actually be performed.

For targets that natively support predication in their command buffers (such as
D3D12's
[ID3D12GraphicsCommandList::SetPredication](https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-setpredication))
this provides a host round-trip-free way of conditionally executing dispatches
and transfers. Unfortunately Vulkan support is still lacking, though Nvidia
supports the
[VK_EXT_conditional_rendering](https://www.saschawillems.de/blog/2018/09/05/vulkan-conditional-rendering/)
extension that exposes the same behavior.

For targets that do not support predication natively it's still possible to
emulate predication with
[indirect dispatches](https://github.com/gpuweb/gpuweb/issues/31). In this model
the workgroup counts normally used to dispatch execution are sourced from
another device buffer at the time the dispatch is made instead of sourced from
the command buffer at the time the dispatch is recorded. Degenerate dispatches
with counts of `0, 0, 0` allow for effective neutering of the dispatch with
minimal overhead (vs. the significant penalty of a host round-trip!).

By modeling such predication at the `flow` level we are able to lower into the
HAL with target-aware predication semantics and fuse indirect dispatch workgroup
count calculations into existing dispatches already being performed such that
overhead is reduced.

### Deduping `flow.executable`s

While still in the `flow` dialect, the executables are target-agnostic. This
makes simple IR tree diffing a potential solution to deduplication. Since most
of the dispatches originate from the same source-language library calls in input
frameworks there's a high likelihood of duplication, and depending on when
inlining is performed we may have stronger or weaker ability to perform the
deduplication. Thanks to the MLIR canonicalization pass (that ensures ops are
rearranged into consistent canonical representations) the IR comparisons can be
done rather trivially.

### Rematerializing CSE'd Expressions

Common subexpression elimination is performed many times during lowering,
however there comes a point where the CSE can introduce false dependencies and
additional allocations that are otherwise avoidable. For example if a
broadcasting operation is CSE'd and then the result is used by two or more
operations that are scheduled independently what would have been a relatively
cheap lowering of the broadcast to a simple index remapping now becomes an
additional dispatch, materialization of an intermediate tensor, and a barrier:

```mlir
%bcast = "mhlo.broadcast_in_dim"(%cst) : (tensor<f32>) -> tensor<1024x10xf32>
%mul1 = mhlo.multiply %arg0, %bcast : tensor<1024x10xf32>
// (pretend something here that prevents fusion)
%mul2 = mhlo.multiply %arg1, %bcast : tensor<1024x10xf32>
```

```mlir
%bcast = flow.dispatch.region(%cst : tensor<f32>) -> tensor<1024x10xf32> {
  %0 = "mhlo.broadcast_in_dim"(%cst) : (tensor<f32>) -> tensor<1024x10xf32>
  return %0 : tensor<1024x10xf32>
}
// a barrier will be required here
%mul1 = flow.dispatch.region(%arg0 : tensor<1024x10xf32>, %bcast : tensor<1024x10xf32>) -> tensor<1024x10xf32> {
  %1 = mhlo.multiply %arg0, %bcast : tensor<1024x10xf32>
  return %1 : tensor<1024x10xf32>
}
%mul2 = flow.dispatch.region(%arg1 : tensor<1024x10xf32>, %bcast : tensor<1024x10xf32>) -> tensor<1024x10xf32> {
  %2 = mhlo.multiply %arg1, %bcast : tensor<1024x10xf32>
  return %2 : tensor<1024x10xf32>
}
```

Instead the broadcast should be rematerialized inside of both dispatch regions
as the cost of doing so is significantly less in compute resources and then the
intermediate tensor will not be required at all. Though at first it may seem
counter-intuitive to undo such a critical optimization as CSE (both to code size
and often to compute) but here it's something we must carefully balance while
looking at the whole system. It gets even more important when considering
multi-device execution as the cost of sharing memory and synchronizing may be
extremely non-trivial.

### Device Placement

While still within the `flow` dialect we have the ability to easily split
streams and safely shuffle around operations. Target execution backends can opt
into such behavior to ensure that device restrictions such as maximum in-flight
memory, maximum scheduling depth, and capabilities are observed. For
heterogeneous configurations the intent is that certain operations, dispatches,
and streams can be attributed to specify which device categories they should be
lowered. The constraint solving that takes place can be provided with generic
heuristics ("big GEMMs go on the accelerator"), profile-guided databases based
on benchmarks, learned traits via ML, etc.

## `hal`: Hardware Abstraction Layer and Multi-Architecture Executables

As the IREE HAL is designed almost 1:1 with a compute-only Vulkan API many of
the techniques classically used in real-time graphics apply. The benefit we have
by modeling our usage of such a low-level API in IR is that the normal work -
some of which is very non-trivial - for managing allocations, tracking resource
lifetime, and ensuring proper synchronization/barriers is something we can apply
the full force of an offline compiler against.

### Allow Targets to Specify `hal.interface`s

The `hal.interface` op specifies the ABI between the scheduler and the device
containing the buffer bindings and additional non-buffer data (parameters,
shapes, specialization flags, etc). Today a naïve ordering is used uniformly for
all targets however it is possible for target backends to opt into providing
their own interfaces based on target configuration. The same `hal.executable`
may have multiple interfaces and the same backend may use one or more. This is
useful for when target capabilities may vary at runtime, such as the
[number of available storage buffer bindings](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPerStageDescriptorStorageBuffers&platform=android)
in Vulkan. By exposing a few `hal.interface` variants with different binding
amounts the Vulkan backend could make better use of the larger number of
bindings available at runtime while still providing support for smaller
configurations.

Once we have multiple `hal.interface`s defined for executables the scheduler
needs to emit HAL ops that properly switch between them. By having a canonical
form for bindings we can ensure that only the differences between the interfaces
will need additional code.

### Target-specific Scheduling Specialization

Though the `flow` dialect attempts to fuse as many ops as possible into dispatch
regions, it's not always possible for all target backends to schedule a region
as a single dispatch. A classic example is algorithms like
[parallel reduction](https://en.wikipedia.org/wiki/Reduction_Operator#PRAM-algorithm)
commonly used on GPUs that may require many dispatches to identical executables,
while other algorithms may vary the executables they use based on the input
parameters such as shape or the target runtime device support.

By default the `flow.dispatch` executable translation to `hal.executable`s is
performed 1:1 and it is assumed that a single dispatch is required. Extending
target backends with scheduling interfaces (enabling them to opt into different
scheduling behavior) will allow the backends to emit any number of
`hal.executable`s and any stream commands (such as additional dispatches or
transfers) they may need. This is effectively equivalent to what would be done
at runtime only because we are still operating on IR prior to buffer allocation
and can use the `hal` ringbuffer primitive. Through this we can elide many of
the allocations that would otherwise be required at runtime (and the
concurrency-limiting false dependencies that usually come along with scratch
memory).

Since the algorithm used may vary based on the parameters of the dispatch (such
as the shape of the reduction which may be dynamically determined) scheduling
specialization may occur even when targeting a single backend. In many cases
folding and canonicalization can eliminate the overhead as whether one
dynamically computed workgroup size is used instead of another the same IR is
present.

### Buffer Usage Tracking

Many explicit hardware APIs require knowing how buffers are used alongside with
where they should be located. For example this additional information determines
caching policy on buffer accesses (write-through, write-back, etc), visibility
of writes across compute units, and the possible MMU properties that may need to
be maintained/matched for the buffer. By using the SSA-form value-semantics of
the MLIR `tensor` as used in the `flow` dialect we have complete information of
where buffers may be used or at least where they enter or leave regions where we
can derive such information.

Analysis passes can run over IR to attribute tensors such that when allocation
is performed when lowering to the `hal` dialect we do so from an allocator
compatible with where the buffer will be used, with memory types chosen based on
the potential cost and location of operations performed (write-only on host vs.
read-write on host and device, etc), and with usage bits indicating what kind of
operations may be performed on the buffer. Many of these are local
transformations as most buffers are only live within very small regions such as
the `flow.stream` encompassing their usage.

Traditional systems need to either use very permissive buffer properties or
heuristics that can introduce additional non-trivial overhead when such
heuristics are incorrect. For example,
[OpenGL had several such usage hints](https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glBufferData.xhtml)
that drivers were then able to use but almost no drivers behaved as desired in
all cases and it lead to additional memory ghosting, copies, readbacks, and
unpredictable performance. For almost all uses of the buffers within an IREE
invocation we instead can know precisely where and how buffers may need to be
moved and do it a minimum number of times if it is required.

### Batched Executable Caching and Precompilation

For targets that may require runtime preprocessing of their executables prior to
dispatch, such as SPIR-V or MSL, the IREE HAL provides a caching and batch
compilation mechanism based on Vulkan's
[Pipeline Cache](https://vulkan.lunarg.com/doc/view/1.0.26.0/linux/vkspec.chunked/ch09s06.html).

Today each executable is compiled on-demand and cached only for the process
lifetime. Though some drivers may provide their own caching we can make better
use of the explicit caching and compilation behavior with the additional
information we have in the compiler.

For any given entry point (or group of entry points) into an IREE module we can
perform reachability analysis to know which executables may be executed when
that entry point is invoked. In this way we can emit pre-invocation compilation
checks (similar to an `std::call_once` block) that provides all required
executables for compilation and allows more efficient compilation through
multithreading the compiler invocations. These same compilation caching function
can be exposed and invoked manually by an application to force pre-compilation
when it is least likely to impact the user, such as a post-install/first-run
step or concurrently while other application features are loading.

We can use zero or more scoped caches for executables within a module.
Completely dynamic modules (such as those emitted in eager-mode usage) may avoid
the caching overhead entirely, while modules that have several primary usage
modes (such as training and inference) may choose to use independent caches for
each such mode.

The caches generated can then be retrieved and saved by the hosting application.
Upon the next execution the application can provide the caches and if still
valid they will be used to avoid compilation.

### Target-aware Executable Compression

An advantage of representing executable binaries in IR after translation is that
we can apply various post-compilation compression and minification techniques
while still know precisely where the executable will be used. This is extremely
important for SPIR-V as it is not designed to be a small at-rest format. Though
the biggest lever we have to control generated code size is higher-level
deduplication and specialization there will still be a sufficiently large number
of executable binaries we will need to embed within the final modules and having
targeted approaches for reducing their size beyond just "gzip everything" is
very powerful.

For example, [SMOL-V](https://github.com/aras-p/smol-v) is a fantastic lossless
SPIR-V compression technique that, when coupled with modern dictionary-based
compression algorithms, can save significant binary size. As a data point, the
SPIR-V corpus SMOL-V uses for testing goes from 4.8MiB of raw SPIR-V to 348KiB
of compressed SMOL-V.

Combined with
[Batched Executable Caching and Precompilation](#batched-executable-caching-and-precompilation)
we can easily use shared dictionaries and other cross-artifact compression in a
relatively plug-in way.

### Target-aware Constant Compression

It's still an area that needs more research but one goal of the IREE design was
to enable efficient target- and context-aware compression of large constants
(typically model weights/parameters/embeddings). This may mean reusing existing
hardware compression formats on GPUs, ML accelerator-specific formats, or
very-low-bit-depth (1-4 bit per value) quantization techniques that cannot be
directly used without first decompressing. The inspiration here is formats like
[Crunch](https://github.com/BinomialLLC/crunch) and
[Basis Universal](https://github.com/BinomialLLC/basis_universal) that perform
["supercompression"](http://gamma.cs.unc.edu/GST/gst.pdf), and we may even be
able to use these directly as then we can make use of GPU hardware samplers to
do the 4-bit to 32-bit decompression, etc.

### Command Buffer Stateful Deduplication

The IREE HAL - much like Vulkan it is based on - eschews much of the state that
traditional APIs have in favor of (mostly) immutable state objects (pipeline
layouts, pipeline states, descriptor sets, etc). There are still a few stateful
entry points in the API, though, and deduplicating or reordering redundant calls
can reduce both IR, API, and execution overhead.

The key place this will have the largest impact is around descriptor set
bindings and push descriptors, both of which are state and can have non-trivial
setup overhead. A canonicalization for such commands that inspects the target
`hal.command_buffer` to see if the same state was set prior and code motion to
move such commands out of loop bodies when possible would be helpful.

### Resource Timeline

A core concept of the IREE scheduler that allows for overlapping in-flight
invocations is that of the resource timeline. This identifies module state that
can be in use by multiple invocations and assigns timeline milestones denoting
when the resource will be in the appropriate state for the current invocation to
proceed. Conceptually it is like a epoch-based synchronization mechanism as
commonly found in garbage collectors to allow for lock-free asynchronous memory
reclamation.

The advantage we have in the IR is that we know both the usage of all resources
thanks to [buffer usage tracking](#buffer-usage-tracking) and the
synchronization domains of all resources (in most cases). This allows us to
effectively assign one timeline semaphore per writeable resource while in
practice having far fewer than 1:1, as for example if two resources are only
ever written in the same command buffer only one semaphore is needed to signal
the completion of both writes.

By transforming IR to sink all resource reads and writes closest to where the
value is used we can enlarge the time windows that can overlap across
invocations that may share those resources. This is similar to what out-of-order
CPUs do with register renaming/reorder buffers/etc and something we can apply
some traditional instruction scheduling techniques to (only here our
'instructions' are entire command buffer dispatches/transfers).

Two degenerate cases of this approach are that of resource indirection
(`util.ptr<tensor<T>>`) and dynamic resource shapes. In these two cases it may
not be possible to continue recording commands even if we are able to ensure
execution is appropriately synchronized. This is where indirect dispatch,
[predication](#predication-of-flowdispatch),
[indirect command buffers](#indirect-command-bufferon-accelerator-execution),
and [VM coroutines](#coroutines-for-batching-and-cooperative-scheduling) can all
help cover for the times where we are unable to transform away the indirection
or emit shape logic without data dependencies.

### Transient Tensor Ringbuffer

(When properly implemented) almost all buffers required during execution never
escape the command buffers they are used in or a single VM invocation. We can
trivially identify this from the explicit captures of `flow.stream` and
`flow.dispatch` ops and the fact that all tensor types have value-semantics.
Only those tensor values loaded-from/stored-to module state or that cross the
exported module function boundary need special consideration while almost
everything else can live transiently only so long as it is required during
execution.

Thanks to this information about buffer usage and lifetime we can use a
[ringbuffer](https://en.wikipedia.org/wiki/Circular_buffer) to store the
transient tensor data and other required data reservations such as uniform
buffers used to pass dynamic parameters (shapes, flags, etc) into dispatches.
This gives the compiler and the application a knob that allows them to control
maximum concurrency (by having a very large ringbuffer) or maximum memory usage
(by having a minimally small ringbuffer).

Allocating tensors from the ringbuffer does not require sophisticated runtime
packing as we can emit IR to calculate required sizes for dynamically shaped
tensors. Whether a basic block reserves `%sz = arith.constant 42 : index` bytes
or `%sz = std.muli %cst, %dyn_dim : index` bytes doesn't materially change how
the allocations are performed. Since almost all usage involves simple write head
bumps there is no need for ahead-of-time memory planning or large fixed
allocations, and since no buffer within the ringbuffer can alias we can have
coarse (*read: low overhead*) guarantees about the availability of certain
regions of the ringbuffer (*"when this event is signaled all prior ringbuffer
writes have completed"*).

Usually any planning we may want to perform can be done in IR via code motion.
For example applying traditional algorithms used to reduce register pressure
will help us attain narrower live windows within the ringbuffer leading to a
larger number of in-flight operations for the same ringbuffer memory usage.

We may end up using both a classical ringbuffer and a variant known as the
[bip buffer](https://www.codeproject.com/Articles/3479/The-Bip-Buffer-The-Circular-Buffer-with-a-Twist)
because it is better for descriptor set utilization (as we can provide many
dispatch parameters with a single base offset bound once at the beginning of a
region).

### Timeline Semaphores on the Module ABI

Functions calls made across modules (either from C++ into the VM, VM->VM, or
VM->C++) should be able to define timeline semaphores used to wait and signal on
the call. We can do this by making all exports automatically have the semaphores
and then make invocations populate them if they were not provided by the caller.
In this way we can allow multiple invocations of exported functions to chain
naturally with internal asynchronous workloads, turning most IREE invocations
into just recording of command buffers that can never block.

When combined with
[VM coroutine support](#coroutines-for-batching-and-cooperative-scheduling) we
even have the ability to interleave any required host execution between the wait
and signal semaphores provided such that the caller never knows on which device
execution is taking place. It's still possible to provide synchronous wrappers
that emulate blocking behavior but by having the core system designed around a
single system-supported primitive we avoid the need for additional things like
interrupt watchdog threads, implicit blocking, and other pitfalls.

### GPU-like CPU Scheduling

One approach to using multiple cores on a CPU is to perform interior
parallelization of operations such as OpenMP or library-call-based custom thread
pools (gemmlowp). This works when each individual operation is relatively costly
vs. potential pipeline bubbles caused by work spinning down near the end of an
operation and spinning up at the beginning of the next.

IREE is designed to handle many more workloads - some of which have very narrow
shapes but very deep pipelines (like search algorithms) - such that the above
approach of multithreading within ops becomes a bottleneck. These workloads are
traditionally very poorly handled by frameworks and issues with
oversubscription, pipeline stalls, and suboptimal system schedulers (such as on
Android) can lead to more time being spent thrashing about than actually
executing real work.

The approach we take here is to treat the cores of a CPU as if they were
computation units on a GPU, each able to perform some set of heterogeneous work
independent of others units. This means that the concurrency we are trying to
model at the `flow` level and communicate to the runtime via the `hal` that
explicitly states which dispatches can overlap and the size of the workgroups
can trivially be used to distribute this work over many cores exactly as a GPU
would do it. Integration with library calls that may require their own threading
(such as Ruy) requires that they be able to use the IREE thread pool instead of
their own.

In this way we can avoid pipeline bubbles and other latency-inducing
unpredictable scheduling. This does not mean that we treat individual units of
work at the same scale as we would for GPUs, but instead that we tile and have
one or more processing units that allows us to work on those tiles. Whether the
tile size is defined by a library call contract, heuristics, or empirically is
TBD, but expect workgroup sizes in the thousands to millions of invocations vs.
normal GPU workgroup sizes in the dozens to hundreds of invocations.

To achieve this style of scheduling efficiently we'll likely use something like
[marl](https://github.com/google/marl) as the scheduler. Marl provides
cross-platform low-overhead fibers and is compatible with this style of
scheduling as it was built for the Swiftshader software rasterizer.

Even if IREE was only targeting CPUs the assertion is that we would still want
to schedule this way and it's only an incidental benefit that if building for
heterogeneous targets the scheduling code may be shared (just with a different
divisor for workgroup count calculations).

## `vm`: Lightweight Virtual Machine

The VM is designed as a dynamic linkage ABI, stable bytecode representation, and
intermediate lowering IR. Many of the optimizations we can perform on it will
benefit all use cases (such as when lowering to LLVM IR) by allowing
higher-level program transformations around synchronization that are difficult
to perform on arbitrary LLVM IR.

### Coroutines for Batching and Cooperative Scheduling

One of the largest features currently missing from the VM is coroutines (aka
user-mode fiber scheduling). Coroutines are what will allow us to have multiple
in-flight invocations into a module - some of which may be waiting on external
events - without the need for complex multithreading logic or state machine
machinations.

In many cases
[once semaphores are exposed to callers](#timeline-semaphores-on-the-module-abi)
we will not need to yield in the VM. The user will call into the module with
provided semaphores, the work to perform will be recorded to one or more command
buffers and submitted to the device, and then control return will return to the
caller immediately.

In cases requiring host readbacks that we were not able to remove, however,
additional VM code may need to run prior to when the final semaphore is
signaled. To preserve the asynchronous interface and immediate execution
guarantees the compiler can emit explicit yield points (`vm.yield`) that are
known-good locations for yielding (such as most resources not required after the
yield having been flushed/discarded, partial synchronization scope availability
if other work may be able to execute concurrently irrespective of the yielded
coroutine, etc).

When the VM encounters the yield at runtime it will suspend the coroutine until
a defined condition is met. Many coroutines can be in various states at any
given time and - thanks to the resource timeline - can still be memory safe. For
example if two stateless invocations are made with a common wait semaphore both
can be recorded and submitted without waiting on each other. If there is
internal module state accessed the invocations are implicitly ordered by
invocation order (similar to what Vulkan calls
[API order](https://vulkan.lunarg.com/doc/view/1.0.26.0/linux/vkspec.chunked/ch02s02.html#fundamentals-queueoperation-apiorder))
based on internal resource timeline semaphores.

Waking the coroutines can be performed by either an application-provided
callback in the case of the application already having a periodic event which is
doing bookkeeping (such as frame end callbacks when rendering or Looper idle
events on Android), giving direct control over the frequency and location which
IREE utilizes to perform additional work. A helper will be provided as well that
runs a dedicated IREE thread to do this, but the expectation is that
applications can often do a better (and importantly more predictable) job.

By utilizing coroutines IREE will have a way to fill traditional pipeline
bubbles even with execution from the same module (let alone across modules) in
the situation where host readbacks or other logic is required. This increases
overall throughput and utilization while reducing host wakeups as many
coroutines can be processed at once to submit new work to the device queues,
though it does not help reduce per-invocation latency.

External code such as the HAL implementation or user ops may provide the wait
handles used for continuation. For example, the HAL can expose a function that
yields and wakes only when one or more timeline semaphores reach their target
values:

```mlir
// submit work
hal.device.yield %semaphore4 >= %sem4_target, %semaphore5 >= %sem5_target
// continue here, possibly much later in time
```

#### Cellular Batching

Though coroutines help throughput there is a way we've found to reduce latency
that's been documented as
[cellular batching](http://madsys.cs.tsinghua.edu.cn/publications/EUROSYS2018-gao.pdf).
This same technique has been implemented in prior internal systems and is one of
the motivating design goals for IREE's creation. The core idea is to identify
small uniform work that can be partitioned and scheduled greedily such as to
enable batching or reduce associated invocation costs (such as refreshing
accelerator SRAM/caches with new parameters). This usually manifests as finding
large GEMM/GEMV operations using the same fixed parameters and either
dynamically increasing the batch size by adding the waiting work (without
deferring the actual execution time) or sequencing them back to back to ensure
better cache utilization. Which approach is taken depends on any data
dependencies that may be present (such as LSTM state feedback edges).

With the foundation of coroutines in IREE it's possible to yield execution at
any given point - including during command buffer recording - and wake on
specific conditions. A majority of the logic can be built into the module itself
with very little need for runtime machinery, as shared VM variables can be used
to track pending work across invocations (even from different parts of the
program) and flush based on logic wholly controlled by the user or compiler
(such as count/max time latency/etc limits). This allows for the large variety
of scheduling behavior various applications may want to use, such as a
zero-latency batch-only-within-this-invocation to a
[Nagle's Algorithm](https://en.wikipedia.org/wiki/Nagle%27s_algorithm)-esque
time or limit based behavior or even some learned model-specific windowing.

Design work is still required on how to represent this in IR but the current
thought is to model the regions in which deferred execution is possible and
beneficial and allow during lowering to the VM additional transformations. This
is similar to how the async-await behavior works in C# where the async keyword
is just sugar that expands to additional generated helper utilities.

A simple strawman representation for sequential dispatch may look like:

```mlir
hal.scheduling_policy @defer_policy {
  // max time, max count, max live memory, etc
}
...
hal.command_buffer.dispatch.deferred @defer_policy, @dispatch, ...
// vm.yield added here during lowering
```

There are many cases to explore and as cellular batching can have performance
benefits of several orders of magnitudes it'll be one of the primary areas of
research in the long-term.

### Lowering to LLVM IR

For scenarios where dynamic module loading is not required and entire modules
can be compiled into applications we can lower the VM IR to LLVM IR within
MLIR's transformation pipeline. Instead of embedding `vm.call` ops that are
dispatched at runtime to things like the HAL we can instead lower to
`llvm::CallInst` to runtime-resolved function pointers. This still enables all
of the flexibility of heterogeneous/runtime-determined devices, pluggable
diagnostics, and backend composition without any need for FlatBuffers or the VM
bytecode interpreter.

The VM was designed to make such a lowering easy and the C-style struct-based
function pointer registration for runtime modules was designed to make emitting
code that used it fairly robust even when linked in dynamically such as when
embedded in shared objects.

An extension of this is what we've been calling 'runtimeless mode', where the
IREE VM linkage code is statically linked into the binary alongside the
generated module LLVM IR. If only a single HAL backend is linked in then (with
some build-fu) we should be able to get call devirtualization to reduce code
size to precisely the functionality used by the module.

### Improved Type Support

Currently the VM only supports two types: `i32` and `vm.ref<T>`. This is an
intentional limitation such that we can determine what is really needed to
express the scheduling we perform, with the idea being that such a limited model
will make it easier to use techniques like
[indirect command buffers](#indirect-command-bufferon-accelerator-execution) to
compile the VM itself to an accelerator executable that dispatches work without
host involvement.

As we port more models we may find a few primitives that are worth bringing into
the VM design such that it's worth potential complications to future porting.
These includes types like `f32` (for simple float calculations/comparisons),
`list`/`dict` (easier python compatibility), and `vector<4xf32>` (for simple
inline calculations that are not worth dispatch overhead/synchronization).

### Indirect Command Buffer/On-Accelerator Execution

Though IREE will use many different tricks such as
[predication](#predication-of-flowdispatch) to build deep pipelines there is
still the requirement that the command recording and submission happens on the
host CPU. Though the cost of this in terms of latency and power use can be
minimized by coalescing and timelines there is still the possibility of
non-trivial roundtrips being introduced that limit performance. For particular
applications like low-power always-on compute or where there is significantly
branchy behavior (such as search algorithms) it is important that the decision
making logic as to what is dispatched runs as close to real-time as possible
within the execution pipeline.

The IREE VM is designed to be runnable on-device in a secure and cooperative way
(no pointers, indirect buffer handles to allow for memory space rearrangement
op-to-op, deterministic execution and explicit yield points, etc).

The recent efforts to bring indirect command buffers to Vulkan and Metal's
[Indirect Command Buffers](https://developer.apple.com/documentation/metal/indirect_command_buffers/encoding_indirect_command_buffers_on_the_gpu)
(that both derive inspiration from
[NV_command_list](https://www.khronos.org/registry/OpenGL/extensions/NV/NV_command_list.txt))
are one such target for this. Either by
[lowering the VM IR to LLVM IR](#lowering-to-llvm-ir) or SPIR-V, by a special
conversion to target-specific forms, or by actually executing the VM bytecode
directly on-device (it's ~1000 LoC) we should be able to prototype what full
on-device usage is like. Even if only some VM functions the compiler deems
useful to schedule on the device are used and the rest run on the host
(particularly those functions calling imported functions) some of the most
costly logic that creates tight coupling of the host and device scheduling can
be limited.
