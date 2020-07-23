# Simple IR Walkthrough

Note that this doc is quite outdated. We expect to update it soon.

## Overview

This walks through the process of lowering TensorFlow python to an IREE module,
demonstrating the MLIR that exists at each stage. Many individual intermediate
transforms are skipped for clarity but the major dialect milestones during
lowering are present.

**NOTE**: this represents the IR as it exists at the time of writing, which is
planned to undergo significant changes soon. Take this more as a conceptual
walkthrough than a reference for what the IR looks like.

## TensorFlow to XLA HLO

The "frontend" in this example is TensorFlow and we import that into MLIR in the
TensorFlow dialect and lower it to the mid-level IR of XLA HLO. Many backends
can consume the XLA HLO (such as TPU and CUDA), not just IREE, meaning that the
work required to convert the TensorFlow ops to the much more restricted set of
XLA HLO is shared amongst many projects.

### TensorFlow Python

This is using the TensorFlow 1.0 syntax producing a GraphDef. IREE is designed
to work best with the TensorFlow 2.0 SavedModel representation.

```python
import tensorflow as tf
with tf.Session() as session:
  arg0 = tf.placeholder(tf.float32, shape=[4])
  arg1 = tf.placeholder(tf.float32, shape=[4])
  result = tf.multiply(arg0, arg1)
  print(session.graph_def)
```

### TensorFlow GraphDef

GraphDefs do not contain information about the feeds and fetches nor do they
identify exported functions. TensorFlow 2.0 makes this significantly easier but
since most are familiar with TF1.0 the GraphDef is displayed here.

```protobuf
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value { shape { dim { size: 4 } } }
  }
}
node {
  name: "Placeholder_1"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value { shape { dim { size: 4 } } }
  }
}
node {
  name: "Mul"
  op: "Mul"
  input: "Placeholder"
  input: "Placeholder_1"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
}
```

### XLA HLO

XLA HLO is the dialect we try to lower to as instead of 1400+ ops in TensorFlow
we end up with ~30 ops that better represent the actual math being performed.
The
[XLA Operation Semantics](https://www.tensorflow.org/xla/operation_semantics)
are well(ish) documented and a great starting point for lowering into other
dialects. The existing
[tf2xla](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/tf2xla)
bridge can be used to convert the ops from GraphDef to XLA HLO, while a
[new implementation](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/mlir/xla/transforms)
based in MLIR is currently being written.

```mlir
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
    attributes { iree.module.export } {
  %0 = mhlo.multiply(%arg0, %arg1) : tensor<4xf32>
  return %0 : tensor<4xf32>
}
```

What was a graph of nodes now looks much more like a traditional program:
there's a function with a well-defined signature, an operation that performs
some math on the operands, and the result of the math is returned.

In the XLA HLO dialect it's possible to express control flow (calls, loops,
conditionals), complex multi-operation regions like reductions, etc. All
TensorFlow graph semantics (control edges, switch/merge, etc) are lowered to
this form, and all data edges are converted to SSA values.

## IREE Module IR

Once lowered to XLA HLO the IREE transformations work to legalize and lower to a
high-level sequencer dialect (iree_hl_seq). At this point we are still operating
on tensors with value-semantics allowing us to use the SSA representation in
MLIR to do some relatively complex (yet easy to express) transforms.

### Dispatch Region Identification

The final IREE module is designed to have as few sequencer operations as
possible. This is achieved by clustering operations into regions such that data
dependencies and execution order are correctly observed and that the dispatch
workload (roughly the shape of the output) is compatible. Jumping ahead a bit,
the dispatch regions correspond to dispatches against the target API (such as
Vulkan vkCmdDispatch) modulo threadgroup sizes. When still operating with value
semantics it's easy to use SSA use-def chains to ensure we are preserving the
expected behavior of the program.

```mlir
func @simple_mul(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32>
  attributes  {iree.module.export} {
  %cst = constant dense<[4, 1, 1]> : tensor<3xi32>
  %0 = iree.dispatch_region[%cst : tensor<3xi32>](%arg2 = %arg0 : tensor<4xf32>, %arg3 = %arg1 : tensor<4xf32>) : tensor<4xf32> {
    %1 = mulf %arg2, %arg3 : tensor<4xf32>
    iree.return %1 : tensor<4xf32>
  }
  return %0 : tensor<4xf32>
}
```

In the above example, the workload is defined by `%cst` as 4x1x1. If there were
other ops that were also of a `dot(4,1,1)` workload we could cluster those here.

Other dispatch-like operations, such as reductions, are also identified and
clustered appropriately at this stage. What we end up with is a top-level IR
performing dispatches with nested regions containing the work to perform. When
all identification has completed the goal is to have no math outside of the
dispatch regions (though copies are permitted).

Additional passes may run that combine, split, or otherwise transform the
dispatch regions based on a set of cost functions or target capability metrics.
For example, to ensure predictable maximum latency larger dispatch regions may
be split based on how much memory bandwidth they are likely to consume.

### Executable Outlining and High-level Sequencer IR

The first step in lowering to the IREE sequencer IR describing the runtime
sequence of operations to perform is to isolate the work being performed from
how it is to be dispatched. We outline dispatch regions into `iree.executable`s
and replace the original `iree.dispatch_region` with `iree_hl_seq.dispatch` ops
referencing those executables. At this point we still have not specified what
our exact lowering targets are, however we know enough to establish the basic
ABI used to pass parameters.

```mlir
module {
  iree.multi_arch_executable @simple_mul_ex_dispatch_0() {
    iree.executable(Unspecified) {
      module {
        func @simple_mul_rgn_dispatch_0(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
  attributes  {iree.executable.export} {
          %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
          %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
          %2 = mulf %0, %1 : tensor<4xf32>
          iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
          iree.return
        }
      }
    }
  }
  func @simple_mul(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32>
  attributes  {iree.module.export} {
    %0 = iree_interp.constant dense<[4, 1, 1]> : tensor<3xi32>
    %1 = "iree_hl_seq.alloc_heap"() : () -> memref<4xf32>
    iree_hl_seq.dispatch simple_mul_ex_dispatch_0::simple_mul_rgn_dispatch_0[%0 : memref<3xi32>](%arg0, %arg1, %1) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    iree_hl_seq.return %1 : memref<4xf32>
  }
}
```

Here we've allocated the output argument for the dispatch region via
`iree_hl_seq.alloc_heap` and passed it as an argument into the dispatch. The
executable entry point function gains a matching output argument where the final
result is stored. The `iree.load_input` and `iree.store_output` pseudo-commands
are used by backends in following lowering steps to determine how to load and
store their arguments.

### Low-level Sequencer IR

Once we've established the signatures between the sequencer and the executable
we can lower the sequencer IR to an explicitly-allocated dialect and perform
memory allocation. Here we attempt to alias/reuse buffers, determine buffers
that can be entirely elided, and reorder dispatches so that they can more easily
be grouped based on required barriers. Thanks to MLIR's built-in folding logic
we can also do some IR optimizations such as converting the generic dispatch to
a `iree_ll_seq.static_dispatch`, as we know the workload size at compile-time.

As part of this we also propagate any static information we can determine, such
as the workload, into the executables. This is to help aid backends in lowering
more efficiently when possible.

```mlir
module {
  iree.multi_arch_executable @simple_mul_ex_dispatch_0[0]() {
    iree.executable(Unspecified) {
      module {
        func @simple_mul_rgn_dispatch_0(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
        attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = iree.load_input(%arg0 : memref<4xf32>) : tensor<4xf32>
          %1 = iree.load_input(%arg1 : memref<4xf32>) : tensor<4xf32>
          %2 = mulf %0, %1 : tensor<4xf32>
          iree.store_output(%2 : tensor<4xf32>, %arg2 : memref<4xf32>)
          iree.return
        }
      }
    }
  }
  func @simple_mul(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32>
  attributes  {iree.module.export, iree.ordinal = 0 : i32} {
    %0 = "iree_ll_seq.alloc_heap"() : () -> memref<4xf32>
    iree_ll_seq.static_dispatch simple_mul_ex_dispatch_0::simple_mul_rgn_dispatch_0[dense<[4, 1, 1]> : tensor<3xi32>](%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    iree_ll_seq.return %0 : memref<4xf32>
  }
}
```

### Executable Lowering to SPIR-V

For each executable and target combination we invoke an MLIR translation to some
target dialect. Here, we are lowering to the SPIR-V dialect, and use the current
IREE-specific XLA HLO-to-SPIR-V lowering passes. Other lowerings, as they become
available, can be swapped in. Below is the `simple_mul_ex_dispatch_0` executable
fully lowered to SPIR-V in the canonical MLIR SPIR-V dialect, which can be
trivially serialized to SPIR-V words. Note how the `iree.load_input` and
`iree.load_output` ops are lowered to storage buffer loads and stores.

```mlir
module {
  spv.module "Logical" "GLSL450" {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    spv.globalVariable @simple_mul_rgn_dispatch_0_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
    spv.globalVariable @simple_mul_rgn_dispatch_0_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
    spv.globalVariable @simple_mul_rgn_dispatch_0_arg_2 bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
    func @simple_mul_rgn_dispatch_0() {
      %0 = spv._address_of @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
      %1 = spv.Load "Input" %0 : vector<3xi32>
      %2 = spv.CompositeExtract %1[0 : i32] : vector<3xi32>
      %3 = spv.CompositeExtract %1[1 : i32] : vector<3xi32>
      %4 = spv.CompositeExtract %1[2 : i32] : vector<3xi32>
      %5 = spv._address_of @simple_mul_rgn_dispatch_0_arg_0 : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
      %6 = spv.constant 0 : i32
      %7 = spv.AccessChain %5[%6, %2] : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
      %8 = spv.Load "StorageBuffer" %7 : f32
      %9 = spv._address_of @simple_mul_rgn_dispatch_0_arg_1 : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
      %10 = spv.constant 0 : i32
      %11 = spv.AccessChain %9[%10, %2] : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
      %12 = spv.Load "StorageBuffer" %11 : f32
      %13 = spv.FMul %8, %12 : f32
      %14 = spv._address_of @simple_mul_rgn_dispatch_0_arg_2 : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
      %15 = spv.constant 0 : i32
      %16 = spv.AccessChain %14[%15, %2] : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
      spv.Store "StorageBuffer" %16, %13 : f32
      spv.Return
    }
    spv.EntryPoint "GLCompute" @simple_mul_rgn_dispatch_0, @globalInvocationID
    spv.ExecutionMode @simple_mul_rgn_dispatch_0 "LocalSize", 1, 1, 1
  } attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}
}
```

### Final Module

Below is the final module containing executables for both the IREE reference
interpreter backend and the Vulkan/SPIR-V backend, as well as the sequencer IR
function detailing how to dispatch the workload.

```mlir
module {
  iree.multi_arch_executable @simple_mul_ex_dispatch_0[0]() {
    iree.executable(IreeBytecode) {
      module {
        func @simple_mul_rgn_dispatch_0(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>)
        attributes  {iree.executable.export, iree.executable.workload = dense<[4, 1, 1]> : tensor<3xi32>, iree.ordinal = 0 : i32} {
          %0 = "iree_ll_interp.alloc_heap"() : () -> memref<4xf32>
          "iree_ll_interp.mul_f"(%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
          %1 = "iree_ll_interp.constant"() {value = dense<0> : tensor<1xi32>} : () -> memref<1xi32>
          %2 = "iree_ll_interp.constant"() {value = dense<4> : tensor<1xi32>} : () -> memref<1xi32>
          "iree_ll_interp.dynamic_copy"(%0, %1, %arg2, %1, %2) : (memref<4xf32>, memref<1xi32>, memref<4xf32>, memref<1xi32>, memref<1xi32>) -> ()
          iree.return
        }
      }
    }
    iree.executable(SPIRV) {
      spv.module "Logical" "GLSL450" {
        spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
        spv.globalVariable @simple_mul_rgn_dispatch_0_arg_0 bind(0, 0) : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
        spv.globalVariable @simple_mul_rgn_dispatch_0_arg_1 bind(0, 1) : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
        spv.globalVariable @simple_mul_rgn_dispatch_0_arg_2 bind(0, 2) : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
        func @simple_mul_rgn_dispatch_0() {
          %0 = spv._address_of @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
          %1 = spv.Load "Input" %0 : vector<3xi32>
          %2 = spv.CompositeExtract %1[0 : i32] : vector<3xi32>
          %3 = spv.CompositeExtract %1[1 : i32] : vector<3xi32>
          %4 = spv.CompositeExtract %1[2 : i32] : vector<3xi32>
          %5 = spv._address_of @simple_mul_rgn_dispatch_0_arg_0 : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
          %6 = spv.constant 0 : i32
          %7 = spv.AccessChain %5[%6, %2] : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
          %8 = spv.Load "StorageBuffer" %7 : f32
          %9 = spv._address_of @simple_mul_rgn_dispatch_0_arg_1 : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
          %10 = spv.constant 0 : i32
          %11 = spv.AccessChain %9[%10, %2] : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
          %12 = spv.Load "StorageBuffer" %11 : f32
          %13 = spv.FMul %8, %12 : f32
          %14 = spv._address_of @simple_mul_rgn_dispatch_0_arg_2 : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
          %15 = spv.constant 0 : i32
          %16 = spv.AccessChain %14[%15, %2] : !spv.ptr<!spv.struct<!spv.array<4 x f32 [4]> [0]>, StorageBuffer>
          spv.Store "StorageBuffer" %16, %13 : f32
          spv.Return
        }
        spv.EntryPoint "GLCompute" @simple_mul_rgn_dispatch_0, @globalInvocationID
        spv.ExecutionMode @simple_mul_rgn_dispatch_0 "LocalSize", 1, 1, 1
      } attributes {capabilities = ["Shader"], extensions = ["SPV_KHR_storage_buffer_storage_class"]}
    }
  }
  func @simple_mul(%arg0: memref<4xf32>, %arg1: memref<4xf32>) -> memref<4xf32>
  attributes  {iree.module.export, iree.ordinal = 0 : i32} {
    %0 = "iree_ll_seq.alloc_heap"() : () -> memref<4xf32>
    iree_ll_seq.static_dispatch simple_mul_ex_dispatch_0::simple_mul_rgn_dispatch_0[dense<[4, 1, 1]> : tensor<3xi32>](%arg0, %arg1, %0) : (memref<4xf32>, memref<4xf32>, memref<4xf32>) -> ()
    iree_ll_seq.return %0 : memref<4xf32>
  }
}
```

## Runtime

### IREE VM

The above IREE module (containing the sequencer function IR and the SPIR-V
executable) can be serialized to a FlatBuffer. This FlatBuffer is optimized for
minimal runtime overhead and there's zero load-time work required. This is
useful in scenarios where either ease of debugging or dynamic deployment is
required (such as when downloading models to run in a store-signed app on
Android or iOS). Since the majority of the compute-intensive work is happening
on the GPU (or CPU) via the generated SPIR-V the overhead for processing the
sequencer IR is minimal, often an order of magnitude less than traditional ML
runtimes.

The VM targets the IREE HAL API, meaning that you get access to Vulkan, CPU, and
other backends that are available in IREE. The HAL API is just an interface,
though, and is easy to map to existing application abstractions that may exist.
This means that implementing a HAL that maps to app primitives gives you access
to the VM without needing to modify the IREE compiler or VM code.

The VM is really simple and effectively does the same as demonstrated below in
the HAL codegen example, just with a bytecode instead of C++ code. This layering
allows us to optimize for the fast case (codegen) while still being able to
reuse almost the entire infrastructure for the dynamic case.

### IREE HAL Codegen

For models that are static at application compile-time and app binaries can be
redeployed if the model changes it's possible to generate C++ code that uses the
IREE HAL API. This avoids the need for a VM at the cost of recompilations when
the model changes and less debugger support. Since the HAL API is still used the
heterogeneous device support IREE provides is still available.

As with the VM the HAL API is just an interface; implementing a custom mapping
from that interface to an existing API is easy and gives the ability to switch
between VM or codegen approaches with no code beyond the interface
implementation required.

**NOTE**: this is not yet fully implemented/open sourced, but is coming soon.
Here's a pseudo-codeish example of what a module would look like:

```c++
class SimpleMulModule : public iree::vm::Module {
 public:
  // Creates the module and prepares it for execution in the given context.
  // This may assign device handles, cache executables, etc.
  static iree::StatusOr<std::unique_ptr<SimpleMulModule>> Create(
      iree::vm::Context* context) {
    // <prepare executable, allocate transient buffers, etc>
  }

  // Synchronous call to @simple_mul. Simplest form of the API and may perform
  // internal pipelining but will appear synchronous to callers.
  //
  // Note that this assumes that the inputs are available and visible to the
  // target devices. If you are exclusively using the synchronous API that will
  // be the case.
  //
  // Matches IR:
  // func @simple_mul(%arg0: memref<4xf32>,
  //                  %arg1: memref<4xf32>) -> memref<4xf32>
  iree::StatusOr<iree::hal::BufferView> simple_mul(
      iree::hal::BufferView arg0,
      iree::hal::BufferView arg1) {
    iree::hal::Device* device = select_device(0);

    // Buffers are allocated conservatively as we don't know what the caller
    // will do with it. Buffers used internally or across async calls can be
    // placed in device memory.
    //
    // Matches IR:
    // %0 = "iree_ll_seq.alloc_heap"() : () -> memref<4xf32>
    ASSIGN_OR_RETURN(auto result, device->allocator()->Allocate(
        iree::hal::MemoryType::kHostLocal |
            iree::hal::MemoryType::kDeviceVisible,
        iree::hal::BufferUsage::kDispatch |
            iree::hal::BufferUsage::kMapping));
    auto result_view = iree::hal::BufferView(
        std::move(result), {4}, sizeof(float));

    // To show that this is just a wrapper around the real execution we just
    // call into the async version of the function.
    ASSIGN_OR_RETURN(auto fence, device->CreateFence(0u));
    auto completed_fence_value = iree::hal::FenceValue{add_ref(fence), 1u};
    RETURN_IF_ERROR(simple_mul(
        device,
        /*wait_semaphore=*/{},
        arg0, arg1, result_view,
        /*signal_semaphore=*/{},
        completed_fence_value));

    // Wait until results are ready.
    RETURN_IF_ERROR(device->WaitAllFences(
        {completed_fence_value}, absl::InfiniteDuration()));

    // The allocated buffer escapes this function.
    // Callers can provide already-allocated buffers with the async API.
    //
    // Matches IR:
    // iree_ll_seq.return %0 : memref<4xf32>
    return result_view;
  }

  // Asynchronous variant of the function that can (optionally) wait on existing
  // semaphores that indicate that arguments are ready for use and
  // (optionally) signal both semaphores and fences when the results are ready.
  //
  // Multiple variants of this API can be exposed such as ones returning a
  // iree::hal::SubmissionBatch that can be submitted by the caller, however
  // this is usually fine for most uses as any additional required submissions
  // are handled internally as needed.
  iree::Status simple_mul(
      iree::hal::Device* device,
      iree::hal::SemaphoreValue wait_semaphore,
      iree::hal::BufferView arg0,
      iree::hal::BufferView arg1,
      iree::hal::BufferView out0,
      iree::hal::SemaphoreValue signal_semaphore,
      iree::hal::FenceValue signal_fence) {
    // Record the command buffer with any commands we can.
    // In more complex examples this would include barriers, events, transfers,
    // and multiple dispatches. In many cases only one command buffer is
    // required however more complex flow control may require multiple.
    //
    // Matches IR:
    // iree_ll_seq.static_dispatch ...
    ASSIGN_OR_RETURN(auto cmd, device->CreateCommandBuffer(
        iree::hal::CommandBufferMode::kOneShot,
        iree::hal::CommandCategory::kDispatch));
    RETURN_IF_ERROR(cmd->Begin());
    iree::hal::DispatchRequest dispatch_request;
    dispatch_request.executable = device_executable(device, 0);
    dispatch_request.workload = {4, 1, 1};
    dispatch_request.bindings = {
      {arg0.buffer, arg0.shape, arg0.element_size},
      {arg1.buffer, arg1.shape, arg1.element_size},
      {out0.buffer, out0.shape, out0.element_size},
    };
    RETURN_IF_ERROR(cmd->Dispatch(dispatch_request));
    RETURN_IF_ERROR(cmd->End());

    // TBD: show resource tracking.

    // Submit for execution using the semaphores we were told to wait on.
    // In more complex examples where we may have to submit multiple command
    // buffers we'll use the wait/signal semaphores as the boundary
    // synchronization primitives.
    auto* command_queue = device->command_queues()[0];
    return command_queue->Submit({
      iree::hal::SubmissionBatch{
        {wait_semaphore},
        {cmd},
        {signal_semaphore},
      },
    }, signal_fence);
  }
};
```

### Custom Codegen

Using the final IREE module (containing executables and sequencer IR) it's
possible to generate code for any target. For example, instead of using the IREE
HAL and C++ one could generate straight C directly against their target API or
hardware (such as directly calling Vulkan or launching DSP executables). We
refer to this form as "runtimeless," as beyond the code required to run the
program there's no more than what one would write by hand if they were very
carefully hand-translating the model.

Because we are still changing the IR we have not yet written a backend that does
this, however we plan to demonstrate this for targeting small embedded systems
and DSPs in the future.
