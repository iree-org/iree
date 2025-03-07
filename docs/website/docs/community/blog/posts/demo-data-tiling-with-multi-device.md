---
date: 2025-03-06
authors:
  - hanhanW
categories:
  - Performance
tags:
  - CPU
---

# Demo: Data-tiling with multi-device

This write-up demonstrates how data-tiling works when there are multiple devices. It is the write-up followed by [How data-tiling works with encoding specialization](./data-tiling-with-encoding-specialization.md).

<!-- more -->

## Setup

Test source program: dt_multi_device.mlir
The program runs a matmul on a device targeting zen4 CPU, and the other matmul on a device targeting VMVX. At the end, the sum of two matmul results is printed.

Note: it's hard to pass flags for the device configs today because MLIR attributes don't really work well in shells with all the #'s and such. In this case, we hardcoded the executable target in the IR for the demo.

```mlir
// x86 CPU that has `+avx512f` feature.
#executable_target_embedded_elf_x86_64_with_encoding_layout = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64",
  {cpu = "znver4", cpu_features = "+avx512f",
   data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
   native_vector_size = 64 : i64,
   target_triple = "x86_64-unknown-unknown-eabi-elf",
   iree.encoding.resolver = #iree_cpu.cpu_encoding_layout<>
   }>

// VMVX with ukernels enabled.
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver = #iree_cpu.vmvx_encoding_layout<>, ukernels = "all"}>

util.global private @device_a = #hal.device.target<"local", {ordinal = 0 : index}, [
  #executable_target_embedded_elf_x86_64_with_encoding_layout
]> : !hal.device
util.global private @device_b = #hal.device.target<"local", {ordinal = 1 : index}, [
  #executable_target_vmvx_bytecode_fb
]> : !hal.device

func.func @foo(
  %lhs: tensor<?x?xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>},
  %rhs: tensor<?x?xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>}) -> (tensor<?x?xf32> {iree.abi.affinity = #hal.device.affinity<@device_a>}) {

  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %K = tensor.dim %lhs, %c1 : tensor<?x?xf32>
  %N = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %op = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  // Execute matmul on device_a and transfer the result to device_b
  %transient_op = flow.tensor.transfer %op : tensor<?x?xf32>{%M, %N} to #hal.device.affinity<@device_b>

  // Transfer input data to device_b
  %lhsb = flow.tensor.transfer %lhs : tensor<?x?xf32>{%M, %K} to #hal.device.affinity<@device_b>
  %rhsb = flow.tensor.transfer %rhs : tensor<?x?xf32>{%K, %N} to #hal.device.affinity<@device_b>
  %initb = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fillb = linalg.fill ins(%cst : f32) outs(%initb : tensor<?x?xf32>) -> tensor<?x?xf32>

  // Execute matmul on device_b and accumulate the result and the result from device_a.
  %opb = linalg.matmul
      ins(%lhsb, %rhsb : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fillb : tensor<?x?xf32>) -> tensor<?x?xf32>
  %add = arith.addf %transient_op, %opb : tensor<?x?xf32>

  // Transfer the result from device_b -> device_a.
  %result_a = flow.tensor.transfer %add : tensor<?x?xf32>{%M, %N} to #hal.device.affinity<@device_a>

  // Return the result on device_a.
  func.return %result_a : tensor<?x?xf32>
}
```

Compilation:

```mlir
iree-compile \
  --iree-execution-model=async-external \
  --iree-global-opt-enable-early-materialization=false \
  --iree-stream-experimental-specialize-encodings \
  ~/dt_multi_device.mlir -o ~/dt_multi_device.vmfb
```

## Walkthrough

Most of the details are as the same as the [previous write-up](./data-tiling-with-encoding-specialization.md). The key is in SpecializeEncoding pass and how we materialize the encodings in backends.

### SpecializeEncoding

IREE deduplicates executables after it outlines dispatches to executables. It is very reasonable in a program because we do not want to generate duplicated artifacts. However, there are issues when multi-device and encodings are involved.

Take a look at the below snippet. There is an executable that set encodings on the source tensor, and there are two dispatch ops. One launch the kernel on device_a, and the other launch the kernel on device_b. It can produce wrong codegen artifacts when bindings types are encoded (i.e., the tensor type has an encoding attribute). Because they can result in different layouts. It is confusing what the input layouts for the executable because there are two possibilities. In this case, we have to duplicate the executable with updated encoding, and modify the dispatch to launch proper executable based on resolved encoding layouts.

```mlir
stream.executable private @ex {
  stream.executable.export public @set_encoding
  builtin.module {
    func.func @set_encoding(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
      %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
      %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
      %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
      flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
      return
    }
  }
}

util.func public @multi_device_set_encoding() {
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@set_encoding(%0, %N, %K) : (tensor<?x?xf32>{%N, %K} in !stream.resource<*>{%c16}, index, index) -> (tensor<?x?xf32, #encoding>{%N, %K} in !stream.resource<*>{%c16})
  %4 = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex::@set_encoding(%3, %N, %K) : (tensor<?x?xf32>{%N, %K} in !stream.resource<*>{%c16}, index, index) -> (tensor<?x?xf32, #encoding>{%N, %K} in !stream.resource<*>{%c16})
  util.return
}
```

Thus, the SpecializeEncoding pass collects all the layout variants per executable, duplicate the executables with updated encodings, and update the dispatch op to launch the corresponding executable. See the below example.

Note that the duplication does not only look at execution affinity, but also look at the layouts for each input operands. Because the actual layout can vary based on where the input operands come from.

```mlir
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_encoding.specialized_encoding<123, tensor<?x?xf32>>]>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_encoding.specialized_encoding<456, tensor<?x?xf32>>]>

// -------------------------------- //
// encoding2 does not have layouts. //
// -------------------------------- //
#encoding2 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2]>
stream.executable private @ex {
  stream.executable.export public @set_encoding
  builtin.module {
    func.func @set_encoding(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
      %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
      %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
      %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding2>

      // --------------------------------------------------------------- //
      // This is the key, which is a #encoding2 -> #encoding conversion. //
      // --------------------------------------------------------------- //
      flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding2> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
      return
    }
  }
}

stream.executable private @ex_dup0 {
  stream.executable.export public @set_encoding
  builtin.module {
    func.func @set_encoding(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
      %c0 = arith.constant 0 : index
      %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
      %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
      %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
      %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
      %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
      %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding2>

      // --------------------------------------------------------------- //
      // This is the key, which is a #encoding1 -> #encoding conversion. //
      // --------------------------------------------------------------- //
      flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding2> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
      return
    }
  }
}
util.func public @multi_device_set_encoding() {
  // Launch @ex::@set_encoding executable, which is specialized for the dispatch op.
  %1 = stream.tensor.dispatch on(#hal.device.affinity<@device_a>) @ex::@set_encoding(%0, %arg2, %arg3) : (tensor<?x?xf32>{%arg2, %arg3} in !stream.resource<*>{%c16}, index, index) -> tensor<?x?xf32, #encoding>{%arg2, %arg3} in !stream.resource<*>{%c16}

  // Launch @ex_dup0::@set_encoding executable, which is specialized for the dispatch op.
  %4 = stream.tensor.dispatch on(#hal.device.affinity<@device_b>) @ex_dup0::@set_encoding(%3, %arg2, %arg3) : (tensor<?x?xf32>{%arg2, %arg3} in !stream.resource<*>{%c16}, index, index) -> tensor<?x?xf32, #encoding1>{%arg2, %arg3} in !stream.resource<*>{%c16}
  util.return
}
```

For more examples, see the [lit tests](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Dialect/Stream/Transforms/test/specialize_encodings.mlir)

### MaterializeEncoding

As shown in the previous section, the encodings attached on bindings are updated. They now have the resolved layouts information. Thus, there are two kind of encodings in an executable. One is for incoming buffers with resolved layouts, and the other is original layout that attached on computation ops (e.g., set_encoding, unset_encoding, matmul, etc). The encodings on the computation ops are materialized to the target device preferred layout.

If multi-device are not involved, they result in the same layout. In this context, we do not need to transfer layouts.

```mlir
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f", encoding = #iree_cpu.cpu_encoding_layout<>}>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [1, 1], outerDimsPerm = [0, 1]}}>]>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding1 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 1, 32, 32>>
func.func @set_encoding_LHS_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256xf32>> -> tensor<1x256xf32>
  %3 = iree_encoding.set_encoding %2 : tensor<1x256xf32> -> tensor<1x256xf32, #encoding1>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32, #encoding1> -> !flow.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  return
}
```

The issue is what if the layouts mismatch? I.e., the incoming buffer layouts are different from the resolved layouts on load/store ops.

The fact is that the attribute attached in the encoding knows the details. We can introduce a `bringToGlobalLayout` interface method, and the attribute implement it. It generates a sequence of operations that bring the current layout to the global layout (e.g., the tensor type without encoding). Then we can introduce a `bringToTiledLayout` interface method. It generates operations that bring the global layout to the target preferred layout.

In this context, the `flow.dispatch.tensor.load/store` materialization patterns can call the interface methods and finish the layout transfer.

This is not done yet, and the work is being [tracked](https://github.com/iree-org/iree/issues/19896).
