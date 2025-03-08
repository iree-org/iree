---
date: 2025-03-06
authors:
  - hanhanW
categories:
  - Performance
tags:
  - CPU
readtime: 10
---

# How data-tiling works with encoding specialization

Data-tiling is a technique that transforms the input data to be in a particular
layout for good performance. It allows you to access data through the cache
hierarchy efficiently and do the computation with very less latency. IREE is a
compiler which sees the whole graph. There are many opportunities to remove
layout-transformation overheads. They may be propagated, fused into other
operations, or be constant-evaluated for weights. IREE uses encodings to apply
data-tiling technique, and the post explores how encodings work in data-tiling.

<!-- more -->

## Setup

The program runs a matmul that has dynamic shapes and f32 element types.

```mlir title="matmul_f32.mlir" linenums="1"
func.func @foo(%lhs: tensor<?x?xf32>, %rhs: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %M = tensor.dim %lhs, %c0 : tensor<?x?xf32>
  %N = tensor.dim %rhs, %c1 : tensor<?x?xf32>
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty(%M, %N) : tensor<?x?xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %op = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%fill : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %op : tensor<?x?xf32>
}
```

Compile for zen4 CPU target:

```bash
iree-compile \
  --output-format=vm-bytecode \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu=znver4 \
  --iree-llvmcpu-target-triple=x86_64-unknown-linux-gnu \
  --iree-global-opt-enable-early-materialization=false \
  ~/matmul_f32.mlir -o ~/matmul_f32.vmfb
```

Note that we have to disable the early materialization. Otherwise, the encodings
are resolved way before specialization.

## Walkthrough

Below is the IR dump after selected data-tiling passes. For more information,
see the [full IR
dump](https://gist.github.com/hanhanW/dc109593d4464937486b037192a2f3e4).

### Set Encodings

This can happen either in GlobalOptimization phase or DispatchCreation phase.
The former one is the current default, and the latter one is for data-tiling
fusion which is still on experimental path. In the example, we focus on the
default path, thus it happens in GlobalOptimization phase.

The pass set the encodings on matmul operands, and propagate the encodings
through the fill op. Because we want to fuse the fill op into the matmul kernel.
The `round_dims_to` is redundant in the configuration, you can ignore it for
now.

<details><summary>IR dump after the pass</summary>
```mlir linenums="1" hl_lines="18-23"
// -----// IR Dump After SetEncodingPass (iree-dispatch-creation-set-encoding) //----- //

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
module {
  util.func public @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @foo(%input0: tensor<?x?xf32>, %input1: tensor<?x?xf32>) -> (%output0: tensor<?x?xf32>)"}} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %2 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?x?xf32>{%0, %1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %5 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<?x?xf32>{%3, %4}
    %6 = iree_encoding.set_encoding %2 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
    %7 = iree_encoding.set_encoding %5 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding1>
    %8 = tensor.empty(%0, %4) : tensor<?x?xf32, #encoding2>
    %9 = linalg.fill ins(%cst : f32) outs(%8 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
    %10 = linalg.matmul ins(%6, %7 : tensor<?x?xf32, #encoding>, tensor<?x?xf32, #encoding1>) outs(%9 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
    %11 = iree_encoding.unset_encoding %10 : tensor<?x?xf32, #encoding2> -> tensor<?x?xf32>{%0, %4}
    %12 = hal.tensor.export %11 "output0" : tensor<?x?xf32>{%0, %4} -> !hal.buffer_view
    util.return %12 : !hal.buffer_view
  }
}
```
</details>

### Outline dispatches

There are four dispatches in total. `dispatch_0` packs the LHS operand,
`dispatch_1` packs the RHS operand, `dispatch_2` is `fill + gemm`, and
`dispatch_3` unpacks the result of matmul.

<details><summary>IR dump after the pass</summary>
```mlir linenums="1" hl_lines="19-28 37-46 55-71 80-89 99-102"
// -----// IR Dump After OutlineDispatchRegionsPass (iree-flow-outline-dispatch-regions) //----- //

#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", cpu_features = "+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+sse4a,+fma,+avx512f,+bmi,+bmi2,+aes,+pclmul,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vbmi,+avx512ifma,+avx512vpopcntdq,+avx512vbmi2,+gfni,+vpclmulqdq,+avx512vnni,+avx512bitalg,+avx512bf16,+adx,+clflushopt,+clwb,+clzero,+cx16,+cx8,+f16c,+fsgsbase,+crc32,+invpcid,+rdpru,+sahf,+lzcnt,+movbe,+mwaitx,+x87,+pku,+evex512,+prfchw,+rdpid,+rdrnd,+rdseed,+sha,+shstk,+vaes,+wbnoinvd,+xsave,+xsavec,+xsaveopt,+xsaves,+fxsr", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver= #iree_cpu.cpu_encoding_layout<>, native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64_]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_local
  flow.executable private @foo_dispatch_0 {
    flow.executable.export public @foo_dispatch_0 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg1: index, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
        return
      }
    }
  }
  flow.executable private @foo_dispatch_1 {
    flow.executable.export public @foo_dispatch_1 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_1(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg1: index, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding1>
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding1> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
        return
      }
    }
  }
  flow.executable private @foo_dispatch_2 {
    flow.executable.export public @foo_dispatch_2 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2, %arg3
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_2(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>, %arg1: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>) {
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
        %4 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0}
        %5 = flow.dispatch.tie_shape %arg1 : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3}
        %6 = flow.dispatch.tie_shape %arg6 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0} -> tensor<?x?xf32, #encoding>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3} -> tensor<?x?xf32, #encoding1>
        %9 = tensor.empty(%2, %3) : tensor<?x?xf32, #encoding2>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
        %11 = linalg.matmul ins(%7, %8 : tensor<?x?xf32, #encoding>, tensor<?x?xf32, #encoding1>) outs(%10 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #encoding2> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        return
      }
    }
  }
  flow.executable private @foo_dispatch_3 {
    flow.executable.export public @foo_dispatch_3 workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      flow.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_3(%arg0: !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>, %arg1: index, %arg2: index, %arg3: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = flow.dispatch.tie_shape %arg0 : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>{%0, %1}
        %3 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>{%0, %1} -> tensor<?x?xf32, #encoding2>
        %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding2> -> tensor<?x?xf32>{%0, %1}
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
  util.func public @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @foo(%input0: tensor<?x?xf32>, %input1: tensor<?x?xf32>) -> (%output0: tensor<?x?xf32>)"}} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %2 = hal.tensor.import %arg0 "input0" : !hal.buffer_view -> tensor<?x?xf32>{%0, %1}
    %3 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %4 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %5 = hal.tensor.import %arg1 "input1" : !hal.buffer_view -> tensor<?x?xf32>{%3, %4}
    %6 = flow.dispatch @foo_dispatch_0::@foo_dispatch_0[%0, %1](%2, %0, %1) : (tensor<?x?xf32>{%0, %1}, index, index) -> tensor<?x?xf32, #encoding>{%0, %1}
    %7 = flow.dispatch @foo_dispatch_1::@foo_dispatch_1[%3, %4](%5, %3, %4) : (tensor<?x?xf32>{%3, %4}, index, index) -> tensor<?x?xf32, #encoding1>{%3, %4}
    %8 = flow.dispatch @foo_dispatch_2::@foo_dispatch_2[%1, %3, %0, %4](%6, %7, %1, %3, %0, %4) : (tensor<?x?xf32, #encoding>{%0, %1}, tensor<?x?xf32, #encoding1>{%3, %4}, index, index, index, index) -> tensor<?x?xf32, #encoding2>{%0, %4}
    %9 = flow.dispatch @foo_dispatch_3::@foo_dispatch_3[%0, %4](%8, %0, %4) : (tensor<?x?xf32, #encoding2>{%0, %4}, index, index) -> tensor<?x?xf32>{%0, %4}
    %10 = hal.tensor.export %9 "output0" : tensor<?x?xf32>{%0, %4} -> !hal.buffer_view
    util.return %10 : !hal.buffer_view
  }
}
```
</details>

### ConvertToStream

The flow executables all become stream executables, and the function arguments
become opaque types (i.e., `stream.binding` types). At the Stream level, it does
not need to know anything about Flow.

The host code is mostly about `stream.tensor.sizeof` and
`stream.tensor.dispatch`. The `sizeof` op takes a tensor type with an optional
encoding. It indicates the storage buffer size that will be used to issue the
allocation later on. The `dispatch` op describes how we call the functions in
the program.

<details><summary>IR dump after the pass</summary>
```mlir linenums="1" hl_lines="100-118"
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", cpu_features = "+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+sse4a,+fma,+avx512f,+bmi,+bmi2,+aes,+pclmul,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vbmi,+avx512ifma,+avx512vpopcntdq,+avx512vbmi2,+gfni,+vpclmulqdq,+avx512vnni,+avx512bitalg,+avx512bf16,+adx,+clflushopt,+clwb,+clzero,+cx16,+cx8,+f16c,+fsgsbase,+crc32,+invpcid,+rdpru,+sahf,+lzcnt,+movbe,+mwaitx,+x87,+pku,+evex512,+prfchw,+rdpid,+rdrnd,+rdseed,+sha,+shstk,+vaes,+wbnoinvd,+xsave,+xsavec,+xsaveopt,+xsaves,+fxsr", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver= #iree_cpu.cpu_encoding_layout<>, native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64_]> : !hal.device
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_local
  stream.executable private @foo_dispatch_0 {
    stream.executable.export public @foo_dispatch_0_set_encoding_LHS_DxD workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_0_set_encoding_LHS_DxD(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
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
  stream.executable private @foo_dispatch_1 {
    stream.executable.export public @foo_dispatch_1_set_encoding_RHS_DxD workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_1_set_encoding_RHS_DxD(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding1>
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding1> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
        return
      }
    }
  }
  stream.executable private @foo_dispatch_2 {
    stream.executable.export public @foo_dispatch_2_matmul_DxDxD_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2, %arg3
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_2_matmul_DxDxD_f32(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !stream.binding) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
        %4 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0}
        %5 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3}
        %6 = stream.binding.subspan %arg6[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0} -> tensor<?x?xf32, #encoding>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3} -> tensor<?x?xf32, #encoding1>
        %9 = tensor.empty(%2, %3) : tensor<?x?xf32, #encoding2>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
        %11 = linalg.matmul ins(%7, %8 : tensor<?x?xf32, #encoding>, tensor<?x?xf32, #encoding1>) outs(%10 : tensor<?x?xf32, #encoding2>) -> tensor<?x?xf32, #encoding2>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #encoding2> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        return
      }
    }
  }
  stream.executable private @foo_dispatch_3 {
    stream.executable.export public @foo_dispatch_3_unset_encoding_RESULT_DxD workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_3_unset_encoding_RESULT_DxD(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>{%0, %1} -> tensor<?x?xf32, #encoding2>
        %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding2> -> tensor<?x?xf32>{%0, %1}
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
  util.func public @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @foo(%input0: tensor<?x?xf32>, %input1: tensor<?x?xf32>) -> (%output0: tensor<?x?xf32>)"}} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32>{%0, %1} : index
    %3 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf32>{%0, %1} in !stream.resource<external>{%2}
    %4 = stream.async.transfer %3 : !stream.resource<external>{%2} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<*>{%2}
    %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    %element_type_f32_0 = hal.element_type<f32> : i32
    %dense_row_major_1 = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%5, %6]) type(%element_type_f32_0) encoding(%dense_row_major_1)
    %7 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32>{%5, %6} : index
    %8 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?x?xf32>{%5, %6} in !stream.resource<external>{%7}
    %9 = stream.async.transfer %8 : !stream.resource<external>{%7} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<*>{%7}
    %10 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32, #encoding>{%0, %1} : index
    %11 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_0::@foo_dispatch_0_set_encoding_LHS_DxD[%0, %1](%4, %0, %1) : (tensor<?x?xf32>{%0, %1} in !stream.resource<*>{%2}, index, index) -> tensor<?x?xf32, #encoding>{%0, %1} in !stream.resource<*>{%10}
    %12 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32, #encoding1>{%5, %6} : index
    %13 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_1::@foo_dispatch_1_set_encoding_RHS_DxD[%5, %6](%9, %5, %6) : (tensor<?x?xf32>{%5, %6} in !stream.resource<*>{%7}, index, index) -> tensor<?x?xf32, #encoding1>{%5, %6} in !stream.resource<*>{%12}
    %14 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32, #encoding2>{%0, %6} : index
    %15 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_2::@foo_dispatch_2_matmul_DxDxD_f32[%1, %5, %0, %6](%11, %13, %1, %5, %0, %6) : (tensor<?x?xf32, #encoding>{%0, %1} in !stream.resource<*>{%10}, tensor<?x?xf32, #encoding1>{%5, %6} in !stream.resource<*>{%12}, index, index, index, index) -> tensor<?x?xf32, #encoding2>{%0, %6} in !stream.resource<*>{%14}
    %16 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32>{%0, %6} : index
    %17 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_3::@foo_dispatch_3_unset_encoding_RESULT_DxD[%0, %6](%15, %0, %6) : (tensor<?x?xf32, #encoding2>{%0, %6} in !stream.resource<*>{%14}, index, index) -> tensor<?x?xf32>{%0, %6} in !stream.resource<*>{%16}
    %18 = stream.async.transfer %17 : !stream.resource<*>{%16} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<external>{%16}
    %19 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %18 : tensor<?x?xf32>{%0, %6} in !stream.resource<external>{%16} -> !hal.buffer_view
    util.return %19 : !hal.buffer_view
  }
}
```
</details>

### Specialize Encoding

There are few key operations in the IR. They are categorized as Stream tensor
ops. They all have the `IREE::Stream::TensorPhase` trait. The SpecializeEncoding
pass uses the `encoding` attribute from the executable_target to resolve the
layouts for the encodings.

In this example, the encoding layout resolver is
`#iree_cpu.cpu_encoding_layout<>`. The attribute implements [the device/host
encoding layout attribute
interface](https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/Codegen/ExternalInterfaces/CPUEncodingExternalModels.cpp).

Before the specialization, only the op information is encoded. After the
specialization, the layout is resolved into a serialized IR.  In this example,
they are dictionary attributes representing the [MaterializeEncodingInfo
struct](https://github.com/iree-org/iree/blob/d7c6c7bae479324676eb3b25234a312581d9350c/compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenTypes.h#L85-L96).

```mlir linenums="1" hl_lines="8-13 24-29"
#map0 = affine_map<(m, n, k) -> (m, k)>
#map1 = affine_map<(m, n, k) -> (k, n)>
#map2 = affine_map<(m, n, k) -> (m, n)>
#executable_target_vmvx_bytecode_fb = #hal.executable.target<"vmvx", "vmvx-bytecode-fb", {iree.encoding.resolver= #iree_cpu.vmvx_encoding_layout<>}>
#executable_target_x86_64 = #hal.executable.target<"llvm-cpu", "xyz", {iree.encoding.resolver= #iree_cpu.cpu_encoding_layout<>, target_triple="x86_64-xyz-xyz", cpu_features="+avx512f"}>
#device_target_local_0_ = #hal.device.target<"local", {ordinal = 0 : index}, [#executable_target_vmvx_bytecode_fb]> : !hal.device
#device_target_local_1_ = #hal.device.target<"local", {ordinal = 1 : index}, [#executable_target_x86_64]> : !hal.device
#encoding = #iree_encoding.encoding<
  operand_index = 0 : index,
  op_type =  matmul,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map0, #map1, #map2]
>
module {
  util.global private @device_a = #device_target_local_0_
  util.global private @device_b = #device_target_local_1_

  util.func public @tensor_sizeof(%d0: index, %d1: index) -> (index, index) {
    %size0 = stream.tensor.sizeof on(#hal.device.affinity<@device_a>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
    %size1 = stream.tensor.sizeof on(#hal.device.affinity<@device_b>) tensor<?x?xf32, #encoding>{%d0, %d1} : index
    util.return %size0, %size1 : index, index
  }
}
// CHECK:       #[[$ENCODING0:.+]] = #iree_encoding.encoding
// CHECK-SAME:    #iree_cpu.vmvx_encoding_layout
// CHECK-SAME:    encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK:       #[[$ENCODING1:.+]] = #iree_encoding.encoding
// CHECK-SAME:    #iree_cpu.cpu_encoding_layout
// CHECK-SAME:    encoding_info = {innerDimsPos = [{{.+}}], innerTileSizes = [{{.+}}], outerDimsPerm = [{{.+}}]}
// CHECK-LABEL: util.func public @tensor_sizeof
// CHECK:         %[[D0_RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING0]]>
// CHECK:         %[[D1_RES:.+]] = stream.tensor.sizeof {{.+}} tensor<?x?xf32, #[[$ENCODING1]]>
// CHECK:         return %[[D0_RES]], %[[D1_RES]]
```

On the executable side, the encodings attached on stream.bindings are also
updated with resolved layouts, which is consistent with the changes in stream
tensor ops. The encodings on other operations (e.g., set_encoding) will be
materialized later on in CodeGen pipeline.

```mlir linenums="1" hl_lines="7-15 17-25 32-33 35-36"
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#executable_target = #hal.executable.target<"llvm-cpu", "xyz", {target_triple = "x86_64-xyz-xyz", cpu_features = "+avx512f", iree.encoding.resolver= #iree_cpu.cpu_encoding_layout<>}>

#encoding = #iree_encoding.encoding<
  operand_index = 0 : index,
  op_type =  matmul,
  element_types = [f32, f32, f32],
  layouts = [#iree_cpu.cpu_encoding_layout<configuration = {
    encoding_info = {innerDimsPos = [0, 1],
                     innerTileSizes = [1, 1],
                     outerDimsPerm = [0, 1]}
  }>]>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#encoding1 = #iree_encoding.encoding<
  operand_index = 0 : index,
  op_type =  matmul,
  element_types = [f32, f32, f32],
  user_indexing_maps = [#map, #map1, #map2],
  round_dims_to = array<i64: 1, 32, 32>>

func.func @set_encoding_LHS_with_layout() attributes {
  hal.executable.target = #executable_target
} {
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<1x256xf32>>
  %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect)
    : !flow.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  %2 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x256xf32>> -> tensor<1x256xf32>
  %3 = iree_encoding.set_encoding %2
    : tensor<1x256xf32> -> tensor<1x256xf32, #encoding1>
  flow.dispatch.tensor.store %3, %1, offsets = [0, 0], sizes = [1, 256], strides = [1, 1] : tensor<1x256xf32, #encoding1> -> !flow.dispatch.tensor<writeonly:tensor<1x256xf32, #encoding>>
  return
}
```

<details><summary>IR dump after the pass</summary>
```mlir linenums="1"
#encoding = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 1], outerDimsPerm = [0, 1]}}>]>
#encoding1 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [1, 0], innerTileSizes = [16, 1], outerDimsPerm = [1, 0]}}>]>
#encoding2 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], layouts = [#iree_cpu.cpu_encoding_layout<configuration = {encoding_info = {innerDimsPos = [0, 1], innerTileSizes = [16, 16], outerDimsPerm = [0, 1]}}>]>
#executable_target_embedded_elf_x86_64_ = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {cpu = "znver4", cpu_features = "+mmx,+popcnt,+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+avx,+avx2,+sse4a,+fma,+avx512f,+bmi,+bmi2,+aes,+pclmul,+avx512vl,+avx512bw,+avx512dq,+avx512cd,+avx512vbmi,+avx512ifma,+avx512vpopcntdq,+avx512vbmi2,+gfni,+vpclmulqdq,+avx512vnni,+avx512bitalg,+avx512bf16,+adx,+clflushopt,+clwb,+clzero,+cx16,+cx8,+f16c,+fsgsbase,+crc32,+invpcid,+rdpru,+sahf,+lzcnt,+movbe,+mwaitx,+x87,+pku,+evex512,+prfchw,+rdpid,+rdrnd,+rdseed,+sha,+shstk,+vaes,+wbnoinvd,+xsave,+xsavec,+xsaveopt,+xsaves,+fxsr", data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", iree.encoding.resolver= #iree_cpu.cpu_encoding_layout<>, native_vector_size = 64 : i64, target_triple = "x86_64-unknown-unknown-eabi-elf"}>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#device_target_local = #hal.device.target<"local", [#executable_target_embedded_elf_x86_64_]> : !hal.device
#encoding3 = #iree_encoding.encoding<operand_index = 0 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding4 = #iree_encoding.encoding<operand_index = 1 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
#encoding5 = #iree_encoding.encoding<operand_index = 2 : index, op_type =  matmul, element_types = [f32, f32, f32], user_indexing_maps = [#map, #map1, #map2], round_dims_to = array<i64: 32, 32, 32>>
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_local
  stream.executable private @foo_dispatch_0 {
    stream.executable.export public @foo_dispatch_0_set_encoding_LHS_DxD workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_0_set_encoding_LHS_DxD(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding3>
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding3> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding>>{%0, %1}
        return
      }
    }
  }
  stream.executable private @foo_dispatch_1 {
    stream.executable.export public @foo_dispatch_1_set_encoding_RHS_DxD workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_1_set_encoding_RHS_DxD(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%0, %1} -> tensor<?x?xf32>
        %5 = iree_encoding.set_encoding %4 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding4>
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32, #encoding4> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding1>>{%0, %1}
        return
      }
    }
  }
  stream.executable private @foo_dispatch_2 {
    stream.executable.export public @foo_dispatch_2_matmul_DxDxD_f32 workgroups(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1, %arg2, %arg3
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_2_matmul_DxDxD_f32(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: !stream.binding) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %0 = flow.dispatch.workload.ordinal %arg2, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg3, 1 : index
        %2 = flow.dispatch.workload.ordinal %arg4, 2 : index
        %3 = flow.dispatch.workload.ordinal %arg5, 3 : index
        %4 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0}
        %5 = stream.binding.subspan %arg1[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3}
        %6 = stream.binding.subspan %arg6[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        %7 = flow.dispatch.tensor.load %4, offsets = [0, 0], sizes = [%2, %0], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding>>{%2, %0} -> tensor<?x?xf32, #encoding3>
        %8 = flow.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%1, %3], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding1>>{%1, %3} -> tensor<?x?xf32, #encoding4>
        %9 = tensor.empty(%2, %3) : tensor<?x?xf32, #encoding5>
        %10 = linalg.fill ins(%cst : f32) outs(%9 : tensor<?x?xf32, #encoding5>) -> tensor<?x?xf32, #encoding5>
        %11 = linalg.matmul ins(%7, %8 : tensor<?x?xf32, #encoding3>, tensor<?x?xf32, #encoding4>) outs(%10 : tensor<?x?xf32, #encoding5>) -> tensor<?x?xf32, #encoding5>
        flow.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%2, %3], strides = [1, 1] : tensor<?x?xf32, #encoding5> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32, #encoding2>>{%2, %3}
        return
      }
    }
  }
  stream.executable private @foo_dispatch_3 {
    stream.executable.export public @foo_dispatch_3_unset_encoding_RESULT_DxD workgroups(%arg0: index, %arg1: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice %arg0, %arg1
      stream.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @foo_dispatch_3_unset_encoding_RESULT_DxD(%arg0: !stream.binding, %arg1: index, %arg2: index, %arg3: !stream.binding) {
        %c0 = arith.constant 0 : index
        %0 = flow.dispatch.workload.ordinal %arg1, 0 : index
        %1 = flow.dispatch.workload.ordinal %arg2, 1 : index
        %2 = stream.binding.subspan %arg0[%c0] : !stream.binding -> !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>{%0, %1}
        %3 = stream.binding.subspan %arg3[%c0] : !stream.binding -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        %4 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<?x?xf32, #encoding2>>{%0, %1} -> tensor<?x?xf32, #encoding5>
        %5 = iree_encoding.unset_encoding %4 : tensor<?x?xf32, #encoding5> -> tensor<?x?xf32>{%0, %1}
        flow.dispatch.tensor.store %5, %3, offsets = [0, 0], sizes = [%0, %1], strides = [1, 1] : tensor<?x?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%0, %1}
        return
      }
    }
  }
  util.func public @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @foo(%input0: tensor<?x?xf32>, %input1: tensor<?x?xf32>) -> (%output0: tensor<?x?xf32>)"}} {
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
    %element_type_f32 = hal.element_type<f32> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f32) encoding(%dense_row_major)
    %2 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32>{%0, %1} : index
    %3 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf32>{%0, %1} in !stream.resource<external>{%2}
    %4 = stream.async.transfer %3 : !stream.resource<external>{%2} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<*>{%2}
    %5 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%5, %6]) type(%element_type_f32) encoding(%dense_row_major)
    %7 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32>{%5, %6} : index
    %8 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?x?xf32>{%5, %6} in !stream.resource<external>{%7}
    %9 = stream.async.transfer %8 : !stream.resource<external>{%7} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<*>{%7}
    %10 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32, #encoding>{%0, %1} : index
    %11 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_0::@foo_dispatch_0_set_encoding_LHS_DxD[%0, %1](%4, %0, %1) : (tensor<?x?xf32>{%0, %1} in !stream.resource<*>{%2}, index, index) -> tensor<?x?xf32, #encoding>{%0, %1} in !stream.resource<*>{%10}
    %12 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32, #encoding1>{%5, %6} : index
    %13 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_1::@foo_dispatch_1_set_encoding_RHS_DxD[%5, %6](%9, %5, %6) : (tensor<?x?xf32>{%5, %6} in !stream.resource<*>{%7}, index, index) -> tensor<?x?xf32, #encoding1>{%5, %6} in !stream.resource<*>{%12}
    %14 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32, #encoding2>{%0, %6} : index
    %15 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_2::@foo_dispatch_2_matmul_DxDxD_f32[%1, %5, %0, %6](%11, %13, %1, %5, %0, %6) : (tensor<?x?xf32, #encoding>{%0, %1} in !stream.resource<*>{%10}, tensor<?x?xf32, #encoding1>{%5, %6} in !stream.resource<*>{%12}, index, index, index, index) -> tensor<?x?xf32, #encoding2>{%0, %6} in !stream.resource<*>{%14}
    %16 = stream.tensor.sizeof on(#hal.device.affinity<@__device_0>) tensor<?x?xf32>{%0, %6} : index
    %17 = stream.tensor.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_3::@foo_dispatch_3_unset_encoding_RESULT_DxD[%0, %6](%15, %0, %6) : (tensor<?x?xf32, #encoding2>{%0, %6} in !stream.resource<*>{%14}, index, index) -> tensor<?x?xf32>{%0, %6} in !stream.resource<*>{%16}
    %18 = stream.async.transfer %17 : !stream.resource<*>{%16} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<external>{%16}
    %19 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %18 : tensor<?x?xf32>{%0, %6} in !stream.resource<external>{%16} -> !hal.buffer_view
    util.return %19 : !hal.buffer_view
  }
}
```
</details>

### Encode Host Tensors

Later on, the `stream.tensor.sizeof` op is lowered to the storage buffer size
calculation. Since the new encoding attribute implements the
[SerializableEncodingAttrInterface](https://github.com/iree-org/iree/blob/298faa58c64bf7db4d65a16e0fa30fa153d2fef2/compiler/src/iree/compiler/Dialect/Encoding/IR/EncodingInterfaces.td#L79-L194)
and it already has all the information. We are able to serve the needs from
target devices.

<details><summary>IR dump after the pass</summary>
```mlir linenums="1"
util.func public @foo(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @foo(%input0: tensor<?x?xf32>, %input1: tensor<?x?xf32>) -> (%output0: tensor<?x?xf32>)"}} {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c4 = arith.constant 4 : index
  %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
  %1 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[1] : index
  %element_type_f32 = hal.element_type<f32> : i32
  %dense_row_major = hal.encoding_type<dense_row_major> : i32
  hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%0, %1]) type(%element_type_f32) encoding(%dense_row_major)
  %2 = arith.muli %0, %c4 : index
  %3 = arith.muli %2, %1 : index
  %4 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<?x?xf32>{%0, %1} in !stream.resource<external>{%3}
  %5 = stream.async.transfer %4 : !stream.resource<external>{%3} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<*>{%3}
  %6 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
  %7 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[1] : index
  hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%6, %7]) type(%element_type_f32) encoding(%dense_row_major)
  %8 = arith.muli %6, %c4 : index
  %9 = arith.muli %8, %7 : index
  %10 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<?x?xf32>{%6, %7} in !stream.resource<external>{%9}
  %11 = stream.async.transfer %10 : !stream.resource<external>{%9} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<*>{%9}
  %12 = arith.ceildivsi %0, %c16 : index
  %13 = arith.muli %12, %c16 : index
  %14 = arith.muli %13, %c4 : index
  %15 = arith.muli %14, %1 : index
  %16 = stream.async.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_0::@foo_dispatch_0_set_encoding_LHS_DxD[%0, %1](%5[%c0 to %3 for %3], %0, %1) : (!stream.resource<*>{%3}, index, index) -> !stream.resource<*>{%15}
  %17 = arith.ceildivsi %7, %c16 : index
  %18 = arith.muli %17, %c16 : index
  %19 = arith.muli %6, %c4 : index
  %20 = arith.muli %19, %18 : index
  %21 = stream.async.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_1::@foo_dispatch_1_set_encoding_RHS_DxD[%6, %7](%11[%c0 to %9 for %9], %6, %7) : (!stream.resource<*>{%9}, index, index) -> !stream.resource<*>{%20}
  %22 = arith.ceildivsi %0, %c16 : index
  %23 = arith.muli %22, %c16 : index
  %24 = arith.ceildivsi %7, %c16 : index
  %25 = arith.muli %24, %c16 : index
  %26 = arith.muli %23, %c4 : index
  %27 = arith.muli %26, %25 : index
  %28 = stream.async.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_2::@foo_dispatch_2_matmul_DxDxD_f32[%1, %6, %0, %7](%16[%c0 to %15 for %15], %21[%c0 to %20 for %20], %1, %6, %0, %7) : (!stream.resource<*>{%15}, !stream.resource<*>{%20}, index, index, index, index) -> !stream.resource<*>{%27}
  %29 = arith.muli %0, %c4 : index
  %30 = arith.muli %29, %7 : index
  %31 = stream.async.dispatch on(#hal.device.affinity<@__device_0>) @foo_dispatch_3::@foo_dispatch_3_unset_encoding_RESULT_DxD[%0, %7](%28[%c0 to %27 for %27], %0, %7) : (!stream.resource<*>{%27}, index, index) -> !stream.resource<*>{%30}
  %32 = stream.async.transfer %31 : !stream.resource<*>{%30} from(#hal.device.affinity<@__device_0>) -> to(#hal.device.affinity<@__device_0>) !stream.resource<external>{%30}
  %33 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %32 : tensor<?x?xf32>{%0, %7} in !stream.resource<external>{%30} -> !hal.buffer_view
  util.return %33 : !hal.buffer_view
}
```
</details>

### CodeGen Encoding

At this moment, all the stream ops are lowered to HAL ops. It is CodeGen's
responsibility to materialize the encodings within the executables.

The layout transfer is not scoped in the doc, so we are not going to talk the
details. The rest of the work is as the same as the configuration without
encoding specialization.

For more information, see [CPU data-tiling
demo](https://gist.github.com/bjacob/32e540ad948a86c07e2cdb49a1a97421) and [GPU
data-tiling
demo](https://gist.github.com/bjacob/51f7d0e308a26f124a4c2aa17762a553).
