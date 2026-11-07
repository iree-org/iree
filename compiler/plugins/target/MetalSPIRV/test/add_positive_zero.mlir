// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-opt %s --iree-hal-serialize-all-executables="dump-binaries-path=%t" --iree-metal-compile-to-metallib=false -o %t/output.mlir
// RUN: FileCheck %s --check-prefix=RHS --input-file=%t/module_rhs_zero_metal0.metal
// RUN: FileCheck %s --check-prefix=LHS --input-file=%t/module_lhs_zero_metal0.metal
// RUN: FileCheck %s --check-prefix=VECTOR --input-file=%t/module_vector_zero_metal0.metal
// RUN: FileCheck %s --check-prefix=NEG --input-file=%t/module_negative_zero_metal0.metal
// RUN: iree-opt %s --iree-hal-serialize-all-executables -o %t/compiled.mlir

// RHS: {{\[\[clang::optnone\]\]}} T spvFAdd(T l, T r)
// RHS-LABEL: kernel void main0
// RHS: spvFAdd(
// RHS-NOT: spvFMul

// LHS: {{\[\[clang::optnone\]\]}} T spvFAdd(T l, T r)
// LHS-LABEL: kernel void main0
// LHS: spvFAdd(

// VECTOR: {{\[\[clang::optnone\]\]}} T spvFAdd(T l, T r)
// VECTOR-LABEL: kernel void main0
// VECTOR: spvFAdd(

// NEG-NOT: spvFAdd
// NEG-LABEL: kernel void main0
// NEG-NOT: spvFAdd

module {
  hal.executable public @rhs_zero {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
          spirv.GlobalVariable @output bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
          spirv.func @main0() "None" {
            %index = spirv.Constant 0 : i32
            %zero = spirv.Constant 0.0 : f32
            %input = spirv.mlir.addressof @input : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
            %output = spirv.mlir.addressof @output : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
            %in = spirv.AccessChain %input[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %out = spirv.AccessChain %output[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %value = spirv.Load "StorageBuffer" %in : f32
            %sum = spirv.FAdd %value, %zero : f32
            spirv.Store "StorageBuffer" %out, %sum : f32
            %one_index = spirv.Constant 1 : i32
            %one = spirv.Constant 1.0 : f32
            %product = spirv.FMul %value, %one : f32
            %out_product = spirv.AccessChain %output[%index, %one_index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            spirv.Store "StorageBuffer" %out_product, %product : f32
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @lhs_zero {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
          spirv.GlobalVariable @output bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
          spirv.func @main0() "None" {
            %index = spirv.Constant 0 : i32
            %zero = spirv.Constant 0.0 : f32
            %input = spirv.mlir.addressof @input : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
            %output = spirv.mlir.addressof @output : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
            %in = spirv.AccessChain %input[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %out = spirv.AccessChain %output[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %value = spirv.Load "StorageBuffer" %in : f32
            %sum = spirv.FAdd %zero, %value : f32
            spirv.Store "StorageBuffer" %out, %sum : f32
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @vector_zero {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spirv.GlobalVariable @output bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
          spirv.func @main0() "None" {
            %index = spirv.Constant 0 : i32
            %zero = spirv.Constant dense<0.0> : vector<4xf32>
            %input = spirv.mlir.addressof @input : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %output = spirv.mlir.addressof @output : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>
            %in = spirv.AccessChain %input[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<4xf32>, StorageBuffer>
            %out = spirv.AccessChain %output[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<vector<4xf32>, stride=16> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<vector<4xf32>, StorageBuffer>
            %value = spirv.Load "StorageBuffer" %in : vector<4xf32>
            %sum = spirv.FAdd %value, %zero : vector<4xf32>
            spirv.Store "StorageBuffer" %out, %sum : vector<4xf32>
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
  hal.executable public @negative_zero {
    hal.executable.variant public @metal target(<"metal-spirv", "metal-msl-fb">) {
      hal.executable.export public @main0 ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>, #hal.pipeline.binding<storage_buffer>]>)
      builtin.module {
        spirv.module Logical GLSL450 requires #spirv.vce<v1.3, [Shader], []> {
          spirv.GlobalVariable @input bind(0, 0) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
          spirv.GlobalVariable @output bind(0, 1) : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
          spirv.func @main0() "None" {
            %index = spirv.Constant 0 : i32
            %zero = spirv.Constant -0.0 : f32
            %input = spirv.mlir.addressof @input : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
            %output = spirv.mlir.addressof @output : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>
            %in = spirv.AccessChain %input[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %out = spirv.AccessChain %output[%index, %index] : !spirv.ptr<!spirv.struct<(!spirv.rtarray<f32, stride=4> [0])>, StorageBuffer>, i32, i32 -> !spirv.ptr<f32, StorageBuffer>
            %value = spirv.Load "StorageBuffer" %in : f32
            %sum = spirv.FAdd %value, %zero : f32
            spirv.Store "StorageBuffer" %out, %sum : f32
            spirv.Return
          }
          spirv.EntryPoint "GLCompute" @main0
          spirv.ExecutionMode @main0 "LocalSize", 1, 1, 1
        }
      }
    }
  }
}
