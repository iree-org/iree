// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary (CPU + Vulkan, etc).
#cpu_target = #hal.device.target<"llvm-cpu", {
  executable_targets = [
    #x86_64_target
  ]
}>

module attributes {transform.with_named_sequence} {

  // Executable containing exported shims and calls to external functions.
  // See the other examples in this directory for in-depth explanations of
  // the IR structure of this executable.
  hal.executable private @executable {
    hal.executable.variant public @x86_64 target(#x86_64_target) objects([
      #hal.executable.object<{
        path = "samples/custom_dispatch/cpu/embedded/functions_x86_64.o"
      }>
    ]) {
      hal.executable.export public @simple_mul_abs_negate ordinal(0)
          layout(#hal.pipeline.layout<push_constants = 1, sets = [
            <0, bindings = [
                <0, storage_buffer, ReadOnly>,
                <1, storage_buffer, ReadOnly>,
                <2, storage_buffer>
            ]>
          ]>) {
      ^bb0(%device: !hal.device, %workload: index):
        %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
        %c1 = arith.constant 1 : index
        hal.return %x, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func private @simple_mul_abs_negate_workgroup(%binding0: memref<?xf32>, %binding1: memref<?xf32>, %binding2: memref<?xf32>, %dim: index, %tid: index) attributes {
          hal.import.static
        }
        func.func @simple_mul_abs_negate() {
          %c0 = arith.constant 0 : index
          %dim_i32 = hal.interface.constant.load[0] : i32
          %dim = arith.index_castui %dim_i32 : i32 to index
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

          %binding0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}
          %binding1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}
          %binding2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}

          func.call @simple_mul_abs_negate_workgroup(%binding0, %binding1, %binding2, %dim, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()
          return
        }
      }
    }  // hal.executable.variant
  }  // hal.executable

  func.func @call_mul_abs_negate(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %dim_i32 = arith.index_cast %dim : index to i32

    // Dispatch a basic `ret = -|lhs * rhs|` using an external function.
    %0 = flow.dispatch @executable::@x86_64::@simple_mul_abs_negate[%dim](%dim_i32, %arg0, %arg1) {
      hal.interface.bindings = [
        #hal.interface.binding<0, 0>,
        #hal.interface.binding<0, 1>,
        #hal.interface.binding<0, 2>
      ],
      // HACK: keep the executable live through DCE. Only required when
      // using the automatic variant selection.
      hal.executable.ref = [@executable]
    } : (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}
    return %0 : tensor<?xf32>
  }

  transform.named_sequence @match_mul_abs_negate(%root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<?xf32>, %rhs: tensor<?xf32>):
        // The matcher does not recurse to the constant index + dim because
        // their only consumer matches only the operation name.
        %c0 = arith.constant 0 : index
        %dim = tensor.dim %lhs, %c0 : tensor<?xf32>
        // --------------------------------------------------------------------
        %empty = tensor.empty(%dim) {"match.operation_name_only"} : tensor<?xf32>
        %mul = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                                affine_map<(d0) -> (d0)>,
                                                affine_map<(d0) -> (d0)>],
                               iterator_types = ["parallel"]}
                               ins(%lhs, %rhs : tensor<?xf32>, tensor<?xf32>)
                               outs(%empty : tensor<?xf32>) {
        ^bb0(%in: f32, %in0: f32, %out: f32):
          %m = arith.mulf %in, %in0 : f32
          linalg.yield %m : f32
        } -> tensor<?xf32>
        %abs = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                                affine_map<(d0) -> (d0)>],
                               iterator_types = ["parallel"]}
                               ins(%mul : tensor<?xf32>)
                               outs(%empty : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %a = math.absf %in : f32
          linalg.yield %a : f32
        } -> tensor<?xf32>
        // The payload root is compared starting from here, walking up the chain
        // of producers
        %neg = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                                affine_map<(d0) -> (d0)>],
                               iterator_types = ["parallel"]}
                               ins(%abs : tensor<?xf32>)
                               outs(%empty : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %n = arith.negf %in : f32
          linalg.yield %n : f32
        } -> tensor<?xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }

  // Rewrite callback for `transform.foreach_match`. The input signature for
  // this sequence must match exactly with the outputs of the matcher. In this
  // case the matcher returns the inputs and outputs to the matched dag directly
  // so we just insert a call to the hand authored function above.
  transform.named_sequence @cast_and_call_dag(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.iree.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.iree.import_symbol @executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.iree.import_symbol @call_mul_abs_negate into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.func.cast_and_call %func(%ins) -> %out after %root {
          // This specifies how to resolve type mismatches between the arguments
          // of the function and the inputs from the matcher. In this example,
          // the only casts this will generate are same-rank tensor casts that
          // drop static information.
          transform.type_conversion.tensor.cast_shape_dynamic_dims
      } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Entry point for the transform interpreter, nested on the full module. This
  // is because the rewrites needed for importing the custom kernel needs to
  // add a new symbol to the module's symbol table.
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    // Gather the set of functions within the module.
    %funcs = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op   
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
            // <matcher name> -> <rewriter name>
            // Multiple matcher-action pairs can be specified comma separated,
            // here we are only doing a single kind of match and replace.
            @match_mul_abs_negate -> @cast_and_call_dag
          : (!transform.any_op) -> (!transform.any_op)
    }
    // Cleanup leftover dead code; cast_and_call does not do replacement, only
    // rewires uses.
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
