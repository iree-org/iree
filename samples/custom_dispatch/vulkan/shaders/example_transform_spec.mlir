// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// The configuration used for executable compilation.
// This specifies the device configurations that support this custom kernel.
#spirv_target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform, GroupNonUniformArithmetic, GroupNonUniformBallot],
                     [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64>
  >
}>

module attributes {transform.with_named_sequence} {
  func.func private @argmax_1d_f32_entry_point(%arg0: tensor<1x?xf32>) -> tensor<1xi64> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<1x?xf32>
    // Note: This is not safe if the dim size exceeds INT32_MAX. To pass a 64
    // bit value it must be broken down into two 32-bit values for the high and
    // low bits.
    %dim_i32 = arith.index_cast %dim : index to i32
    // Inline external dispatch that conforms to the ABI that the kernel
    // requires. This is the primary reason for the surrounding function as
    // details like tensor shape and push constants need to line up after
    // splicing in the custom dispatch. This allows the kernel author to manage
    // such details by hand without needing the rewrite patterns to worry about
    // things like order of push constants.
    %4 = hal.dispatch.extern "main"[%dim](%dim_i32, %arg0) : (i32, tensor<1x?xf32>{%dim}) -> tensor<1xi64>
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %c1_0 = arith.constant 1 : index
        hal.return %c1_0, %c1_0, %c1_0 : index, index, index
      }   
      layout(#hal.pipeline.layout<push_constants = 1, sets = [
        <0, bindings = [
            <0, storage_buffer, ReadOnly>,
            <1, storage_buffer>
        ]>
      ]>)
      bindings([
        #hal.interface.binding<0, 0>, 
        #hal.interface.binding<0, 1>
      ])  
      objects({
        #spirv_target ordinal(0) = [ 
          #hal.executable.object<{
            path = "samples/custom_dispatch/vulkan/shaders/one_workgroup_argmax_subgroup_f32.spv"
          }>
        ]
      })
    return %4 : tensor<1xi64>
  }

  // Custom matcher for argmax operations equivalent to the custom kernel. This
  // matcher will be run one-by-one on all operations contained within the
  // target function. On success, it will return the handle to the matched
  // argmax operation.
  transform.named_sequence @match_argmax(%generic: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    // Fail fast on non-linalg generics.
    transform.match.operation_name %generic ["linalg.generic"] : !transform.any_op
    %matched = transform.match.structured failures(propagate) %generic : (!transform.any_op) -> (!transform.any_op) {
    ^bb1(%argmax: !transform.any_op):
      // Verify that the rank (i.e. number of loops) of the linalg op is 2,
      // with one parallel iterator and one reduction iterator.
      // TODO: Add optionality for the parallel dimensions.
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %argmax : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c2 : !transform.param<i64>
      transform.match.structured.dim %argmax[0] {parallel} : !transform.any_op
      transform.match.structured.dim %argmax[-1] {reduction} : !transform.any_op

      // Verify a single input (target vector to compute the argmax of) and two
      // outputs, one for the maximum value and one for the index.
      %c1 = transform.param.constant 1 : i64 -> !transform.param<i64>
      %n_inputs = transform.match.structured.num_inputs %argmax : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_inputs, %c1 : !transform.param<i64>
      %n_outputs = transform.match.structured.num_inits %argmax : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %n_outputs, %c2 : !transform.param<i64>
  
      transform.match.structured.yield %argmax : !transform.any_op 
    }

    // Verify the operand shapes of the linalg op. For example, in the below,
    // dim 0 must be statically 1, and dim 1 must be statically divisible by 64.
    %in0 = transform.get_operand %matched[0] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %in0 = tensor<1x?xf32> : !transform.any_value
    transform.iree.match.dim_is_multiple_of %in0[1], 64 : !transform.any_value
    %out0 = transform.get_operand %matched[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %out0 = tensor<1xf32> : !transform.any_value
    %out1 = transform.get_operand %matched[2] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %out1 = tensor<1xi64> : !transform.any_value

    // Verify the region of the argmax op. This does a structural comparison of
    // region(s) of the payload operation against the single operation contained
    // within the body of this operation. This does no verification of other
    // input types/attributes. This is because typically for kernel matching,
    // the most important part to get exactly right is the inner loop. Otherwise
    // small variations to shape information and iterator counts and such are
    // better suited for more general matchers.
    transform.iree.match.regions %matched : !transform.any_op {
      ^bb0(%target: tensor<1x?xf32>, %empty_max: tensor<1xf32>, %empty_idx: tensor<1xi64>):
        %5:2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                                affine_map<(d0, d1) -> (d0)>,
                                                affine_map<(d0, d1) -> (d0)>],
                               iterator_types = ["parallel", "reduction"]}
                               ins(%target : tensor<1x?xf32>)
                               outs(%empty_max, %empty_idx : tensor<1xf32>, tensor<1xi64>) {
        ^bb0(%in: f32, %out: f32, %out_0: i64):
          %6 = linalg.index 1 : index
          %7 = arith.index_cast %6 : index to i64
          %8 = arith.maximumf %in, %out : f32
          %9 = arith.cmpf ogt, %in, %out : f32
          %10 = arith.select %9, %7, %out_0 : i64
          linalg.yield %8, %10 : f32, i64
        } -> (tensor<1xf32>, tensor<1xi64>)
    }
    transform.yield %generic : !transform.any_op
  }

  // Rewrite callback for `transform.foreach_match`. The input signature for
  // this sequence must match exactly with the outputs of the matcher. In this
  // case we just take the argmax as an input, import the entry point for the
  // custom kernel authored above, and replace the users of the argmax with a
  // call to the function.
  transform.named_sequence @cast_and_call_argmax(%argmax: !transform.any_op {transform.readonly}) {
    %module = transform.iree.get_nearest_symbol_table %argmax : (!transform.any_op) -> !transform.any_op
    %func = transform.iree.import_symbol @argmax_1d_f32_entry_point into %module : (!transform.any_op) -> !transform.any_op
    %ins = transform.get_operand %argmax[0] : (!transform.any_op) -> !transform.any_value
    %outs = transform.get_result %argmax[1] : (!transform.any_op) -> !transform.any_value
    transform.func.cast_and_call %func(%ins) -> %outs before %argmax {
          // This specifies how to resolve type mismatches between the arguments
          // of the function and the inputs to the argmax. In this example, the
          // only casts this will generate are same-rank tensor casts that drop
          // static information.
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
            //
            // Note that the operations within the module are walked in
            // post-order, meaning actions must be very careful in their
            // replacements not to modify successors of operations. Nested
            // regions and DAG roots will be visited last so it is safest to
            // do matching + replacement on the root of the DAG rather than
            // trying to look ahead. The other option is to avoid dce/cse until
            // after the walk is complete.
            @match_argmax -> @cast_and_call_argmax
          : (!transform.any_op) -> (!transform.any_op)
    }
    // Cleanup now dead instances of argmax.
    transform.apply_dce to %module : !transform.any_op
    transform.yield
  }
}
