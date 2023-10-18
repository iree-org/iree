// RUN: iree-opt %s

// The required configuration for the custom dispatch. This tells the compiler
// the requisite target information needed to support the associated custom
// shader.
#spirv_target = #hal.executable.target<"vulkan", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64>
  >
}>

#layout = #hal.pipeline.layout<push_constants = 1, sets = [
  <0, bindings = [
      <0, storage_buffer, ReadOnly>,
      <1, storage_buffer, ReadOnly>,
      <2, storage_buffer>
  ]>
]>
#bindings = [
  #hal.interface.binding<0, 0>,
  #hal.interface.binding<0, 1>,
  #hal.interface.binding<0, 2>
]
#objects = #hal.executable.objects<{
  #spirv_target = [
    #hal.executable.object<{
      path = "samples/custom_dispatch/vulkan/shaders/simple_mul.spv"
    }>
  ]
}>

#attrdict = {
  export = "main",
  layout = #layout,
  bindings = #bindings,
  objects = #objects
}

module {
  pdl.pattern : benefit(1) {
    %lhs_type = pdl.type : tensor<?xf32>
    %rhs_type = pdl.type : tensor<?xf32>
    %out_type = pdl.type : tensor<?xf32>
    %lhs = pdl.operand : %lhs_type
    %rhs = pdl.operand : %rhs_type

    // Match the target operation(s) for rewriting and the original arguments and result types.
    %mul = pdl.operation "arith.mulf" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%out_type : !pdl.type)

    pdl.rewrite %mul {
      // Constant attributes for rewriting. `attrdict` contains the layout/binding/object
      // required for the custom dispatch.
      %attrdict = pdl.attribute = #attrdict
      %tied_operands = pdl.attribute = array<i64>
      %c1_idx = pdl.attribute = 1 : index
      %apply_map = pdl.attribute = affine_map<()[s0] -> (s0 ceildiv 64)>

      %range = pdl.range %lhs : !pdl.value
      %res_type_range = pdl.range %out_type : !pdl.type
      %workload = pdl.apply_native_rewrite "get_tensor_sizes"(%range : !pdl.range<value>) : !pdl.range<value>
      %new_dims = pdl.apply_native_rewrite "convert_index_to_i32"(%workload : !pdl.range<value>) : !pdl.range<value>
      %arg_range = pdl.range %new_dims, %lhs, %rhs : !pdl.range<value>, !pdl.value, !pdl.value
      %arg_dims = pdl.apply_native_rewrite "get_tensor_sizes"(%arg_range : !pdl.range<value>) : !pdl.range<value>

      // Create the extern dispatch op. The workgroup count region has not been constructed at
      // this point.
       %extern = pdl.apply_native_rewrite "create_dispatch_extern"(
                                         %mul, %workload, %res_type_range, %workload,
                                         %arg_range, %arg_dims, %tied_operands,
                                         %attrdict : !pdl.operation, !pdl.range<value>, !pdl.range<type>, !pdl.range<value>,
                                                     !pdl.range<value>, !pdl.range<value>, !pdl.attribute,
                                                     !pdl.attribute) : !pdl.operation

      // Emplace the workgroup count region based on the workload, return handles to the new block
      // arguments, and set the insertion point to the new block.
      %wkg_args = pdl.apply_native_rewrite "emplace_extern_workgroup_count"(%extern : !pdl.operation) : !pdl.range<value>

      // Workgroup count of 1 along y and z.
      %c1_op = pdl.operation "arith.constant" {"value" = %c1_idx}
      %c1 = pdl.result 0 of %c1_op

      // Extract the workload argument; pdl_interp provides an extract op, however it isn't
      // allowed to be used inside a pdl.pattern region, hence this hack.
      %workload_arg = pdl.apply_native_rewrite "extract_value"(%wkg_args, %c1_idx : !pdl.range<value>, !pdl.attribute) : !pdl.value

      // Compute x and create the terminator of the workgroup count region.
      %index_type = pdl.type : index
      %x_op = pdl.operation "affine.apply"(%workload_arg : !pdl.value) {"map" = %apply_map} -> (%index_type : !pdl.type)
      %x = pdl.result 0 of %x_op
      %res = pdl.operation "hal.return"(%x, %c1, %c1 : !pdl.value, !pdl.value, !pdl.value)

      %new_result = pdl.result 0 of %extern
      pdl.replace %mul with (%new_result : !pdl.value)
    }   
  }
}
