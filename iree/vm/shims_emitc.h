// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_SHIMS_EMITC_H_
#define IREE_VM_SHIMS_EMITC_H_

#include "iree/base/attributes.h"
#include "iree/vm/module.h"
#include "iree/vm/shims.h"
#include "iree/vm/stack.h"

// see calling convention in module.h

#define EMITC_DEFINE_SHIMS(arg_types, ret_types) \
  EMITC_FIXED_TYPEDEF(arg_types, ret_types)      \
  EMITC_FIXED_SHIM(arg_types, ret_types)         \
  EMITC_FIXED_IMPORT(arg_types, ret_types)

#define EMITC_FIXED_TYPEDEF(arg_types, ret_types)                             \
  EMITC_FIXED_TYPEDEF_IMPL(arg_types, ret_types, INPUT_PARAMETERS(arg_types), \
                           OUTPUT_PARAMETERS(ret_types))

#define EMITC_FIXED_SHIM(arg_types, ret_types)                            \
  EMITC_FIXED_SHIM_IMPL(arg_types, ret_types, INPUT_ARGUMENTS(arg_types), \
                        OUTPUT_ARGUMENTS(ret_types))

#define EMITC_FIXED_IMPORT(arg_types, ret_types)                             \
  EMITC_FIXED_IMPORT_IMPL(arg_types, ret_types, INPUT_PARAMETERS(arg_types), \
                          OUTPUT_PARAMETERS(ret_types),                      \
                          PACK_ARGUMENTS(arg_types),                         \
                          UNPACK_RESULTS(ret_types))

#define EMITC_VLA_IMPORT(non_var_arg_types, var_arg_types, ret_types) \
  EMITC_VLA_IMPORT_IMPL(                                              \
      non_var_arg_types, var_arg_types, ret_types,                    \
      ARGUMENT_PACK_SIZE(non_var_arg_types),                          \
      ARGUMENT_PACK_SIZE(var_arg_types),                              \
      PACK_VARARG_ARGUMENTS(non_var_arg_types, ptr, varargs),         \
      PACK_VARARG_ARGUMENTS(var_arg_types, ptr, varargs),             \
      DEFINE_VARARG_RESULTS(ret_types, varargs), UNPACK_RESULTS(ret_types))

#define EMITC_FIXED_TYPEDEF_IMPL(arg_types, ret_types, input_parameters, \
                                 output_parameters)                      \
  typedef iree_status_t (*call_0##arg_types##_##ret_types##_t)(          \
      iree_vm_stack_t * IREE_RESTRICT stack, void* IREE_RESTRICT module, \
      void* IREE_RESTRICT module_state input_parameters output_parameters);

#define EMITC_FIXED_SHIM_IMPL(arg_types, ret_types, input_arguments, \
                              output_arguments)                      \
  static iree_status_t call_0##arg_types##_##ret_types##_shim(       \
      iree_vm_stack_t* IREE_RESTRICT stack,                          \
      const iree_vm_function_call_t* IREE_RESTRICT call,             \
      call_0##arg_types##_##ret_types##_t target_fn,                 \
      void* IREE_RESTRICT module, void* IREE_RESTRICT module_state,  \
      iree_vm_execution_result_t* IREE_RESTRICT out_result) {        \
    const iree_vm_abi_##arg_types##_t* args =                        \
        iree_vm_abi_##arg_types##_checked_deref(call->arguments);    \
    iree_vm_abi_##ret_types##_t* rets =                              \
        iree_vm_abi_##ret_types##_checked_deref(call->results);      \
                                                                     \
    if (IREE_UNLIKELY(!args || !rets)) {                             \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,          \
                              "argument/result signature mismatch"); \
    }                                                                \
                                                                     \
    iree_vm_abi_##ret_types##_reset(rets);                           \
    return target_fn(stack, module,                                  \
                     module_state input_arguments output_arguments); \
  }

#define EMITC_FIXED_IMPORT_IMPL(arg_types, ret_types, input_parameters,    \
                                output_parameters, pack_arguments,         \
                                unpack_results)                            \
  static iree_status_t call_0##arg_types##_##ret_types##_import(           \
      iree_vm_stack_t* IREE_RESTRICT stack,                                \
      const iree_vm_function_t* IREE_RESTRICT import input_parameters      \
          output_parameters) {                                             \
    iree_vm_abi_##arg_types##_t arguments;                                 \
    pack_arguments;                                                        \
                                                                           \
    iree_vm_abi_##ret_types##_t results;                                   \
    iree_vm_abi_##ret_types##_reset(&results);                             \
                                                                           \
    iree_vm_function_call_t call;                                          \
    call.function = *import;                                               \
    call.arguments = iree_make_byte_span(&arguments, sizeof(arguments));   \
    call.results = iree_make_byte_span(&results, sizeof(results));         \
                                                                           \
    iree_vm_execution_result_t result;                                     \
    memset(&result, 0, sizeof(result));                                    \
                                                                           \
    iree_status_t status =                                                 \
        import->module->begin_call(import->module, stack, &call, &result); \
                                                                           \
    unpack_results;                                                        \
                                                                           \
    return status;                                                         \
  }

#define EMITC_VLA_IMPORT_IMPL(non_var_arg_types, var_arg_types, ret_types,   \
                              non_var_arg_size, var_arg_size, pack_args,     \
                              pack_var_args, define_results, unpack_results) \
  static iree_status_t                                                       \
      call_0##non_var_arg_types##C##var_arg_types##D_##ret_types##_import(   \
          iree_vm_stack_t* IREE_RESTRICT stack,                              \
          const iree_vm_function_t* IREE_RESTRICT import, int32_t spanCount, \
          ...) {                                                             \
    int32_t totalSize =                                                      \
        non_var_arg_size + sizeof(int32_t) + spanCount * var_arg_size;       \
                                                                             \
    iree_vm_abi_##ret_types##_t results;                                     \
    iree_vm_abi_##ret_types##_reset(&results);                               \
                                                                             \
    iree_vm_function_call_t call;                                            \
    call.function = *import;                                                 \
    call.arguments =                                                         \
        iree_make_byte_span((uint8_t*)iree_alloca(totalSize), totalSize);    \
    call.results = iree_make_byte_span(&results, sizeof(results));           \
                                                                             \
    memset(call.arguments.data, 0, call.arguments.data_length);              \
                                                                             \
    uint8_t* ptr = call.arguments.data;                                      \
    va_list varargs;                                                         \
    va_start(varargs, spanCount);                                            \
                                                                             \
    pack_args;                                                               \
    for (int i = 0; i < spanCount; i++) {                                    \
      pack_var_args                                                          \
    }                                                                        \
    define_results;                                                          \
                                                                             \
    va_end(varargs);                                                         \
                                                                             \
    iree_vm_execution_result_t result;                                       \
    memset(&result, 0, sizeof(result));                                      \
                                                                             \
    iree_status_t status =                                                   \
        import->module->begin_call(import->module, stack, &call, &result);   \
                                                                             \
    unpack_results;                                                          \
                                                                             \
    return status;                                                           \
  }

#if 1  // TODO(simon-camp): Meh... We need to hardcode every "argument/result
       // pack" (or use a foreach macro)

#define ARGUMENT_PACK_SIZE(types) (ARGUMENT_PACK_SIZE_##types)
#define PACK_ARGUMENTS(types) PACK_ARGUMENTS_##types
#define UNPACK_RESULTS(types) UNPACK_RESULTS_##types
#define INPUT_ARGUMENTS(types) INPUT_ARGUMENTS_##types
#define OUTPUT_ARGUMENTS(types) OUTPUT_ARGUMENTS_##types
#define INPUT_PARAMETERS(types) INPUT_PARAMETERS_##types
#define OUTPUT_PARAMETERS(types) OUTPUT_PARAMETERS_##types
#define PACK_VARARG_ARGUMENTS(types, dest, varargs) \
  PACK_VARARG_ARGUMENTS_##types(dest, varargs)
#define DEFINE_VARARG_RESULTS(types, varargs) \
  DEFINE_VARARG_RESULTS_##types(varargs)

#define INPUT_ARGUMENTi(n) args->i##n
#define OUTPUT_ARGUMENT_i(n) &rets->i##n
#define PACK_ARGUMENT_i(n) arguments.i##n = arg##n
#define INPUT_PARAMETER_i(n) int32_t arg##n
#define OUTPUT_PARAMETER_i(n) int32_t* ret##n
#define ARGUMENT_SIZE_i sizeof(int32_t)
#define UNPACK_RESULT_i(n) *ret##n = results.i##n

#define COPY_VARARG_r(dest, varargs)                                           \
  iree_vm_ref_assign(va_arg(varargs, iree_vm_ref_t*), (iree_vm_ref_t*)(dest)); \
  dest += sizeof(iree_vm_ref_t);

#define UNPACK_VARARG_r(varargs, n) \
  OUTPUT_PARAMETER_r(n) = va_arg(varargs, iree_vm_ref_t*);
#define INPUT_PARAMETER_r(n) iree_vm_ref_t* arg##n
#define OUTPUT_PARAMETER_r(n) iree_vm_ref_t* ret##n
#define ARGUMENT_SIZE_r sizeof(iree_vm_ref_t)
#define UNPACK_RESULTr(n) iree_vm_ref_move(&results.r##n, ret##n);

#define ARGUMENT_PACK_SIZE_i ARGUMENT_SIZE_i
#define PACK_ARGUMENTS_i PACK_ARGUMENT_i(0);
#define UNPACK_RESULTS_i UNPACK_RESULT_i(0);
#define INPUT_ARGUMENTS_i , INPUT_ARGUMENTi(0)
#define OUTPUT_ARGUMENTS_i , OUTPUT_ARGUMENT_i(0)
#define INPUT_PARAMETERS_i , INPUT_PARAMETER_i(0)
#define OUTPUT_PARAMETERS_i , OUTPUT_PARAMETER_i(0)

#define PACK_ARGUMENTS_ii \
  PACK_ARGUMENT_i(0);     \
  PACK_ARGUMENT_i(1);
#define UNPACK_RESULTS_ii \
  UNPACK_RESULT_i(0);     \
  UNPACK_RESULT_i(1);
#define INPUT_ARGUMENTS_ii , INPUT_ARGUMENTi(0), INPUT_ARGUMENTi(1)
#define OUTPUT_ARGUMENTS_ii , OUTPUT_ARGUMENT_i(0), OUTPUT_ARGUMENT_i(1)
#define INPUT_PARAMETERS_ii , INPUT_PARAMETER_i(0), INPUT_PARAMETER_i(1)
#define OUTPUT_PARAMETERS_ii , OUTPUT_PARAMETER_i(0), OUTPUT_PARAMETER_i(1)

#define PACK_VARARG_ARGUMENTS_r(dest, varargs) COPY_VARARG_r(dest, varargs);
#define DEFINE_VARARG_RESULTS_r(varargs) UNPACK_VARARG_r(varargs, 0);
#define ARGUMENT_PACK_SIZE_r ARGUMENT_SIZE_r
#define PACK_ARGUMENTS_r PACK_ARGUMENT_r(0);
#define UNPACK_RESULTS_r UNPACK_RESULTr(0);
#define INPUT_ARGUMENTS_r , INPUT_ARGUMENTr(0)
#define OUTPUT_ARGUMENTS_r , OUTPUT_ARGUMENT_r(0)
#define INPUT_PARAMETERS_r , INPUT_PARAMETER_r(0)
#define OUTPUT_PARAMETERS_r , OUTPUT_PARAMETER_r(0)

#define PACK_VARARG_ARGUMENTS_rrr(dest, varargs) \
  COPY_VARARG_r(dest, varargs);                  \
  COPY_VARARG_r(dest, varargs);                  \
  COPY_VARARG_r(dest, varargs);
#define ARGUMENT_PACK_SIZE_rrr \
  ARGUMENT_SIZE_r + ARGUMENT_SIZE_r + ARGUMENT_SIZE_r

#define PACK_ARGUMENTS_v
#define UNPACK_RESULTS_v
#define INPUT_ARGUMENTS_v
#define OUTPUT_ARGUMENTS_v
#define INPUT_PARAMETERS_v
#define OUTPUT_PARAMETERS_v

#endif  // Meh...

EMITC_DEFINE_SHIMS(v, v)
EMITC_DEFINE_SHIMS(v, i)
EMITC_DEFINE_SHIMS(i, i)
EMITC_DEFINE_SHIMS(ii, i)

EMITC_VLA_IMPORT(rrr, r, r)

#endif  // IREE_VM_SHIMS_EMITC_H_
