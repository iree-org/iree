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

#define EMITC_VLA_IMPORT(non_var_arg_types, var_arg_types, ret_types)   \
  EMITC_VLA_IMPORT_IMPL(                                                \
      non_var_arg_types, var_arg_types, ret_types,                      \
      ARGUMENTS_SIZE(non_var_arg_types), ARGUMENTS_SIZE(var_arg_types), \
      PACK_VARARG_ARGUMENTS(non_var_arg_types, ptr, varargs),           \
      PACK_VARARG_ARGUMENTS(var_arg_types, ptr, varargs),               \
      UNPACK_VARARG_RESULTS(ret_types, varargs), UNPACK_RESULTS(ret_types))

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
    /*const*/ IREE_VM_ABI_TYPE_NAME(arg_types)* args =               \
        iree_vm_abi_##arg_types##_checked_deref(call->arguments);    \
    IREE_VM_ABI_TYPE_NAME(ret_types)* rets =                         \
        iree_vm_abi_##ret_types##_checked_deref(call->results);      \
                                                                     \
    /* TODO(simon-camp): Doesn't work for v 'v' */                   \
    /* \
    if (IREE_UNLIKELY(!args || !rets)) {                             \
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,          \
                              "argument/result signature mismatch"); \
    }                                                                \
    */ \                                                                 
    \
    iree_vm_abi_##ret_types##_reset(rets);                                                             \
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
    IREE_VM_ABI_TYPE_NAME(arg_types) arguments;                            \
    iree_vm_abi_##arg_types##_reset(&arguments);                           \
    pack_arguments;                                                        \
                                                                           \
    IREE_VM_ABI_TYPE_NAME(ret_types) results;                              \
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
    IREE_VM_ABI_TYPE_NAME(ret_types) results;                                \
    iree_vm_abi_##ret_types##_reset(&results);                               \
                                                                             \
    iree_vm_function_call_t call;                                            \
    call.function = *import;                                                 \
    call.arguments.data_length = totalSize;                                  \
    call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length); \
    call.results = iree_make_byte_span(&results, sizeof(results));           \
                                                                             \
    memset(call.arguments.data, 0, call.arguments.data_length);              \
                                                                             \
    uint8_t* ptr = call.arguments.data;                                      \
    va_list varargs;                                                         \
    va_start(varargs, spanCount);                                            \
                                                                             \
    pack_args;                                                               \
    memcpy(ptr, &spanCount, sizeof(int32_t));                                \
    ptr += sizeof(int32_t);                                                  \
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

#define ARGUMENTS_SIZE(types) (ARGUMENTS_SIZE_##types)
#define INPUT_ARGUMENTS(types) INPUT_ARGUMENTS_##types
#define OUTPUT_ARGUMENTS(types) OUTPUT_ARGUMENTS_##types
#define INPUT_PARAMETERS(types) INPUT_PARAMETERS_##types
#define OUTPUT_PARAMETERS(types) OUTPUT_PARAMETERS_##types
#define PACK_ARGUMENTS(types) PACK_ARGUMENTS_##types
#define UNPACK_RESULTS(types) UNPACK_RESULTS_##types
#define PACK_VARARG_ARGUMENTS(types, dest, varargs) \
  PACK_VARARG_ARGUMENTS_##types(dest, varargs)
#define UNPACK_VARARG_RESULTS(types, varargs) \
  UNPACK_VARARG_RESULTS_##types(varargs)

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)

// i
#define ARGUMENT_SIZE_i sizeof(int32_t)
#define INPUT_ARGUMENT_i(n) args->i##n
#define OUTPUT_ARGUMENT_i(n) &rets->i##n
#define INPUT_PARAMETER_i(n) int32_t arg##n
#define OUTPUT_PARAMETER_i(n) int32_t* ret##n
#define PACK_ARGUMENT_i(n) arguments.i##n = arg##n
#define UNPACK_RESULT_i(n) *ret##n = results.i##n
#define PACK_VARARG_i(dest, varargs) \
  PACK_VARARG_i_IMPL(dest, varargs, CONCAT(_temp, __COUNTER__))
#define PACK_VARARG_i_IMPL(dest, varargs, temp) \
  int32_t temp = va_arg(varargs, int32_t);      \
  memcpy(dest, &temp, sizeof(int32_t));         \
  dest += sizeof(int32_t)
#define UNPACK_VARARG_i(varargs, n) \
  OUTPUT_PARAMETER_i(n) = va_arg(varargs, int32_t*)

// r
#define ARGUMENT_SIZE_r sizeof(iree_vm_ref_t)
#define INPUT_ARGUMENT_r(n) &args->r##n
#define OUTPUT_ARGUMENT_r(n) &rets->r##n
#define INPUT_PARAMETER_r(n) iree_vm_ref_t* arg##n
#define OUTPUT_PARAMETER_r(n) iree_vm_ref_t* ret##n
#define PACK_ARGUMENT_r(n) iree_vm_ref_assign(arg##n, &arguments.r##n)
#define UNPACK_RESULT_r(n) iree_vm_ref_move(&results.r##n, ret##n)
#define PACK_VARARG_r(dest, varargs) \
  PACK_VARARG_r_IMPL(dest, varargs, CONCAT(_temp, __COUNTER__))
#define PACK_VARARG_r_IMPL(dest, varargs, temp)          \
  iree_vm_ref_t* temp = va_arg(varargs, iree_vm_ref_t*); \
  iree_vm_ref_assign(temp, (iree_vm_ref_t*)(dest));      \
  dest += sizeof(iree_vm_ref_t)
#define UNPACK_VARARG_r(varargs, n) \
  OUTPUT_PARAMETER_r(n) = va_arg(varargs, iree_vm_ref_t*)

// i
#define ARGUMENTS_SIZE_i ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_i , INPUT_ARGUMENT_i(0)
#define OUTPUT_ARGUMENTS_i , OUTPUT_ARGUMENT_i(0)
#define INPUT_PARAMETERS_i , INPUT_PARAMETER_i(0)
#define OUTPUT_PARAMETERS_i , OUTPUT_PARAMETER_i(0)
#define PACK_ARGUMENTS_i PACK_ARGUMENT_i(0);
#define UNPACK_RESULTS_i UNPACK_RESULT_i(0);
#define PACK_VARARG_ARGUMENTS_i(dest, varargs) PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_i(varargs) UNPACK_VARARG_i(varargs, 0);

// irii
#define ARGUMENTS_SIZE_irii \
  ARGUMENT_SIZE_i + ARGUMENT_SIZE_r + ARGUMENT_SIZE_i + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_irii                                       \
  , INPUT_ARGUMENT_i(0), INPUT_ARGUMENT_r(1), INPUT_ARGUMENT_i(2), \
      INPUT_ARGUMENT_i(3)
#define OUTPUT_ARGUMENTS_irii                                         \
  , OUTPUT_ARGUMENT_i(0), OUTPUT_ARGUMENT_r(1), OUTPUT_ARGUMENT_i(2), \
      OUTPUT_ARGUMENT_i(3)
#define INPUT_PARAMETERS_irii                                         \
  , INPUT_PARAMETER_i(0), INPUT_PARAMETER_r(1), INPUT_PARAMETER_i(2), \
      INPUT_PARAMETER_i(3)
#define OUTPUT_PARAMETERS_iri                                            \
  , OUTPUT_PARAMETER_i(0), OUTPUT_PARAMETER_r(1), OUTPUT_PARAMETER_i(2), \
      OUTPUT_PARAMETER_i(3)
#define PACK_ARGUMENTS_irii \
  PACK_ARGUMENT_i(0);       \
  PACK_ARGUMENT_r(1);       \
  PACK_ARGUMENT_i(2);       \
  PACK_ARGUMENT_i(3);
#define UNPACK_RESULTS_irii \
  UNPACK_RESULT_i(0);       \
  UNPACK_RESULT_r(1);       \
  UNPACK_RESULT_i(2);       \
  UNPACK_RESULT_i(3);
#define PACK_VARARG_ARGUMENTS_irii(dest, varargs) \
  PACK_VARARG_i(dest, varargs);                   \
  PACK_VARARG_r(dest, varargs);                   \
  PACK_VARARG_i(dest, varargs);                   \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_irii(varargs) \
  UNPACK_VARARG_i(varargs, 0);              \
  UNPACK_VARARG_r(varargs, 1);              \
  UNPACK_VARARG_i(varargs, 2);              \
  UNPACK_VARARG_i(varargs, 2);

// ii
#define ARGUMENTS_SIZE_ii ARGUMENT_SIZE_i + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_ii , INPUT_ARGUMENT_i(0), INPUT_ARGUMENT_i(1)
#define OUTPUT_ARGUMENTS_ii , OUTPUT_ARGUMENT_i(0), OUTPUT_ARGUMENT_i(1)
#define INPUT_PARAMETERS_ii , INPUT_PARAMETER_i(0), INPUT_PARAMETER_i(1)
#define OUTPUT_PARAMETERS_ii , OUTPUT_PARAMETER_i(0), OUTPUT_PARAMETER_i(1)
#define PACK_ARGUMENTS_ii \
  PACK_ARGUMENT_i(0);     \
  PACK_ARGUMENT_i(1);
#define UNPACK_RESULTS_ii \
  UNPACK_RESULT_i(0);     \
  UNPACK_RESULT_i(1);
#define PACK_VARARG_ARGUMENTS_ii(dest, varargs) \
  PACK_VARARG_i(dest, varargs);                 \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_ii(varargs) \
  UNPACK_VARARG_i(varargs, 0);            \
  UNPACK_VARARG_i(varargs, 1);

// iii
#define ARGUMENTS_SIZE_iii ARGUMENT_SIZE_i + ARGUMENT_SIZE_i + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_iii \
  , INPUT_ARGUMENT_i(0), INPUT_ARGUMENT_i(1), INPUT_ARGUMENT_i(2)
#define OUTPUT_ARGUMENTS_iii \
  , OUTPUT_ARGUMENT_i(0), OUTPUT_ARGUMENT_i(1), OUTPUT_ARGUMENT_i(2)
#define INPUT_PARAMETERS_iii \
  , INPUT_PARAMETER_i(0), INPUT_PARAMETER_i(1), INPUT_PARAMETER_i(2)
#define OUTPUT_PARAMETERS_iii \
  , OUTPUT_PARAMETER_i(0), OUTPUT_PARAMETER_i(1), OUTPUT_PARAMETER_i(2)
#define PACK_ARGUMENTS_iii \
  PACK_ARGUMENT_i(0);      \
  PACK_ARGUMENT_i(1);      \
  PACK_ARGUMENT_i(2);
#define UNPACK_RESULTS_iii \
  UNPACK_RESULT_i(0);      \
  UNPACK_RESULT_i(1);      \
  UNPACK_RESULT_i(2);
#define PACK_VARARG_ARGUMENTS_iii(dest, varargs) \
  PACK_VARARG_i(dest, varargs);                  \
  PACK_VARARG_i(dest, varargs);                  \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_iii(varargs) \
  UNPACK_VARARG_i(varargs, 0);             \
  UNPACK_VARARG_i(varargs, 1);             \
  UNPACK_VARARG_i(varargs, 2);

// r
#define ARGUMENTS_SIZE_r ARGUMENT_SIZE_r
#define INPUT_ARGUMENTS_r , INPUT_ARGUMENT_r(0)
#define OUTPUT_ARGUMENTS_r , OUTPUT_ARGUMENT_r(0)
#define INPUT_PARAMETERS_r , INPUT_PARAMETER_r(0)
#define OUTPUT_PARAMETERS_r , OUTPUT_PARAMETER_r(0)
#define PACK_ARGUMENTS_r PACK_ARGUMENT_r(0);
#define UNPACK_RESULTS_r UNPACK_RESULT_r(0);
#define PACK_VARARG_ARGUMENTS_r(dest, varargs) PACK_VARARG_r(dest, varargs);
#define UNPACK_VARARG_RESULTS_r(varargs) UNPACK_VARARG_r(varargs, 0);

// ri
#define ARGUMENTS_SIZE_ri ARGUMENT_SIZE_r + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_ri , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_i(1)
#define OUTPUT_ARGUMENTS_ri , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_i(1)
#define INPUT_PARAMETERS_ri , INPUT_PARAMETER_r(0), INPUT_PARAMETER_i(1)
#define OUTPUT_PARAMETERS_ri , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_i(1)
#define PACK_ARGUMENTS_ri \
  PACK_ARGUMENT_r(0);     \
  PACK_ARGUMENT_i(1);
#define UNPACK_RESULTS_ri \
  UNPACK_RESULT_r(0);     \
  UNPACK_RESULT_i(1);
#define PACK_VARARG_ARGUMENTS_ri(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                 \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_ri(varargs) \
  UNPACK_VARARG_r(varargs, 0);            \
  UNPACK_VARARG_i(varargs, 1);

// rii
#define ARGUMENTS_SIZE_rii ARGUMENT_SIZE_r + ARGUMENT_SIZE_i + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_rii \
  , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_i(1), INPUT_ARGUMENT_i(2)
#define OUTPUT_ARGUMENTS_rii \
  , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_i(1), OUTPUT_ARGUMENT_i(2)
#define INPUT_PARAMETERS_rii \
  , INPUT_PARAMETER_r(0), INPUT_PARAMETER_i(1), INPUT_PARAMETER_i(2)
#define OUTPUT_PARAMETERS_rii \
  , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_i(1), OUTPUT_PARAMETER_i(2)
#define PACK_ARGUMENTS_rii \
  PACK_ARGUMENT_r(0);      \
  PACK_ARGUMENT_i(1);      \
  PACK_ARGUMENT_i(2);
#define UNPACK_RESULTS_rii \
  UNPACK_RESULT_r(0);      \
  UNPACK_RESULT_i(1);      \
  UNPACK_RESULT_i(2);
#define PACK_VARARG_ARGUMENTS_rii(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                  \
  PACK_VARARG_i(dest, varargs);                  \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_rii(varargs) \
  UNPACK_VARARG_r(varargs, 0);             \
  UNPACK_VARARG_i(varargs, 1);             \
  UNPACK_VARARG_i(varargs, 2);

// riii
#define ARGUMENTS_SIZE_riii \
  ARGUMENT_SIZE_r + ARGUMENT_SIZE_i + ARGUMENT_SIZE_i + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_riii                                       \
  , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_i(1), INPUT_ARGUMENT_i(2), \
      INPUT_ARGUMENT_i(3)
#define OUTPUT_ARGUMENTS_riii                                         \
  , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_i(1), OUTPUT_ARGUMENT_i(2), \
      OUTPUT_ARGUMENT_i(3)
#define INPUT_PARAMETERS_riii                                         \
  , INPUT_PARAMETER_r(0), INPUT_PARAMETER_i(1), INPUT_PARAMETER_i(2), \
      INPUT_PARAMETER_i(3)
#define OUTPUT_PARAMETERS_riii                                           \
  , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_i(1), OUTPUT_PARAMETER_i(2), \
      OUTPUT_PARAMETER_i(3)
#define PACK_ARGUMENTS_riii \
  PACK_ARGUMENT_r(0);       \
  PACK_ARGUMENT_i(1);       \
  PACK_ARGUMENT_i(2);       \
  PACK_ARGUMENT_i(3);
#define UNPACK_RESULTS_riii \
  UNPACK_RESULT_r(0);       \
  UNPACK_RESULT_i(1);       \
  UNPACK_RESULT_i(2);       \
  UNPACK_RESULT_i(3);
#define PACK_VARARG_ARGUMENTS_riii(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                   \
  PACK_VARARG_i(dest, varargs);                   \
  PACK_VARARG_i(dest, varargs);                   \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_riii(varargs) \
  UNPACK_VARARG_r(varargs, 0);              \
  UNPACK_VARARG_i(varargs, 1);              \
  UNPACK_VARARG_i(varargs, 2);              \
  UNPACK_VARARG_i(varargs, 3);

// rr
#define ARGUMENTS_SIZE_rr ARGUMENT_SIZE_r + ARGUMENT_SIZE_r
#define INPUT_ARGUMENTS_rr , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_r(1)
#define OUTPUT_ARGUMENTS_rr , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_r(1)
#define INPUT_PARAMETERS_rr , INPUT_PARAMETER_r(0), INPUT_PARAMETER_r(1)
#define OUTPUT_PARAMETERS_rr , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_r(1)
#define PACK_ARGUMENTS_rr \
  PACK_ARGUMENT_r(0);     \
  PACK_ARGUMENT_r(1);
#define UNPACK_RESULTS_rr \
  UNPACK_RESULT_r(0);     \
  UNPACK_RESULT_r(1);
#define PACK_VARARG_ARGUMENTS_rr(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                 \
  PACK_VARARG_r(dest, varargs);
#define UNPACK_VARARG_RESULTS_rr(varargs) \
  UNPACK_VARARG_r(varargs, 0);            \
  UNPACK_VARARG_r(varargs, 1);

// rri
#define ARGUMENTS_SIZE_rri ARGUMENT_SIZE_r + ARGUMENT_SIZE_r + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_rri \
  , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_r(1), INPUT_ARGUMENT_i(2)
#define OUTPUT_ARGUMENTS_rri \
  , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_r(1), OUTPUT_ARGUMENT_i(2)
#define INPUT_PARAMETERS_rri \
  , INPUT_PARAMETER_r(0), INPUT_PARAMETER_r(1), INPUT_PARAMETER_i(2)
#define OUTPUT_PARAMETERS_rri \
  , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_r(1), OUTPUT_PARAMETER_i(2)
#define PACK_ARGUMENTS_rri \
  PACK_ARGUMENT_r(0);      \
  PACK_ARGUMENT_r(1);      \
  PACK_ARGUMENT_i(2);
#define UNPACK_RESULTS_rri \
  UNPACK_RESULT_r(0);      \
  UNPACK_RESULT_r(1);      \
  UNPACK_RESULT_i(2);
#define PACK_VARARG_ARGUMENTS_rri(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                  \
  PACK_VARARG_r(dest, varargs);                  \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_rri(varargs) \
  UNPACK_VARARG_r(varargs, 0);             \
  UNPACK_VARARG_r(varargs, 1);             \
  UNPACK_VARARG_i(varargs, 2);

// rriiii
#define ARGUMENTS_SIZE_rriiii                                             \
  ARGUMENT_SIZE_r + ARGUMENT_SIZE_r + ARGUMENT_SIZE_i + ARGUMENT_SIZE_i + \
      ARGUMENT_SIZE_i + ARGUMENT_SIZE_i
#define INPUT_ARGUMENTS_rriiii                                     \
  , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_r(1), INPUT_ARGUMENT_i(2), \
      INPUT_ARGUMENT_i(3), INPUT_ARGUMENT_i(4), INPUT_ARGUMENT_i(5)
#define OUTPUT_ARGUMENTS_rriiii                                       \
  , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_r(1), OUTPUT_ARGUMENT_i(2), \
      OUTPUT_ARGUMENT_i(3), OUTPUT_ARGUMENT_i(4), OUTPUT_ARGUMENT_i(5)
#define INPUT_PARAMETERS_rriiii                                       \
  , INPUT_PARAMETER_r(0), INPUT_PARAMETER_r(1), INPUT_PARAMETER_i(2), \
      INPUT_PARAMETER_i(3), INPUT_PARAMETER_i(4), INPUT_PARAMETER_i(5)
#define OUTPUT_PARAMETERS_rriiii                                         \
  , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_r(1), OUTPUT_PARAMETER_i(2), \
      OUTPUT_PARAMETER_i(3), OUTPUT_PARAMETER_i(4), OUTPUT_PARAMETER_i(5)
#define PACK_ARGUMENTS_rriiii \
  PACK_ARGUMENT_r(0);         \
  PACK_ARGUMENT_r(1);         \
  PACK_ARGUMENT_i(2);         \
  PACK_ARGUMENT_i(3);         \
  PACK_ARGUMENT_i(4);         \
  PACK_ARGUMENT_i(5);
#define UNPACK_RESULTS_rriiii \
  UNPACK_RESULT_r(0);         \
  UNPACK_RESULT_r(1);         \
  UNPACK_RESULT_i(2);         \
  UNPACK_RESULT_i(3);         \
  UNPACK_RESULT_i(4);         \
  UNPACK_RESULT_i(5);
#define PACK_VARARG_ARGUMENTS_rriiii(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                     \
  PACK_VARARG_r(dest, varargs);                     \
  PACK_VARARG_i(dest, varargs);                     \
  PACK_VARARG_i(dest, varargs);                     \
  PACK_VARARG_i(dest, varargs);                     \
  PACK_VARARG_i(dest, varargs);
#define UNPACK_VARARG_RESULTS_rriiii(varargs) \
  UNPACK_VARARG_r(varargs, 0);                \
  UNPACK_VARARG_r(varargs, 1);                \
  UNPACK_VARARG_i(varargs, 2);                \
  UNPACK_VARARG_i(varargs, 3);                \
  UNPACK_VARARG_i(varargs, 4);                \
  UNPACK_VARARG_i(varargs, 5);

// rrr
#define ARGUMENTS_SIZE_rrr ARGUMENT_SIZE_r + ARGUMENT_SIZE_r + ARGUMENT_SIZE_r
#define INPUT_ARGUMENTS_rrr \
  , INPUT_ARGUMENT_r(0), INPUT_ARGUMENT_r(1), INPUT_ARGUMENT_r(2)
#define OUTPUT_ARGUMENTS_rrr \
  , OUTPUT_ARGUMENT_r(0), OUTPUT_ARGUMENT_r(1), OUTPUT_ARGUMENT_r(2)
#define INPUT_PARAMETERS_rrr \
  , INPUT_PARAMETER_r(0), INPUT_PARAMETER_r(1), INPUT_PARAMETER_r(2)
#define OUTPUT_PARAMETERS_rrr \
  , OUTPUT_PARAMETER_r(0), OUTPUT_PARAMETER_r(1), OUTPUT_PARAMETER_r(2)
#define PACK_ARGUMENTS_rrr \
  PACK_ARGUMENT_r(0);      \
  PACK_ARGUMENT_r(1);      \
  PACK_ARGUMENT_r(2);
#define UNPACK_RESULTS_rrr \
  UNPACK_RESULT_r(0);      \
  UNPACK_RESULT_r(1);      \
  UNPACK_RESULT_r(2);
#define PACK_VARARG_ARGUMENTS_rrr(dest, varargs) \
  PACK_VARARG_r(dest, varargs);                  \
  PACK_VARARG_r(dest, varargs);                  \
  PACK_VARARG_r(dest, varargs);
#define UNPACK_VARARG_RESULTS_rrr(varargs) \
  UNPACK_VARARG_r(varargs, 0);             \
  UNPACK_VARARG_r(varargs, 1);             \
  UNPACK_VARARG_r(varargs, 2);

// v
#define ARGUMENTS_SIZE_v 0
#define INPUT_ARGUMENTS_v
#define OUTPUT_ARGUMENTS_v
#define INPUT_PARAMETERS_v
#define OUTPUT_PARAMETERS_v
#define PACK_ARGUMENTS_v
#define UNPACK_RESULTS_v
#define PACK_VARARG_ARGUMENTS_v(dest, varargs)
#define UNPACK_VARARG_RESULTS_v(varargs)

#endif  // Meh...

EMITC_DEFINE_SHIMS(i, i)
EMITC_DEFINE_SHIMS(ii, i)
EMITC_DEFINE_SHIMS(r, r)
EMITC_DEFINE_SHIMS(r, v)
EMITC_DEFINE_SHIMS(rii, r)
EMITC_DEFINE_SHIMS(riii, r)
EMITC_DEFINE_SHIMS(riii, v)
EMITC_DEFINE_SHIMS(rriiii, v)
EMITC_DEFINE_SHIMS(rr, r)
EMITC_DEFINE_SHIMS(rr, v)
EMITC_DEFINE_SHIMS(rrr, ii)
EMITC_DEFINE_SHIMS(v, i)
EMITC_DEFINE_SHIMS(v, r)
EMITC_DEFINE_SHIMS(v, v)

EMITC_VLA_IMPORT(ri, iii, r)
EMITC_VLA_IMPORT(ri, r, r)
EMITC_VLA_IMPORT(rii, i, r)
EMITC_VLA_IMPORT(rri, irii, v)
EMITC_VLA_IMPORT(rrr, r, r)

#endif  // IREE_VM_SHIMS_EMITC_H_
