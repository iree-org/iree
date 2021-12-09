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

#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define CONCAT_IMPL(a, b) a##b
#define TUPLE_UNPACK(...) TUPLE_UNPACK_IMPL __VA_ARGS__
#define TUPLE_UNPACK_IMPL(...) __VA_ARGS__

#define NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define NARGS(...) NTH_ARG(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1)

#define INC(n) INC_##n
#define INC_0 1
#define INC_1 2
#define INC_2 3
#define INC_3 4
#define INC_4 5
#define INC_5 6
#define INC_6 7
#define INC_7 8
#define INC_8 9
#define INC_9 10
#define INC_10 11

#define JOIN(...) JOIN_IMPL(__VA_ARGS__)
#define JOIN_IMPL(...) CONCAT(JOIN_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define JOIN_0(...)
#define JOIN_1(a, ...) CONCAT(a, JOIN_0(__VA_ARGS__))
#define JOIN_2(a, ...) CONCAT(a, JOIN_1(__VA_ARGS__))
#define JOIN_3(a, ...) CONCAT(a, JOIN_2(__VA_ARGS__))
#define JOIN_4(a, ...) CONCAT(a, JOIN_3(__VA_ARGS__))
#define JOIN_5(a, ...) CONCAT(a, JOIN_4(__VA_ARGS__))
#define JOIN_6(a, ...) CONCAT(a, JOIN_5(__VA_ARGS__))
#define JOIN_7(a, ...) CONCAT(a, JOIN_6(__VA_ARGS__))
#define JOIN_8(a, ...) CONCAT(a, JOIN_7(__VA_ARGS__))
#define JOIN_9(a, ...) CONCAT(a, JOIN_8(__VA_ARGS__))
#define JOIN_10(a, ...) CONCAT(a, JOIN_9(__VA_ARGS__))

// call f(idx, arg) for each argument
#define FOR_EACH(f, ...) FOR_EACH_IMPL(f, NARGS(__VA_ARGS__), __VA_ARGS__)
#define FOR_EACH_IMPL(f, n, ...) CONCAT(FOR_EACH_, n)(f, 0, __VA_ARGS__)
#define FOR_EACH_0(f, ...)
#define FOR_EACH_1(f, i, a, ...) f(i, a) FOR_EACH_0(f, INC(i), __VA_ARGS__)
#define FOR_EACH_2(f, i, a, ...) f(i, a) FOR_EACH_1(f, INC(i), __VA_ARGS__)
#define FOR_EACH_3(f, i, a, ...) f(i, a) FOR_EACH_2(f, INC(i), __VA_ARGS__)
#define FOR_EACH_4(f, i, a, ...) f(i, a) FOR_EACH_3(f, INC(i), __VA_ARGS__)
#define FOR_EACH_5(f, i, a, ...) f(i, a) FOR_EACH_4(f, INC(i), __VA_ARGS__)
#define FOR_EACH_6(f, i, a, ...) f(i, a) FOR_EACH_5(f, INC(i), __VA_ARGS__)
#define FOR_EACH_7(f, i, a, ...) f(i, a) FOR_EACH_6(f, INC(i), __VA_ARGS__)
#define FOR_EACH_8(f, i, a, ...) f(i, a) FOR_EACH_7(f, INC(i), __VA_ARGS__)
#define FOR_EACH_9(f, i, a, ...) f(i, a) FOR_EACH_8(f, INC(i), __VA_ARGS__)
#define FOR_EACH_10(f, i, a, ...) f(i, a) FOR_EACH_9(f, INC(i), __VA_ARGS__)

#define TYPE_JOIN(types) JOIN(TUPLE_UNPACK(types))

#define EMITC_DEFINE_SHIMS(arg_types, ret_types)                  \
  EMITC_FIXED_TYPEDEF(arg_types, TYPE_JOIN(arg_types), ret_types, \
                      TYPE_JOIN(ret_types))                       \
  EMITC_FIXED_SHIM(arg_types, TYPE_JOIN(arg_types), ret_types,    \
                   TYPE_JOIN(ret_types))                          \
  EMITC_FIXED_IMPORT(arg_types, TYPE_JOIN(arg_types), ret_types,  \
                     TYPE_JOIN(ret_types))

#define EMITC_FIXED_TYPEDEF(arg_types, arg_types_string, ret_types, \
                            ret_types_string)                       \
  EMITC_FIXED_TYPEDEF_IMPL(arg_types_string, ret_types_string,      \
                           INPUT_PARAMETERS(arg_types),             \
                           OUTPUT_PARAMETERS(ret_types))

#define EMITC_FIXED_SHIM(arg_types, arg_types_string, ret_types, \
                         ret_types_string)                       \
  EMITC_FIXED_SHIM_IMPL(arg_types_string, ret_types_string,      \
                        INPUT_ARGUMENTS(arg_types),              \
                        OUTPUT_ARGUMENTS(ret_types))

#define EMITC_FIXED_IMPORT(arg_types, arg_types_string, ret_types,     \
                           ret_types_string)                           \
  EMITC_FIXED_IMPORT_IMPL(                                             \
      arg_types_string, ret_types_string, INPUT_PARAMETERS(arg_types), \
      OUTPUT_PARAMETERS(ret_types), PACK_ARGUMENTS(arg_types),         \
      UNPACK_RESULTS(ret_types))

#define EMITC_VLA_IMPORT(non_var_arg_types, var_arg_types, ret_types)        \
  EMITC_VLA_IMPORT_INDIRECT(non_var_arg_types, TYPE_JOIN(non_var_arg_types), \
                            var_arg_types, TYPE_JOIN(var_arg_types),         \
                            ret_types, TYPE_JOIN(ret_types))
#define EMITC_VLA_IMPORT_INDIRECT(non_var_arg_types, non_var_arg_types_string, \
                                  var_arg_types, var_arg_types_string,         \
                                  ret_types, ret_types_string)                 \
  EMITC_VLA_IMPORT_IMPL(                                                       \
      non_var_arg_types_string, var_arg_types_string, ret_types_string,        \
      ARGUMENTS_SIZE(non_var_arg_types), ARGUMENTS_SIZE(var_arg_types),        \
      PACK_VARARG_ARGUMENTS(non_var_arg_types),                                \
      PACK_VARARG_ARGUMENTS(var_arg_types), UNPACK_VARARG_RESULTS(ret_types),  \
      UNPACK_RESULTS(ret_types))

#define EMITC_FIXED_TYPEDEF_IMPL(arg_types, ret_types, input_parameters, \
                                 output_parameters)                      \
  typedef iree_status_t (*call_0##arg_types##_##ret_types##_t)(          \
      iree_vm_stack_t * IREE_RESTRICT stack, void* IREE_RESTRICT module, \
      void* IREE_RESTRICT module_state input_parameters output_parameters);

// TODO(simon-camp): We should check the args and rets pointers for NULL, but
// need to special case type 'v'
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
    if (!iree_status_is_ok(status)) {                                      \
      return status;                                                       \
    }                                                                      \
                                                                           \
    unpack_results;                                                        \
                                                                           \
    return status;                                                         \
  }

#define EMITC_VA_LIST_NAME varargs
#define EMITC_CALL_ARG_PTR_NAME ptr
#define EMITC_VLA_IMPORT_IMPL(non_var_arg_types, var_arg_types, ret_types,    \
                              non_var_arg_size, var_arg_size, pack_args,      \
                              pack_var_args, define_results, unpack_results)  \
  static iree_status_t                                                        \
      call_0##non_var_arg_types##C##var_arg_types##D_##ret_types##_import(    \
          iree_vm_stack_t* IREE_RESTRICT stack,                               \
          const iree_vm_function_t* IREE_RESTRICT import, int32_t span_count, \
          ...) {                                                              \
    iree_host_size_t total_size =                                             \
        non_var_arg_size + sizeof(int32_t) + span_count * var_arg_size;       \
                                                                              \
    IREE_VM_ABI_TYPE_NAME(ret_types) results;                                 \
    iree_vm_abi_##ret_types##_reset(&results);                                \
                                                                              \
    iree_vm_function_call_t call;                                             \
    call.function = *import;                                                  \
    call.arguments.data_length = total_size;                                  \
    call.arguments.data = (uint8_t*)iree_alloca(call.arguments.data_length);  \
    call.results = iree_make_byte_span(&results, sizeof(results));            \
                                                                              \
    memset(call.arguments.data, 0, call.arguments.data_length);               \
                                                                              \
    uint8_t* EMITC_CALL_ARG_PTR_NAME = call.arguments.data;                   \
    va_list EMITC_VA_LIST_NAME;                                               \
    va_start(EMITC_VA_LIST_NAME, span_count);                                 \
                                                                              \
    pack_args;                                                                \
    memcpy(EMITC_CALL_ARG_PTR_NAME, &span_count, sizeof(int32_t));            \
    EMITC_CALL_ARG_PTR_NAME += sizeof(int32_t);                               \
    for (int32_t i = 0; i < span_count; i++) {                                \
      pack_var_args                                                           \
    }                                                                         \
    define_results;                                                           \
                                                                              \
    va_end(EMITC_VA_LIST_NAME);                                               \
                                                                              \
    iree_vm_execution_result_t result;                                        \
    memset(&result, 0, sizeof(result));                                       \
                                                                              \
    iree_status_t status =                                                    \
        import->module->begin_call(import->module, stack, &call, &result);    \
                                                                              \
    if (!iree_status_is_ok(status)) {                                         \
      return status;                                                          \
    }                                                                         \
                                                                              \
    unpack_results;                                                           \
                                                                              \
    return status;                                                            \
  }

#define ARGUMENTS_SIZE(types) (0 FOR_EACH(ARGUMENT_SIZE, TUPLE_UNPACK(types)))
#define ARGUMENT_SIZE(idx, arg) +ARGUMENT_SIZE_##arg
#define ARGUMENT_SIZE_f sizeof(float)
#define ARGUMENT_SIZE_i sizeof(int32_t)
#define ARGUMENT_SIZE_r sizeof(iree_vm_ref_t)
#define ARGUMENT_SIZE_v 0

#define INPUT_ARGUMENTS(types) FOR_EACH(INPUT_ARGUMENT, TUPLE_UNPACK(types))
#define INPUT_ARGUMENT(idx, arg) INPUT_ARGUMENT_##arg(idx)
#define INPUT_ARGUMENT_f(idx) , args->f##idx
#define INPUT_ARGUMENT_i(idx) , args->i##idx
#define INPUT_ARGUMENT_r(idx) , &args->r##idx
#define INPUT_ARGUMENT_v(idx)

#define OUTPUT_ARGUMENTS(types) FOR_EACH(OUTPUT_ARGUMENT, TUPLE_UNPACK(types))
#define OUTPUT_ARGUMENT(idx, arg) OUTPUT_ARGUMENT_##arg(idx)
#define OUTPUT_ARGUMENT_f(idx) , &rets->f##idx
#define OUTPUT_ARGUMENT_i(idx) , &rets->i##idx
#define OUTPUT_ARGUMENT_r(idx) , &rets->r##idx
#define OUTPUT_ARGUMENT_v(idx)

#define INPUT_PARAMETERS(types) FOR_EACH(INPUT_PARAMETER, TUPLE_UNPACK(types))
#define INPUT_PARAMETER(idx, arg) INPUT_PARAMETER_##arg(idx)
#define INPUT_PARAMETER_f(idx) , float arg##idx
#define INPUT_PARAMETER_i(idx) , int32_t arg##idx
#define INPUT_PARAMETER_r(idx) , iree_vm_ref_t* arg##idx
#define INPUT_PARAMETER_v(idx)

#define OUTPUT_PARAMETERS(types) FOR_EACH(OUTPUT_PARAMETER, TUPLE_UNPACK(types))
#define OUTPUT_PARAMETER(idx, arg) OUTPUT_PARAMETER_##arg(idx)
#define OUTPUT_PARAMETER_f(idx) , float* ret##idx
#define OUTPUT_PARAMETER_i(idx) , int32_t* ret##idx
#define OUTPUT_PARAMETER_r(idx) , iree_vm_ref_t* ret##idx
#define OUTPUT_PARAMETER_v(idx)

#define PACK_ARGUMENTS(types) FOR_EACH(PACK_ARGUMENT, TUPLE_UNPACK(types))
#define PACK_ARGUMENT(idx, arg) PACK_ARGUMENT_##arg(idx)
#define PACK_ARGUMENT_f(idx) arguments.f##idx = arg##idx;
#define PACK_ARGUMENT_i(idx) arguments.i##idx = arg##idx;
#define PACK_ARGUMENT_r(idx) iree_vm_ref_assign(arg##idx, &arguments.r##idx);
#define PACK_ARGUMENT_v(idx)

#define UNPACK_RESULTS(types) FOR_EACH(UNPACK_RESULT, TUPLE_UNPACK(types))
#define UNPACK_RESULT(idx, arg) UNPACK_RESULT_##arg(idx)
#define UNPACK_RESULT_f(idx) *ret##idx = results.f##idx;
#define UNPACK_RESULT_i(idx) *ret##idx = results.i##idx;
#define UNPACK_RESULT_r(idx) iree_vm_ref_move(&results.r##idx, ret##idx);
#define UNPACK_RESULT_v(idx)

#define PACK_VARARG_ARGUMENTS(types) FOR_EACH(PACK_VARARG, TUPLE_UNPACK(types))
#define PACK_VARARG(idx, arg) PACK_VARARG_##arg(idx)
#define PACK_VARARG_i(idx)                                        \
  PACK_VARARG_i_IMPL(EMITC_CALL_ARG_PTR_NAME, EMITC_VA_LIST_NAME, \
                     CONCAT(_temp, __COUNTER__))
#define PACK_VARARG_i_IMPL(dest, varargs, temp) \
  int32_t temp = va_arg(varargs, int32_t);      \
  memcpy(dest, &temp, sizeof(int32_t));         \
  dest += sizeof(int32_t);
#define PACK_VARARG_r(idx)                                        \
  PACK_VARARG_r_IMPL(EMITC_CALL_ARG_PTR_NAME, EMITC_VA_LIST_NAME, \
                     CONCAT(_temp, __COUNTER__))
#define PACK_VARARG_r_IMPL(dest, varargs, temp)          \
  iree_vm_ref_t* temp = va_arg(varargs, iree_vm_ref_t*); \
  iree_vm_ref_assign(temp, (iree_vm_ref_t*)(dest));      \
  dest += sizeof(iree_vm_ref_t);
#define PACK_VARARG_v(idx)

#define UNPACK_VARARG_RESULTS(types) \
  FOR_EACH(UNPACK_VARARG, TUPLE_UNPACK(types))
#define UNPACK_VARARG(idx, arg) UNPACK_VARARG_##arg(idx)
#define UNPACK_VARARG_i(idx) \
  int32_t* ret##idx = va_arg(EMITC_VA_LIST_NAME, int32_t*);
#define UNPACK_VARARG_r(idx) \
  iree_vm_ref_t* ret##idx = va_arg(EMITC_VA_LIST_NAME, iree_vm_ref_t*);
#define UNPACK_VARARG_v(idx)

EMITC_DEFINE_SHIMS((i), (i))
EMITC_DEFINE_SHIMS((i, i), (i))
EMITC_DEFINE_SHIMS((i, r, i, i), (v))
EMITC_DEFINE_SHIMS((r), (i))
EMITC_DEFINE_SHIMS((r), (i, i))
EMITC_DEFINE_SHIMS((r), (i, i, i))
EMITC_DEFINE_SHIMS((r), (i, i, i, i))
EMITC_DEFINE_SHIMS((r), (r))
EMITC_DEFINE_SHIMS((r), (v))
EMITC_DEFINE_SHIMS((r, i), (i))
// TODO(dajuro): Enable as soon as #7845 has landed
// EMITC_DEFINE_SHIMS((r, i), (f))
EMITC_DEFINE_SHIMS((r, i), (r))
EMITC_DEFINE_SHIMS((r, i), (v))
EMITC_DEFINE_SHIMS((r, i, i), (i))
EMITC_DEFINE_SHIMS((r, i, i), (r))
EMITC_DEFINE_SHIMS((r, i, i), (v))
EMITC_DEFINE_SHIMS((r, i, f), (v))
EMITC_DEFINE_SHIMS((r, i, i, i), (r))
EMITC_DEFINE_SHIMS((r, i, i, i), (v))
EMITC_DEFINE_SHIMS((r, i, i, r, i, i), (r))
EMITC_DEFINE_SHIMS((r, i, i, i, r, i, i), (r))
EMITC_DEFINE_SHIMS((r, i, r, i, i), (v))
EMITC_DEFINE_SHIMS((r, r), (i))
EMITC_DEFINE_SHIMS((r, r), (r))
EMITC_DEFINE_SHIMS((r, r), (v))
EMITC_DEFINE_SHIMS((r, r), (i, i))
EMITC_DEFINE_SHIMS((r, r, r), (i, i))
EMITC_DEFINE_SHIMS((r, r, i, i, i, i), (v))
EMITC_DEFINE_SHIMS((r, r, i, r, i), (v))
EMITC_DEFINE_SHIMS((r, r, i, r, i, i), (v))
EMITC_DEFINE_SHIMS((r, r, r, i, i, i), (v))
EMITC_DEFINE_SHIMS((v), (i))
EMITC_DEFINE_SHIMS((v), (r))
EMITC_DEFINE_SHIMS((v), (v))

EMITC_VLA_IMPORT((r), (i), (i))
EMITC_VLA_IMPORT((r), (r), (v))
EMITC_VLA_IMPORT((r, i), (i), (r))
EMITC_VLA_IMPORT((r, i, i), (i), (r))
EMITC_VLA_IMPORT((r, i), (i, i), (r))
EMITC_VLA_IMPORT((r, i), (r), (r))
EMITC_VLA_IMPORT((r, r, r), (r), (r))
EMITC_VLA_IMPORT((r, r), (i, r, i, i), (r))
EMITC_VLA_IMPORT((r, r, i), (i), (v))
EMITC_VLA_IMPORT((r, r, i, i), (i), (v))
EMITC_VLA_IMPORT((r, r, i), (i, r, i, i), (v))
EMITC_VLA_IMPORT((r, r, i, r), (i), (v))

#endif  // IREE_VM_SHIMS_EMITC_H_
