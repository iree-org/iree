// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_VM_MODULE_ABI_PACKING_H_
#define IREE_VM_MODULE_ABI_PACKING_H_

#include <memory>
#include <tuple>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/vm/builtin_types.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/ref_cc.h"
#include "iree/vm/stack.h"

namespace iree {
namespace vm {
namespace packing {

namespace impl {

// Workaround required to ensure proper evaluation order of parameter packs.
// MSVC (and other compilers, like clang-cl in MSVC compat mode) may evaluate
// parameter pack function arguments in any order. This shim allows us to expand
// the parameter pack inside of an initializer list, which unlike function
// arguments must be evaluated by the compiler in the order the elements appear
// in the list.
//
// Example:
//  impl::order_sequence{(ExpandedAction(), 0)...};
//
// More information:
// https://stackoverflow.com/questions/29194858/order-of-function-calls-in-variadic-template-expansion
struct order_sequence {
  template <typename... T>
  order_sequence(T&&...) {}
};

// Coming in C++20, but not widely available yet.
template <class T>
struct remove_cvref {
  typedef std::remove_cv_t<std::remove_reference_t<T>> type;
};

// Counts the total number of leaf elements in a type tree.
// For example:
//   LeafCount<int>::value == 1
//   LeafCount<std::tuple<int, int>> == 2
//   LeafCount<std::tuple<int, std::tuple<int, int>>>::value == 3
template <typename T>
struct LeafCount {
  constexpr static int value = 1;
};
template <typename... Ts>
struct LeafCount<std::tuple<Ts...>> {
  template <int I, typename... Tail>
  struct Adder;
  template <int I, typename T, typename... Tail>
  struct Adder<I, T, Tail...> {
    constexpr static int value =
        LeafCount<T>::value + Adder<sizeof...(Tail), Tail...>::value;
  };
  template <typename T, typename... Tail>
  struct Adder<1, T, Tail...> {
    constexpr static int value = LeafCount<T>::value;
  };
  constexpr static int value = Adder<sizeof...(Ts), Ts...>::value;
};

}  // namespace impl

//===----------------------------------------------------------------------===//
// Parameter unpacking
//===----------------------------------------------------------------------===//

namespace impl {
template <typename T>
struct ParamUnpack;
}  // namespace impl

struct Unpacker {
  // Register storage for the caller frame the registers will be sourced from.
  const iree_vm_registers_t* registers;
  // Argument register list within the caller frame storage.
  const iree_vm_register_list_t* argument_list;
  // Optional variadic argument segment sizes.
  const iree_vm_register_list_t* variadic_segment_size_list;
  // Current flattened argument ordinal mapping into the argument_list list.
  int argument_ordinal = 0;
  // Ordinal of the current variadic segment in the variadic_segment_size_list.
  int segment_ordinal = 0;
  // Current unpack status, set to failure on the first error encountered.
  Status status;

  template <typename... Ts>
  static StatusOr<std::tuple<typename impl::ParamUnpack<
      typename std::remove_reference<Ts>::type>::storage_type...>>
  LoadSequence(const iree_vm_registers_t* registers,
               const iree_vm_register_list_t* argument_list,
               const iree_vm_register_list_t* variadic_segment_size_list) {
    // TODO(#1991): verify argument_list and variadic_segment_size_list are
    // valid to unpack (counts match expectations, etc).
    auto params = std::make_tuple(
        typename impl::ParamUnpack<
            typename impl::remove_cvref<Ts>::type>::storage_type()...);
    Unpacker unpacker(registers, argument_list, variadic_segment_size_list);
    ApplyLoad<Ts...>(&unpacker, params,
                     std::make_index_sequence<sizeof...(Ts)>());
    IREE_RETURN_IF_ERROR(std::move(unpacker.status));
    return std::move(params);
  }

 private:
  Unpacker(const iree_vm_registers_t* registers,
           const iree_vm_register_list_t* argument_list,
           const iree_vm_register_list_t* variadic_segment_size_list)
      : registers(registers),
        argument_list(argument_list),
        variadic_segment_size_list(variadic_segment_size_list) {}

  template <typename... Ts, typename T, size_t... I>
  static void ApplyLoad(Unpacker* unpacker, T&& params,
                        std::index_sequence<I...>) {
    impl::order_sequence{
        (impl::ParamUnpack<typename std::tuple_element<
             I, std::tuple<Ts...>>::type>::Load(unpacker, std::get<I>(params)),
         0)...};
  }
};

namespace impl {

template <typename T>
struct ParamUnpack {
  using storage_type = T;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    out_param = static_cast<T>(
        unpacker->registers->i32[reg & unpacker->registers->i32_mask]);
  }
};

template <>
struct ParamUnpack<int64_t> {
  using storage_type = int64_t;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    out_param = static_cast<int64_t>(
        unpacker->registers->i32[reg & (unpacker->registers->i32_mask & ~1)]);
  }
};

template <>
struct ParamUnpack<uint64_t> {
  using storage_type = uint64_t;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    out_param = static_cast<uint64_t>(
        unpacker->registers->i32[reg & (unpacker->registers->i32_mask & ~1)]);
  }
};

template <>
struct ParamUnpack<opaque_ref> {
  using storage_type = opaque_ref;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    auto* reg_ptr =
        &unpacker->registers->ref[reg & unpacker->registers->ref_mask];
    if (iree_vm_ref_is_null(reg_ptr)) {
      unpacker->status = InvalidArgumentErrorBuilder(IREE_LOC)
                         << "argument " << (unpacker->segment_ordinal - 1)
                         << " (" << typeid(storage_type).name() << ")"
                         << " must not be a null";
    } else {
      iree_vm_ref_retain_or_move(reg & IREE_REF_REGISTER_MOVE_BIT, reg_ptr,
                                 &out_param);
    }
  }
};

template <>
struct ParamUnpack<absl::optional<opaque_ref>> {
  using storage_type = absl::optional<opaque_ref>;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    auto* reg_ptr =
        &unpacker->registers->ref[reg & unpacker->registers->ref_mask];
    if (!iree_vm_ref_is_null(reg_ptr)) {
      out_param = {opaque_ref()};
      iree_vm_ref_retain_or_move(reg & IREE_REF_REGISTER_MOVE_BIT, reg_ptr,
                                 &out_param.value());
    }
  }
};

template <typename T>
struct ParamUnpack<ref<T>> {
  using storage_type = ref<T>;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    auto* reg_ptr =
        &unpacker->registers->ref[reg & unpacker->registers->ref_mask];
    if (reg_ptr->type == ref_type_descriptor<T>::get()->type) {
      if (reg & IREE_REF_REGISTER_MOVE_BIT) {
        out_param = vm::assign_ref(reinterpret_cast<T*>(reg_ptr->ptr));
        memset(reg_ptr, 0, sizeof(*reg_ptr));
      } else {
        out_param = vm::retain_ref(reinterpret_cast<T*>(reg_ptr->ptr));
      }
    } else if (reg_ptr->type != IREE_VM_REF_TYPE_NULL) {
      unpacker->status = InvalidArgumentErrorBuilder(IREE_LOC)
                         << "Parameter " << (unpacker->segment_ordinal - 1)
                         << " contains a reference to the wrong type; have "
                         << iree_vm_ref_type_name(reg_ptr->type).data
                         << " but expected "
                         << ref_type_descriptor<T>::get()->type_name.data
                         << " (" << typeid(storage_type).name() << ")";
    } else {
      unpacker->status = InvalidArgumentErrorBuilder(IREE_LOC)
                         << "Parameter " << (unpacker->segment_ordinal - 1)
                         << " (" << typeid(storage_type).name()
                         << ") must not be null";
    }
  }
};

template <typename T>
struct ParamUnpack<absl::optional<ref<T>>> {
  using storage_type = absl::optional<ref<T>>;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    auto* reg_ptr =
        &unpacker->registers->ref[reg & unpacker->registers->ref_mask];
    if (reg_ptr->type == ref_type_descriptor<T>::get()->type) {
      iree_vm_ref_retain_or_move(reg & IREE_REF_REGISTER_MOVE_BIT, reg_ptr,
                                 &out_param.value());
    } else if (reg_ptr->type != IREE_VM_REF_TYPE_NULL) {
      unpacker->status = InvalidArgumentErrorBuilder(IREE_LOC)
                         << "Parameter " << (unpacker->segment_ordinal - 1)
                         << " contains a reference to the wrong type; have "
                         << iree_vm_ref_type_name(reg_ptr->type).data
                         << " but expected "
                         << ref_type_descriptor<T>::get()->type_name.data
                         << " (" << typeid(storage_type).name() << ")";
    } else {
      // NOTE: null is allowed here!
      out_param = {};
    }
  }
};

template <>
struct ParamUnpack<absl::string_view> {
  using storage_type = absl::string_view;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    ++unpacker->segment_ordinal;
    uint16_t reg =
        unpacker->argument_list->registers[unpacker->argument_ordinal++];
    auto* reg_ptr =
        &unpacker->registers->ref[reg & unpacker->registers->ref_mask];
    if (reg_ptr->type ==
        ref_type_descriptor<iree_vm_ro_byte_buffer_t>::get()->type) {
      auto byte_span =
          reinterpret_cast<iree_vm_ro_byte_buffer_t*>(reg_ptr->ptr)->data;
      out_param = absl::string_view{
          reinterpret_cast<const char*>(byte_span.data), byte_span.data_length};
    } else if (reg_ptr->type != IREE_VM_REF_TYPE_NULL) {
      unpacker->status = InvalidArgumentErrorBuilder(IREE_LOC)
                         << "Parameter " << (unpacker->segment_ordinal - 1)
                         << " contains a reference to the wrong type; have "
                         << iree_vm_ref_type_name(reg_ptr->type).data
                         << " but expected "
                         << ref_type_descriptor<iree_vm_ro_byte_buffer_t>::get()
                                ->type_name.data
                         << " (" << typeid(storage_type).name() << ")";
    } else {
      // NOTE: empty string is allowed here!
      out_param = {};
    }
  }
};

template <typename U, size_t S>
struct ParamUnpack<std::array<U, S>>;
template <typename... Ts>
struct ParamUnpack<std::tuple<Ts...>>;
template <typename U>
struct ParamUnpack<absl::Span<U>>;

template <typename U, size_t S>
struct ParamUnpack<std::array<U, S>> {
  using element_type = typename impl::remove_cvref<U>::type;
  using storage_type = std::array<element_type, S>;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    for (int i = 0; i < S; ++i) {
      ParamUnpack::Load(unpacker, out_param[i]);
    }
  }
};

template <typename... Ts>
struct ParamUnpack<std::tuple<Ts...>> {
  using storage_type = std::tuple<typename impl::remove_cvref<Ts>::type...>;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    UnpackTuple(unpacker, out_param, std::make_index_sequence<sizeof...(Ts)>());
  }
  template <size_t... I>
  static void UnpackTuple(Unpacker* unpacker, storage_type& params,
                          std::index_sequence<I...>) {
    impl::order_sequence{
        (ParamUnpack<typename std::tuple_element<I, std::tuple<Ts...>>::type>::
             Load(unpacker, std::get<I>(params)),
         0)...};
  }
};

template <typename U>
struct ParamUnpack<absl::Span<U>> {
  using element_type = typename impl::remove_cvref<U>::type;
  using storage_type = absl::InlinedVector<element_type, 16>;
  static void Load(Unpacker* unpacker, storage_type& out_param) {
    const uint16_t count = unpacker->variadic_segment_size_list
                               ->registers[unpacker->segment_ordinal++];
    // TODO(benvanik): this may be too many, but it's better than nothing.
    out_param.reserve(count);
    int32_t original_segment_ordinal = unpacker->segment_ordinal;
    while (unpacker->segment_ordinal - original_segment_ordinal < count) {
      out_param.push_back({});
      ParamUnpack<element_type>::Load(unpacker, out_param.back());
    }
    unpacker->segment_ordinal = original_segment_ordinal;
  }
};

}  // namespace impl

//===----------------------------------------------------------------------===//
// Result packing
//===----------------------------------------------------------------------===//

namespace impl {
template <typename T>
struct ResultPack;
}  // namespace impl

struct Packer {
  // Caller frame register storage.
  const iree_vm_registers_t* registers;
  // Registers within the caller frame to store results into.
  const iree_vm_register_list_t* result_list;
  // Current flattened result ordinal mapping into the result_list list.
  int result_ordinal = 0;
  // Result packing status used to return packing failures.
  Status status;

  template <typename Results>
  static Status StoreSequence(const iree_vm_registers_t* registers,
                              const iree_vm_register_list_t* result_list,
                              Results results) {
    static const int kLeafResultCount = impl::LeafCount<Results>::value;
    if (kLeafResultCount > 0 &&
        (!result_list || result_list->size != kLeafResultCount)) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Function returns results but no result registers provided or "
                "result count mismatch";
    }

    Packer state(registers, result_list);
    impl::ResultPack<Results>::Store(&state, std::move(results));
    return std::move(state.status);
  }

 private:
  Packer(const iree_vm_registers_t* registers,
         const iree_vm_register_list_t* result_list)
      : registers(registers), result_list(result_list) {}
};

namespace impl {

template <typename T>
struct ResultPack {
  static void Store(Packer* packer, T value) {
    uint16_t reg = packer->result_list->registers[packer->result_ordinal++];
    packer->registers->i32[reg & packer->registers->i32_mask] =
        static_cast<int32_t>(value);
  }
};

template <>
struct ResultPack<int64_t> {
  static void Store(Packer* packer, int64_t value) {
    uint16_t reg = packer->result_list->registers[packer->result_ordinal++];
    packer->registers->i32[reg & (packer->registers->i32_mask & ~1)] =
        static_cast<int64_t>(value);
  }
};

template <>
struct ResultPack<uint64_t> {
  static void Store(Packer* packer, uint64_t value) {
    uint16_t reg = packer->result_list->registers[packer->result_ordinal++];
    packer->registers->i32[reg & (packer->registers->i32_mask & ~1)] =
        static_cast<uint64_t>(value);
  }
};

template <>
struct ResultPack<opaque_ref> {
  static void Store(Packer* packer, opaque_ref value) {
    if (!value) {
      packer->status = InvalidArgumentErrorBuilder(IREE_LOC)
                       << "Result (" << typeid(opaque_ref).name()
                       << ") must not be null";
      return;
    }
    uint16_t reg = packer->result_list->registers[packer->result_ordinal++];
    auto* reg_ptr = &packer->registers->ref[reg & packer->registers->ref_mask];
    iree_vm_ref_move(value.get(), reg_ptr);
  }
};

template <typename T>
struct ResultPack<ref<T>> {
  static void Store(Packer* packer, ref<T> value) {
    if (!value) {
      packer->status = InvalidArgumentErrorBuilder(IREE_LOC)
                       << "Result (" << typeid(ref<T>).name()
                       << ") must not be null";
      return;
    }
    uint16_t reg = packer->result_list->registers[packer->result_ordinal++];
    auto* reg_ptr = &packer->registers->ref[reg & packer->registers->ref_mask];
    iree_vm_ref_wrap_assign(value.release(), value.type(), reg_ptr);
  }
};

template <typename U, size_t S>
struct ResultPack<std::array<U, S>>;
template <typename... Ts>
struct ResultPack<std::tuple<Ts...>>;

template <typename T>
struct ResultPack<absl::optional<ref<T>>> {
  static void Store(Packer* packer, absl::optional<ref<T>> value) {
    uint16_t reg = packer->result_list->registers[packer->result_ordinal++];
    auto* reg_ptr = &packer->registers->ref[reg & packer->registers->ref_mask];
    if (value.has_value()) {
      iree_vm_ref_wrap_assign(value.release(), value.type(), reg_ptr);
    }
  }
};

template <typename U, size_t S>
struct ResultPack<std::array<U, S>> {
  static void Store(Packer* packer, std::array<U, S> value) {
    for (int i = 0; i < S; ++i) {
      ResultPack<U>::Store(packer, std::move(value[i]));
    }
  }
};

template <typename... Ts>
struct ResultPack<std::tuple<Ts...>> {
  static void Store(Packer* packer, std::tuple<Ts...> results) {
    PackTuple(packer, results, std::make_index_sequence<sizeof...(Ts)>());
  }
  template <typename... T, size_t... I>
  static inline void PackTuple(Packer* packer, std::tuple<T...>& value,
                               std::index_sequence<I...>) {
    impl::order_sequence{
        (ResultPack<typename std::tuple_element<I, std::tuple<T...>>::type>::
             Store(packer, std::move(std::get<I>(value))),
         0)...};
  }
};

}  // namespace impl

//===----------------------------------------------------------------------===//
// Function wrapping
//===----------------------------------------------------------------------===//

template <typename Owner, typename Results, typename... Params>
struct DispatchFunctor {
  using FnPtr = StatusOr<Results> (Owner::*)(Params...);

  template <typename T, size_t... I>
  constexpr static auto TupleToArray(const T& t, std::index_sequence<I...>) {
    return std::array<uint16_t, std::tuple_size<T>::value>{std::get<I>(t)...};
  }

  static Status Call(void (Owner::*ptr)(), Owner* self, iree_vm_stack_t* stack,
                     const iree_vm_function_call_t* call,
                     iree_vm_execution_result_t* out_result) {
    iree_vm_stack_frame_t* caller_frame = iree_vm_stack_current_frame(stack);
    IREE_ASSIGN_OR_RETURN(
        auto params, Unpacker::LoadSequence<Params...>(
                         &caller_frame->registers, call->argument_registers,
                         call->variadic_segment_size_list));

    auto results_or =
        ApplyFn(reinterpret_cast<FnPtr>(ptr), self, std::move(params),
                std::make_index_sequence<sizeof...(Params)>());
    if (!results_or.ok()) {
      return std::move(results_or).status();
    }

    return Packer::StoreSequence<Results>(&caller_frame->registers,
                                          call->result_registers,
                                          std::move(results_or).value());
  }

  template <typename T, size_t... I>
  static StatusOr<Results> ApplyFn(FnPtr ptr, Owner* self, T&& params,
                                   std::index_sequence<I...>) {
    return (self->*ptr)(std::move(std::get<I>(params))...);
  }
};

template <typename Owner, typename... Params>
struct DispatchFunctorVoid {
  using FnPtr = Status (Owner::*)(Params...);

  static Status Call(void (Owner::*ptr)(), Owner* self, iree_vm_stack_t* stack,
                     const iree_vm_function_call_t* call,
                     iree_vm_execution_result_t* out_result) {
    if (call->result_registers && call->result_registers->size > 0) {
      return InvalidArgumentErrorBuilder(IREE_LOC)
             << "Function returns no results but was provided result registers";
    }

    iree_vm_stack_frame_t* caller_frame = iree_vm_stack_current_frame(stack);
    IREE_ASSIGN_OR_RETURN(
        auto params, Unpacker::LoadSequence<Params...>(
                         &caller_frame->registers, call->argument_registers,
                         call->variadic_segment_size_list));

    return ApplyFn(reinterpret_cast<FnPtr>(ptr), self, std::move(params),
                   std::make_index_sequence<sizeof...(Params)>());
  }

  template <typename T, size_t... I>
  static Status ApplyFn(FnPtr ptr, Owner* self, T&& params,
                        std::index_sequence<I...>) {
    return (self->*ptr)(std::move(std::get<I>(params))...);
  }
};

}  // namespace packing

template <typename Owner>
struct NativeFunction {
  const char* name;
  void (Owner::*const ptr)();
  Status (*const call)(void (Owner::*ptr)(), Owner* self,
                       iree_vm_stack_t* stack,
                       const iree_vm_function_call_t* call,
                       iree_vm_execution_result_t* out_result);
};

template <typename Owner, typename Result, typename... Params>
constexpr NativeFunction<Owner> MakeNativeFunction(
    const char* name, StatusOr<Result> (Owner::*fn)(Params...)) {
  using dispatch_functor_t = packing::DispatchFunctor<Owner, Result, Params...>;
  return {name, (void (Owner::*)())fn, &dispatch_functor_t::Call};
}

template <typename Owner, typename... Params>
constexpr NativeFunction<Owner> MakeNativeFunction(
    const char* name, Status (Owner::*fn)(Params...)) {
  using dispatch_functor_t = packing::DispatchFunctorVoid<Owner, Params...>;
  return {name, (void (Owner::*)())fn, &dispatch_functor_t::Call};
}

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_MODULE_ABI_PACKING_H_
