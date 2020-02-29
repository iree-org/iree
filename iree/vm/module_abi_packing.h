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

#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/api_util.h"
#include "iree/base/status.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/ref_cc.h"
#include "iree/vm/stack.h"
#include "iree/vm/types.h"

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

template <typename T>
struct ParamUnpack;

struct ParamUnpackState {
  iree_vm_stack_frame_t* frame;
  int i32_ordinal = 0;
  int ref_ordinal = 0;
  int varargs_ordinal = 0;
  Status status;

  template <typename... Ts>
  static StatusOr<std::tuple<typename ParamUnpack<
      typename std::remove_reference<Ts>::type>::storage_type...>>
  LoadSequence(iree_vm_stack_frame_t* frame) {
    auto params = std::make_tuple(
        typename ParamUnpack<
            typename impl::remove_cvref<Ts>::type>::storage_type()...);

    ParamUnpackState param_state{frame};
    ApplyLoad<Ts...>(&param_state, params,
                     std::make_index_sequence<sizeof...(Ts)>());
    RETURN_IF_ERROR(param_state.status);
    return std::move(params);
  }

  template <typename... Ts, typename T, size_t... I>
  static void ApplyLoad(ParamUnpackState* param_state, T&& params,
                        std::index_sequence<I...>) {
    impl::order_sequence{
        (ParamUnpack<typename std::tuple_element<I, std::tuple<Ts...>>::type>::
             Load(param_state, std::get<I>(params)),
         0)...};
  }
};

template <typename T>
struct ParamUnpack {
  using storage_type = T;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    out_param = static_cast<T>(
        param_state->frame->registers.i32[param_state->i32_ordinal++]);
  }
};

template <>
struct ParamUnpack<opaque_ref> {
  using storage_type = opaque_ref;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    auto* reg = &param_state->frame->registers.ref[param_state->ref_ordinal++];
    if (iree_vm_ref_is_null(reg)) {
      param_state->status = InvalidArgumentErrorBuilder(IREE_LOC)
                            << "argument " << (param_state->varargs_ordinal - 1)
                            << " (" << typeid(storage_type).name() << ")"
                            << " must not be a null";
    } else {
      iree_vm_ref_move(reg, &out_param);
    }
  }
};

template <>
struct ParamUnpack<absl::optional<opaque_ref>> {
  using storage_type = absl::optional<opaque_ref>;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    auto* reg = &param_state->frame->registers.ref[param_state->ref_ordinal++];
    if (!iree_vm_ref_is_null(reg)) {
      out_param = {opaque_ref()};
      iree_vm_ref_move(reg, &out_param.value());
    }
  }
};

template <typename T>
struct ParamUnpack<ref<T>> {
  using storage_type = ref<T>;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    auto& ref_storage =
        param_state->frame->registers.ref[param_state->ref_ordinal++];
    if (ref_storage.type == ref_type_descriptor<T>::get()->type) {
      // Move semantics.
      out_param = ref<T>{reinterpret_cast<T*>(ref_storage.ptr)};
      std::memset(&ref_storage, 0, sizeof(ref_storage));
    } else if (ref_storage.type != IREE_VM_REF_TYPE_NULL) {
      param_state->status =
          InvalidArgumentErrorBuilder(IREE_LOC)
          << "Parameter " << (param_state->varargs_ordinal - 1)
          << " contains a reference to the wrong type; have "
          << iree_vm_ref_type_name(ref_storage.type).data << " but expected "
          << ref_type_descriptor<T>::get()->type_name.data << " ("
          << typeid(storage_type).name() << ")";
    } else {
      param_state->status =
          InvalidArgumentErrorBuilder(IREE_LOC)
          << "Parameter " << (param_state->varargs_ordinal - 1) << "("
          << typeid(storage_type).name() << ") must not be null";
    }
  }
};

template <typename T>
struct ParamUnpack<absl::optional<ref<T>>> {
  using storage_type = absl::optional<ref<T>>;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    auto& ref_storage =
        param_state->frame->registers.ref[param_state->ref_ordinal++];
    if (ref_storage.type == ref_type_descriptor<T>::get()->type) {
      // Move semantics.
      out_param = ref<T>{reinterpret_cast<T*>(ref_storage.ptr)};
      std::memset(&ref_storage, 0, sizeof(ref_storage));
    } else if (ref_storage.type != IREE_VM_REF_TYPE_NULL) {
      param_state->status =
          InvalidArgumentErrorBuilder(IREE_LOC)
          << "Parameter " << (param_state->varargs_ordinal - 1)
          << " contains a reference to the wrong type; have "
          << iree_vm_ref_type_name(ref_storage.type).data << " but expected "
          << ref_type_descriptor<T>::get()->type_name.data << " ("
          << typeid(storage_type).name() << ")";
    } else {
      // NOTE: null is allowed here!
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
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    for (int i = 0; i < S; ++i) {
      ParamUnpack::Load(param_state, out_param[i]);
    }
  }
};

template <typename... Ts>
struct ParamUnpack<std::tuple<Ts...>> {
  using storage_type = std::tuple<typename impl::remove_cvref<Ts>::type...>;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    ++param_state->varargs_ordinal;
    UnpackTuple(param_state, out_param,
                std::make_index_sequence<sizeof...(Ts)>());
  }
  template <size_t... I>
  static void UnpackTuple(ParamUnpackState* param_state, storage_type& params,
                          std::index_sequence<I...>) {
    impl::order_sequence{
        (ParamUnpack<typename std::tuple_element<I, std::tuple<Ts...>>::type>::
             Load(param_state, std::get<I>(params)),
         0)...};
  }
};

template <typename U>
struct ParamUnpack<absl::Span<U>> {
  using element_type = typename impl::remove_cvref<U>::type;
  using storage_type = std::vector<element_type>;
  static void Load(ParamUnpackState* param_state, storage_type& out_param) {
    const uint8_t count = param_state->frame->return_registers
                              ->registers[param_state->varargs_ordinal++];
    int32_t original_varargs_ordinal = param_state->varargs_ordinal;
    out_param.resize(count);
    for (int i = 0; i < count; ++i) {
      ParamUnpack<element_type>::Load(param_state, out_param[i]);
    }
    param_state->varargs_ordinal = original_varargs_ordinal;
  }
};

//===----------------------------------------------------------------------===//
// Result packing
//===----------------------------------------------------------------------===//

struct ResultPackState {
  iree_vm_stack_frame_t* frame;
  int i32_ordinal = 0;
  int ref_ordinal = 0;
  Status status;
};

template <typename T>
struct ResultRegister {
  constexpr static auto value = std::make_tuple<uint8_t>(0);
};

template <typename T>
struct ResultRegister<ref<T>> {
  constexpr static auto value = std::make_tuple<uint8_t>(
      IREE_REF_REGISTER_TYPE_BIT | IREE_REF_REGISTER_MOVE_BIT);
};

template <typename... Ts>
struct ResultRegister<std::tuple<Ts...>> {
  constexpr static auto value = std::tuple_cat(ResultRegister<Ts>::value...);
};

template <typename T>
struct ResultPack {
  static void Store(ResultPackState* result_state, T value) {
    result_state->frame->registers.i32[result_state->i32_ordinal++] =
        static_cast<int32_t>(value);
  }
};

template <typename T>
struct ResultPack<ref<T>> {
  static void Store(ResultPackState* result_state, ref<T> value) {
    if (!value) {
      result_state->status = InvalidArgumentErrorBuilder(IREE_LOC)
                             << "Result (" << typeid(ref<T>).name()
                             << ") must not be null";
      return;
    }
    auto* reg_ptr =
        &result_state->frame->registers.ref[result_state->ref_ordinal++];
    std::memset(reg_ptr, 0, sizeof(*reg_ptr));
    iree_vm_ref_wrap_assign(value.release(), value.type(), reg_ptr);
  }
};

template <typename T>
struct ResultPack<absl::optional<ref<T>>> {
  static void Store(ResultPackState* result_state,
                    absl::optional<ref<T>> value) {
    auto* reg_ptr =
        &result_state->frame->registers.ref[result_state->ref_ordinal++];
    std::memset(reg_ptr, 0, sizeof(*reg_ptr));
    if (value.has_value()) {
      iree_vm_ref_wrap_assign(value.release(), value.type(), reg_ptr);
    }
  }
};

template <typename... Ts>
struct ResultPack<std::tuple<Ts...>> {
  static void Store(ResultPackState* result_state, std::tuple<Ts...> results) {
    PackTuple(result_state, results, std::make_index_sequence<sizeof...(Ts)>());
  }
  template <typename... T, size_t... I>
  static inline void PackTuple(ResultPackState* result_state,
                               std::tuple<T...>& value,
                               std::index_sequence<I...>) {
    impl::order_sequence{
        (ResultPack<typename std::tuple_element<I, std::tuple<T...>>::type>::
             Store(result_state, std::move(std::get<I>(value))),
         0)...};
  }
};

//===----------------------------------------------------------------------===//
// Function wrapping
//===----------------------------------------------------------------------===//

template <typename Owner, typename Results, typename... Params>
struct DispatchFunctor {
  using FnPtr = StatusOr<Results> (Owner::*)(Params...);

  template <typename T, uint8_t... I>
  static constexpr auto ConstTupleOr(std::integer_sequence<uint8_t, I...>) {
    return std::make_tuple(
        (static_cast<uint8_t>(std::get<I>(ResultRegister<T>::value) | I))...);
  }

  template <typename T, size_t... I>
  static constexpr auto TupleToArray(const T& t, std::index_sequence<I...>) {
    return std::array<uint8_t, std::tuple_size<T>::value>{std::get<I>(t)...};
  }

  static Status Call(void (Owner::*ptr)(), Owner* self, iree_vm_stack_t* stack,
                     iree_vm_stack_frame_t* frame,
                     iree_vm_execution_result_t* out_result) {
    ASSIGN_OR_RETURN(auto params,
                     ParamUnpackState::LoadSequence<Params...>(frame));

    frame->return_registers = nullptr;
    frame->registers.ref_register_count = 0;

    auto results_or =
        ApplyFn(reinterpret_cast<FnPtr>(ptr), self, std::move(params),
                std::make_index_sequence<sizeof...(Params)>());
    if (!results_or.ok()) {
      return std::move(results_or).status();
    }

    static const int kLeafCount = impl::LeafCount<Results>::value;
    static const auto kResultList = TupleToArray(
        std::tuple_cat(
            std::make_tuple<uint8_t>(static_cast<uint8_t>(kLeafCount)),
            ConstTupleOr<Results>(
                std::make_integer_sequence<uint8_t, kLeafCount>())),
        std::make_index_sequence<1 + kLeafCount>());
    frame->return_registers =
        reinterpret_cast<const iree_vm_register_list_t*>(kResultList.data());

    ResultPackState result_state{frame};
    auto results = std::move(results_or).ValueOrDie();
    ResultPack<Results>::Store(&result_state, std::move(results));
    return result_state.status;
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
                     iree_vm_stack_frame_t* frame,
                     iree_vm_execution_result_t* out_result) {
    ASSIGN_OR_RETURN(auto params,
                     ParamUnpackState::LoadSequence<Params...>(frame));

    frame->return_registers = nullptr;
    frame->registers.ref_register_count = 0;

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
                       iree_vm_stack_t* stack, iree_vm_stack_frame_t* frame,
                       iree_vm_execution_result_t* out_result);
};

template <typename Owner, typename Result, typename... Params>
constexpr NativeFunction<Owner> MakeNativeFunction(
    const char* name, StatusOr<Result> (Owner::*fn)(Params...)) {
  return {name, (void (Owner::*)())fn,
          &packing::DispatchFunctor<Owner, Result, Params...>::Call};
}

template <typename Owner, typename... Params>
constexpr NativeFunction<Owner> MakeNativeFunction(
    const char* name, Status (Owner::*fn)(Params...)) {
  return {name, (void (Owner::*)())fn,
          &packing::DispatchFunctorVoid<Owner, Params...>::Call};
}

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_MODULE_ABI_PACKING_H_
