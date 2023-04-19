// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_VM_MODULE_ABI_PACKING_H_
#define IREE_VM_MODULE_ABI_PACKING_H_

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/span.h"
#include "iree/vm/module.h"
#include "iree/vm/ref.h"
#include "iree/vm/stack.h"

// std::string_view is available starting in C++17.
// Prior to that only IREE's C iree_string_view_t is available.
#if defined(__has_include)
#if __has_include(<string_view>) && __cplusplus >= 201703L
#define IREE_HAVE_STD_STRING_VIEW 1
#include <string_view>
#endif  // __has_include(<string_view>)
#endif  // __has_include

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

}  // namespace impl

template <typename T>
using enable_if_primitive =
    typename std::enable_if<std::is_arithmetic<T>::value ||
                            std::is_enum<T>::value>::type;
template <typename T>
using enable_if_not_primitive = typename std::enable_if<!(
    std::is_arithmetic<T>::value || std::is_enum<T>::value)>::type;

//===----------------------------------------------------------------------===//
// Compile-time string literals
//===----------------------------------------------------------------------===//

// Compile-time constant string.
// This allows us to concat string literals and produce a single flattened
// char[] containing the results. Includes a \0 so the character storage is
// length N + 1 and can be accessed as a c_str.
//
// Use the `literal` helper function to define a const string literal without
// needing the size.
//
// Example:
//  // produces: const_string<2>("ab")
//  constexpr const auto str = literal("a") + literal("b");
template <size_t N>
class const_string {
 public:
  constexpr const_string(const char (&data)[N + 1])
      : const_string(data, std::make_index_sequence<N>()) {}
  template <size_t N1, typename std::enable_if<(N1 <= N), bool>::type = true>
  constexpr const_string(const const_string<N1>& lhs,
                         const const_string<N - N1>& rhs)
      : const_string{lhs, rhs, std::make_index_sequence<N1>{},
                     std::make_index_sequence<N - N1>{}} {}

  constexpr std::size_t size() const { return N; }
  constexpr const char* data() const { return data_; }
  constexpr const char* c_str() const { return data_; }
  constexpr operator const char*() const { return data_; }
  constexpr char operator[](size_t i) const { return data_[i]; }

 private:
  template <size_t... PACK>
  constexpr const_string(const char (&data)[N + 1],
                         std::index_sequence<PACK...>)
      : data_{data[PACK]..., '\0'} {}
  template <size_t N1, size_t... PACK1, size_t... PACK2>
  constexpr const_string(const const_string<N1>& lhs,
                         const const_string<N - N1>& rhs,
                         std::index_sequence<PACK1...>,
                         std::index_sequence<PACK2...>)
      : data_{lhs[PACK1]..., rhs[PACK2]..., '\0'} {}

  const char data_[N + 1];
};

template <size_t N1, size_t N2>
constexpr auto operator+(const const_string<N1>& lhs,
                         const const_string<N2>& rhs) {
  return const_string<N1 + N2>(lhs, rhs);
}

// Defines a compile-time constant string literal.
template <size_t N_PLUS_1>
constexpr auto literal(const char (&data)[N_PLUS_1]) {
  return const_string<N_PLUS_1 - 1>(data);
}

constexpr auto concat_impl() { return literal(""); }
template <typename T>
constexpr auto concat_impl(const T& lhs) {
  return lhs;
}
template <typename T, typename... Ts>
constexpr auto concat_impl(const T& lhs, const Ts&... s) {
  return lhs + concat_impl(s...);
}

// Concatenates one or more const_string values into a new const_string.
//
// Example:
//  constexpr const auto abc = concat_literals(literal("a"),
//                                             literal("b"),
//                                             literal("c"));
template <typename... Ts>
constexpr auto concat_literals(const Ts&... s) {
  return concat_impl(s...);
}

template <size_t C, typename T>
struct splat_impl {
  static constexpr auto apply(const T& v) {
    return concat_literals(v, splat_impl<C - 1, T>::apply(v));
  }
};
template <typename T>
struct splat_impl<1, T> {
  static constexpr auto apply(const T& v) { return v; }
};

// Splats a single const_string value C times.
//
// Example:
//  constexpr const auto aaa = splat_literal<3>(literal("a"));
template <size_t C, typename T>
constexpr auto splat_literal(const T& v) {
  return splat_impl<C, T>::apply(v);
}

//===----------------------------------------------------------------------===//
// Calling convention format generation
//===----------------------------------------------------------------------===//
// Prototyped here: https://godbolt.org/z/Tvhh7M

template <typename T>
struct cconv_map;

template <>
struct cconv_map<int32_t> {
  static constexpr const auto conv_chars = literal("i");
};
template <>
struct cconv_map<uint32_t> {
  static constexpr const auto conv_chars = literal("i");
};

template <>
struct cconv_map<int64_t> {
  static constexpr const auto conv_chars = literal("I");
};
template <>
struct cconv_map<uint64_t> {
  static constexpr const auto conv_chars = literal("I");
};

template <>
struct cconv_map<float> {
  static constexpr const auto conv_chars = literal("f");
};
template <>
struct cconv_map<double> {
  static constexpr const auto conv_chars = literal("F");
};

template <>
struct cconv_map<opaque_ref> {
  static constexpr const auto conv_chars = literal("r");
};
template <typename T>
struct cconv_map<ref<T>> {
  static constexpr const auto conv_chars = literal("r");
};
template <>
struct cconv_map<iree_string_view_t> {
  static constexpr const auto conv_chars = literal("r");
};
#if defined(IREE_HAVE_STD_STRING_VIEW)
template <>
struct cconv_map<std::string_view> {
  static constexpr const auto conv_chars = literal("r");
};
#endif  // IREE_HAVE_STD_STRING_VIEW

template <typename U, size_t S>
struct cconv_map<std::array<U, S>> {
  static constexpr const auto conv_chars = splat_literal<S>(
      cconv_map<typename impl::remove_cvref<U>::type>::conv_chars);
};

template <typename... Ts>
struct cconv_map<std::tuple<Ts...>> {
  static constexpr const auto conv_chars = concat_literals(
      cconv_map<typename impl::remove_cvref<Ts>::type>::conv_chars...);
};

template <typename U>
struct cconv_map<iree::span<U>> {
  static constexpr const auto conv_chars = concat_literals(
      literal("C"), cconv_map<typename impl::remove_cvref<U>::type>::conv_chars,
      literal("D"));
};

template <typename Result, size_t ParamsCount, typename... Params>
struct cconv_storage {
  static const iree_string_view_t value() {
    static constexpr const auto value = concat_literals(
        literal("0"),
        concat_literals(
            cconv_map<
                typename impl::remove_cvref<Params>::type>::conv_chars...),
        literal("_"),
        concat_literals(
            cconv_map<typename impl::remove_cvref<Result>::type>::conv_chars));
    static constexpr const auto str =
        iree_string_view_t{value.data(), value.size()};
    return str;
  }
};

template <typename Result>
struct cconv_storage<Result, 0> {
  static const iree_string_view_t value() {
    static constexpr const auto value = concat_literals(
        literal("0v_"),
        concat_literals(
            cconv_map<typename impl::remove_cvref<Result>::type>::conv_chars));
    static constexpr const auto str =
        iree_string_view_t{value.data(), value.size()};
    return str;
  }
};

template <size_t ParamsCount, typename... Params>
struct cconv_storage_void {
  static const iree_string_view_t value() {
    static constexpr const auto value = concat_literals(
        literal("0"),
        concat_literals(
            cconv_map<
                typename impl::remove_cvref<Params>::type>::conv_chars...),
        literal("_v"));
    static constexpr const auto str =
        iree_string_view_t{value.data(), value.size()};
    return str;
  }
};

template <>
struct cconv_storage_void<0> {
  static const iree_string_view_t value() {
    static constexpr const auto value = concat_literals(literal("0v_v"));
    static constexpr const auto str =
        iree_string_view_t{value.data(), value.size()};
    return str;
  }
};

//===----------------------------------------------------------------------===//
// Parameter unpacking
//===----------------------------------------------------------------------===//

// TODO(benvanik): see if we can't use `extern template` to share
// implementations of these and prevent code bloat across many modules.
// We can also try some non-templated base functions (like "UnpackI32") that the
// templated ones simply wrap with type casts.

namespace impl {

using params_ptr_t = uint8_t*;

template <typename T, typename EN = void>
struct ParamUnpack;
template <>
struct ParamUnpack<opaque_ref>;
template <typename T>
struct ParamUnpack<ref<T>>;
template <typename T>
struct ParamUnpack<const ref<T>>;
template <>
struct ParamUnpack<iree_string_view_t>;
#if defined(IREE_HAVE_STD_STRING_VIEW)
template <>
struct ParamUnpack<std::string_view>;
#endif  // IREE_HAVE_STD_STRING_VIEW
template <typename U, size_t S>
struct ParamUnpack<std::array<U, S>>;
template <typename... Ts>
struct ParamUnpack<std::tuple<Ts...>>;
template <typename U>
struct ParamUnpack<iree::span<U>, enable_if_not_primitive<U>>;
template <typename U>
struct ParamUnpack<iree::span<U>, enable_if_primitive<U>>;

struct Unpacker {
  template <typename... Ts>
  static StatusOr<std::tuple<typename impl::ParamUnpack<
      typename std::remove_reference<Ts>::type>::storage_type...>>
  LoadSequence(iree_byte_span_t storage) {
    auto params = std::make_tuple(
        typename impl::ParamUnpack<
            typename impl::remove_cvref<Ts>::type>::storage_type()...);
    Status status;
    params_ptr_t ptr = storage.data;
    ApplyLoad<Ts...>(status, ptr, params,
                     std::make_index_sequence<sizeof...(Ts)>());
    IREE_RETURN_IF_ERROR(std::move(status));
    params_ptr_t limit = storage.data + storage.data_length;
    if (IREE_UNLIKELY(ptr != limit)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "argument buffer unpacking failure; consumed %zu of %zu bytes",
          (reinterpret_cast<intptr_t>(ptr) -
           reinterpret_cast<intptr_t>(storage.data)),
          storage.data_length);
    }
    return std::move(params);
  }

 private:
  template <typename... Ts, typename T, size_t... I>
  static void ApplyLoad(Status& status, params_ptr_t& ptr, T&& params,
                        std::index_sequence<I...>) {
    impl::order_sequence{
        (impl::ParamUnpack<typename impl::remove_cvref<
             typename std::tuple_element<I, std::tuple<Ts...>>::type>::type>::
             Load(status, ptr, std::get<I>(params)),
         0)...};
  }
};

// Common primitive types (`i32`, `i64`, `f32`, enums, etc).
template <typename T>
struct ParamUnpack<T, enable_if_primitive<T>> {
  using storage_type = T;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    out_param = *reinterpret_cast<const T*>(ptr);
    ptr += sizeof(T);
  }
};

// An opaque ref type (`vm.ref<?>`), possibly null.
template <>
struct ParamUnpack<opaque_ref> {
  using storage_type = opaque_ref;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    iree_vm_ref_retain(reinterpret_cast<iree_vm_ref_t*>(ptr), &out_param);
    ptr += sizeof(iree_vm_ref_t);
  }
};

// A `vm.ref<T>` type, possibly null.
// Ownership is transferred to the parameter.
template <typename T>
struct ParamUnpack<ref<T>> {
  using storage_type = ref<T>;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    auto* reg_ptr = reinterpret_cast<iree_vm_ref_t*>(ptr);
    ptr += sizeof(iree_vm_ref_t);
    if (reg_ptr->type == ref_type_descriptor<T>::type()) {
      out_param = vm::retain_ref(reinterpret_cast<T*>(reg_ptr->ptr));
      memset(reg_ptr, 0, sizeof(*reg_ptr));
    } else if (IREE_UNLIKELY(reg_ptr->type != IREE_VM_REF_TYPE_NULL)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter contains a reference to the wrong type; "
          "have %.*s but expected %.*s",
          (int)iree_vm_ref_type_name(reg_ptr->type).size,
          iree_vm_ref_type_name(reg_ptr->type).data,
          (int)iree_vm_ref_type_name(ref_type_descriptor<T>::type()).size,
          iree_vm_ref_type_name(ref_type_descriptor<T>::type()).data);
    } else {
      out_param = {};
    }
  }
};

// TODO(benvanik): merge with above somehow?
template <typename T>
struct ParamUnpack<const ref<T>> {
  using storage_type = ref<T>;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    auto* reg_ptr = reinterpret_cast<iree_vm_ref_t*>(ptr);
    ptr += sizeof(iree_vm_ref_t);
    if (reg_ptr->type == ref_type_descriptor<T>::type()) {
      out_param = vm::retain_ref(reinterpret_cast<T*>(reg_ptr->ptr));
      memset(reg_ptr, 0, sizeof(*reg_ptr));
    } else if (IREE_UNLIKELY(reg_ptr->type != IREE_VM_REF_TYPE_NULL)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter contains a reference to the wrong type; "
          "have %.*s but expected %.*s",
          (int)iree_vm_ref_type_name(reg_ptr->type).size,
          iree_vm_ref_type_name(reg_ptr->type).data,
          (int)iree_vm_ref_type_name(ref_type_descriptor<T>::type()).size,
          iree_vm_ref_type_name(ref_type_descriptor<T>::type()).data);
    } else {
      out_param = {};
    }
  }
};

// An `util.byte_buffer` containing a string.
// The string view is aliased directly into the underlying byte buffer.
template <>
struct ParamUnpack<iree_string_view_t> {
  using storage_type = iree_string_view_t;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    auto* reg_ptr = reinterpret_cast<iree_vm_ref_t*>(ptr);
    ptr += sizeof(iree_vm_ref_t);
    if (reg_ptr->type == ref_type_descriptor<iree_vm_buffer_t>::type()) {
      auto byte_span = reinterpret_cast<iree_vm_buffer_t*>(reg_ptr->ptr)->data;
      out_param = iree_make_string_view(
          reinterpret_cast<const char*>(byte_span.data), byte_span.data_length);
    } else if (IREE_UNLIKELY(reg_ptr->type != IREE_VM_REF_TYPE_NULL)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter contains a reference to the wrong type; "
          "have %.*s but expected %.*s",
          (int)iree_vm_ref_type_name(reg_ptr->type).size,
          iree_vm_ref_type_name(reg_ptr->type).data,
          (int)iree_vm_ref_type_name(
              ref_type_descriptor<iree_vm_buffer_t>::type())
              .size,
          iree_vm_ref_type_name(ref_type_descriptor<iree_vm_buffer_t>::type())
              .data);
    } else {
      // NOTE: empty string is allowed here!
      out_param = iree_string_view_empty();
    }
  }
};
#if defined(IREE_HAVE_STD_STRING_VIEW)
template <>
struct ParamUnpack<std::string_view> {
  using storage_type = std::string_view;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    auto* reg_ptr = reinterpret_cast<iree_vm_ref_t*>(ptr);
    ptr += sizeof(iree_vm_ref_t);
    if (reg_ptr->type == ref_type_descriptor<iree_vm_buffer_t>::type()) {
      auto byte_span = reinterpret_cast<iree_vm_buffer_t*>(reg_ptr->ptr)->data;
      out_param = std::string_view{
          reinterpret_cast<const char*>(byte_span.data), byte_span.data_length};
    } else if (IREE_UNLIKELY(reg_ptr->type != IREE_VM_REF_TYPE_NULL)) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "parameter contains a reference to the wrong type; "
          "have %.*s but expected %.*s",
          (int)iree_vm_ref_type_name(reg_ptr->type).size,
          iree_vm_ref_type_name(reg_ptr->type).data,
          (int)iree_vm_ref_type_name(
              ref_type_descriptor<iree_vm_buffer_t>::type())
              .size,
          iree_vm_ref_type_name(ref_type_descriptor<iree_vm_buffer_t>::type())
              .data);
    } else {
      // NOTE: empty string is allowed here!
      out_param = {};
    }
  }
};
#endif  // IREE_HAVE_STD_STRING_VIEW

// Arrays are C++ ABI only representing a fixed repeated field (`i32, i32`).
template <typename U, size_t S>
struct ParamUnpack<std::array<U, S>> {
  using element_type = typename impl::remove_cvref<U>::type;
  using storage_type = std::array<element_type, S>;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    for (size_t i = 0; i < S; ++i) {
      ParamUnpack::Load(status, ptr, out_param[i]);
    }
  }
};

// Tuples (`tuple<i32, i64>`) expand to just their flattened contents.
template <typename... Ts>
struct ParamUnpack<std::tuple<Ts...>> {
  using storage_type = std::tuple<typename impl::remove_cvref<Ts>::type...>;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    UnpackTuple(status, ptr, out_param,
                std::make_index_sequence<sizeof...(Ts)>());
  }
  template <size_t... I>
  static void UnpackTuple(Status& status, params_ptr_t& ptr,
                          storage_type& params, std::index_sequence<I...>) {
    impl::order_sequence{
        (ParamUnpack<typename std::tuple_element<I, std::tuple<Ts...>>::type>::
             Load(status, ptr, std::get<I>(params)),
         0)...};
  }
};

// Complex variadic span (like `tuple<i32, tuple<ref<...>, i64>>...`).
// We need to allocate storage here so that we can marshal the element type out.
// In the future we could check that all subelements are primitives and alias if
// the host machine endianness is the same.
template <typename U>
struct ParamUnpack<iree::span<U>, enable_if_not_primitive<U>> {
  using element_type = typename impl::remove_cvref<U>::type;
  using storage_type = std::vector<element_type>;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    iree_host_size_t count = *reinterpret_cast<const int32_t*>(ptr);
    ptr += sizeof(int32_t);
    out_param.resize(count);
    for (iree_host_size_t i = 0; i < count; ++i) {
      ParamUnpack<element_type>::Load(status, ptr, out_param[i]);
    }
  }
};

// Simple primitive variadic span (like `i32...`). We can alias directly into
// the argument buffer so long as endianness matches.
template <typename U>
struct ParamUnpack<iree::span<U>, enable_if_primitive<U>> {
  using element_type = U;
  using storage_type = iree::span<const element_type>;
  static void Load(Status& status, params_ptr_t& ptr, storage_type& out_param) {
    iree_host_size_t count = *reinterpret_cast<const int32_t*>(ptr);
    ptr += sizeof(int32_t);
    out_param =
        iree::span<U>(reinterpret_cast<const element_type*>(ptr), count);
    ptr += sizeof(element_type) * count;
  }
};

}  // namespace impl

//===----------------------------------------------------------------------===//
// Result packing
//===----------------------------------------------------------------------===//

namespace impl {

using result_ptr_t = uint8_t*;

template <typename T>
struct ResultPack {
  static void Store(result_ptr_t& ptr, T value) {
    *reinterpret_cast<T*>(ptr) = value;
    ptr += sizeof(T);
  }
};

template <>
struct ResultPack<opaque_ref> {
  static void Store(result_ptr_t& ptr, opaque_ref value) {
    iree_vm_ref_move(value.get(), reinterpret_cast<iree_vm_ref_t*>(ptr));
    ptr += sizeof(iree_vm_ref_t);
  }
};

template <typename T>
struct ResultPack<ref<T>> {
  static void Store(result_ptr_t& ptr, ref<T> value) {
    iree_vm_ref_wrap_assign(value.get(), value.type(),
                            reinterpret_cast<iree_vm_ref_t*>(ptr));
    value.release();
    ptr += sizeof(iree_vm_ref_t);
  }
};

template <typename U, size_t S>
struct ResultPack<std::array<U, S>>;
template <typename... Ts>
struct ResultPack<std::tuple<Ts...>>;

template <typename U, size_t S>
struct ResultPack<std::array<U, S>> {
  static void Store(result_ptr_t& ptr, std::array<U, S> value) {
    for (size_t i = 0; i < S; ++i) {
      ResultPack<U>::Store(ptr, std::move(value[i]));
    }
  }
};

template <typename... Ts>
struct ResultPack<std::tuple<Ts...>> {
  static void Store(result_ptr_t& ptr, std::tuple<Ts...> results) {
    PackTuple(ptr, results, std::make_index_sequence<sizeof...(Ts)>());
  }
  template <typename... T, size_t... I>
  static inline void PackTuple(result_ptr_t& ptr, std::tuple<T...>& value,
                               std::index_sequence<I...>) {
    impl::order_sequence{
        (ResultPack<typename std::tuple_element<I, std::tuple<T...>>::type>::
             Store(ptr, std::move(std::get<I>(value))),
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

  static Status Call(void (Owner::*ptr)(), Owner* self, iree_vm_stack_t* stack,
                     iree_vm_function_call_t call) {
    // Marshal arguments into types/locals we can forward to the function.
    IREE_ASSIGN_OR_RETURN(
        auto params, impl::Unpacker::LoadSequence<Params...>(call.arguments));

    // Call the target function with the params.
    IREE_ASSIGN_OR_RETURN(
        auto results,
        ApplyFn(reinterpret_cast<FnPtr>(ptr), self, std::move(params),
                std::make_index_sequence<sizeof...(Params)>()));

    // Marshal call results back into the ABI results buffer.
    impl::result_ptr_t result_ptr = call.results.data;
    impl::ResultPack<Results>::Store(result_ptr, std::move(results));

    return OkStatus();
  }

  template <typename T, size_t... I>
  static StatusOr<Results> ApplyFn(FnPtr ptr, Owner* self, T&& params,
                                   std::index_sequence<I...>) {
    return (self->*ptr)(std::move(std::get<I>(params))...);
  }
};

// A DispatchFunctor specialization for methods with no return values.
template <typename Owner, typename... Params>
struct DispatchFunctorVoid {
  using FnPtr = Status (Owner::*)(Params...);

  static Status Call(void (Owner::*ptr)(), Owner* self, iree_vm_stack_t* stack,
                     iree_vm_function_call_t call) {
    IREE_ASSIGN_OR_RETURN(
        auto params, impl::Unpacker::LoadSequence<Params...>(call.arguments));
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
  iree_string_view_t name;
  iree_string_view_t cconv;
  void (Owner::*const ptr)();
  Status (*const call)(void (Owner::*ptr)(), Owner* self,
                       iree_vm_stack_t* stack, iree_vm_function_call_t call);
};

template <typename Owner, typename Result, typename... Params>
constexpr NativeFunction<Owner> MakeNativeFunction(
    const char* name, StatusOr<Result> (Owner::*fn)(Params...)) {
  using dispatch_functor_t = packing::DispatchFunctor<Owner, Result, Params...>;
  return {iree_make_cstring_view(name),
          packing::cconv_storage<Result, sizeof...(Params), Params...>::value(),
          (void (Owner::*)())fn, &dispatch_functor_t::Call};
}

template <typename Owner, typename... Params>
constexpr NativeFunction<Owner> MakeNativeFunction(
    const char* name, Status (Owner::*fn)(Params...)) {
  using dispatch_functor_t = packing::DispatchFunctorVoid<Owner, Params...>;
  return {iree_make_cstring_view(name),
          packing::cconv_storage_void<sizeof...(Params), Params...>::value(),
          (void (Owner::*)())fn, &dispatch_functor_t::Call};
}

}  // namespace vm
}  // namespace iree

#endif  // IREE_VM_MODULE_ABI_PACKING_H_
