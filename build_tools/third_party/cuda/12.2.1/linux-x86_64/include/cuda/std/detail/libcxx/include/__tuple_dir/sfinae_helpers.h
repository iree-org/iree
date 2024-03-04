//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_SFINAE_HELPERS_H
#define _LIBCUDACXX___TUPLE_SFINAE_HELPERS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__fwd/tuple.h"
#include "../__tuple_dir/make_tuple_types.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_like.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__tuple_dir/tuple_types.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_assignable.h"
#include "../__type_traits/is_constructible.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/remove_cvref.h"
#include "../__type_traits/remove_reference.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef _LIBCUDACXX_CXX03_LANG

template <bool ..._Preds>
struct __all_dummy;

template <bool ..._Pred>
using __all = _IsSame<__all_dummy<_Pred...>, __all_dummy<((void)_Pred, true)...>>;

struct __tuple_sfinae_base {
  template <class, class>
  struct __test_size : false_type {};

  template <class ..._Tp, class ..._Up>
  struct __test_size<__tuple_types<_Tp...>, __tuple_types<_Up...>>
    : _BoolConstant<sizeof...(_Tp) == sizeof...(_Up)> {};

  template <template <class, class...> class,
            class _Tp, class _Up, bool = __test_size<_Tp, _Up>::value>
  struct __test : false_type {};

  template <template <class, class...> class _Trait,
            class ..._LArgs, class ..._RArgs>
  struct __test<_Trait, __tuple_types<_LArgs...>, __tuple_types<_RArgs...>, true>
      : __all<_Trait<_LArgs, _RArgs>::value...> {};

  template <class _FromArgs, class _ToArgs>
  using __constructible = __test<is_constructible, _ToArgs, _FromArgs>;
  template <class _FromArgs, class _ToArgs>
  using __convertible = __test<is_convertible, _FromArgs, _ToArgs>;
  template <class _FromArgs, class _ToArgs>
  using __assignable = __test<is_assignable, _ToArgs, _FromArgs>;
};

// __tuple_convertible

template <class _Tp, class _Up, bool = __tuple_like<__libcpp_remove_reference_t<_Tp>>::value,
                                bool = __tuple_like<_Up>::value>
struct __tuple_convertible
    : public false_type {};

template <class _Tp, class _Up>
struct __tuple_convertible<_Tp, _Up, true, true>
    : public __tuple_sfinae_base::__convertible<
      typename __make_tuple_types<_Tp>::type
    , typename __make_tuple_types<_Up>::type
    >
{};

// __tuple_constructible

template <class _Tp, class _Up, bool = __tuple_like<__libcpp_remove_reference_t<_Tp>>::value,
                                bool = __tuple_like<_Up>::value>
struct __tuple_constructible
    : public false_type {};

template <class _Tp, class _Up>
struct __tuple_constructible<_Tp, _Up, true, true>
    : public __tuple_sfinae_base::__constructible<
      typename __make_tuple_types<_Tp>::type
    , typename __make_tuple_types<_Up>::type
    >
{};

// __tuple_assignable

template <class _Tp, class _Up, bool = __tuple_like<__libcpp_remove_reference_t<_Tp>>::value,
                                bool = __tuple_like<_Up>::value>
struct __tuple_assignable
    : public false_type {};

template <class _Tp, class _Up>
struct __tuple_assignable<_Tp, _Up, true, true>
    : public __tuple_sfinae_base::__assignable<
      typename __make_tuple_types<_Tp>::type
    , typename __make_tuple_types<_Up&>::type
    >
{};


template <size_t _Ip, class ..._Tp>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, tuple<_Tp...> >
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename tuple_element<_Ip, __tuple_types<_Tp...> >::type type;
};

template <bool _IsTuple, class _SizeTrait, size_t _Expected>
struct __tuple_like_with_size_imp : false_type {};

template <class _SizeTrait, size_t _Expected>
struct __tuple_like_with_size_imp<true, _SizeTrait, _Expected>
    : integral_constant<bool, _SizeTrait::value == _Expected> {};

template <class _Tuple, size_t _ExpectedSize,
          class _RawTuple = __remove_cvref_t<_Tuple>>
using __tuple_like_with_size _LIBCUDACXX_NODEBUG_TYPE = __tuple_like_with_size_imp<
                                   __tuple_like<_RawTuple>::value,
                                   tuple_size<_RawTuple>, _ExpectedSize
                              >;

struct _LIBCUDACXX_TYPE_VIS __check_tuple_constructor_fail {
    template <int&...>
    using __enable_explicit_default = false_type;
    template <int&...>
    using __enable_implicit_default = false_type;
    template <class ...>
    using __enable_explicit = false_type;
    template <class ...>
    using __enable_implicit = false_type;
    template <class ...>
    using __enable_assign = false_type;
};
#endif // !defined(_LIBCUDACXX_CXX03_LANG)
#if _LIBCUDACXX_STD_VER > 14

template <bool _CanCopy, bool _CanMove>
struct __sfinae_ctor_base {};
template <>
struct __sfinae_ctor_base<false, false> {
  __sfinae_ctor_base() = default;
  __sfinae_ctor_base(__sfinae_ctor_base const&) = delete;
  __sfinae_ctor_base(__sfinae_ctor_base &&) = delete;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base&&) = default;
};
template <>
struct __sfinae_ctor_base<true, false> {
  __sfinae_ctor_base() = default;
  __sfinae_ctor_base(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base(__sfinae_ctor_base &&) = delete;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base&&) = default;
};
template <>
struct __sfinae_ctor_base<false, true> {
  __sfinae_ctor_base() = default;
  __sfinae_ctor_base(__sfinae_ctor_base const&) = delete;
  __sfinae_ctor_base(__sfinae_ctor_base &&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base const&) = default;
  __sfinae_ctor_base& operator=(__sfinae_ctor_base&&) = default;
};

template <bool _CanCopy, bool _CanMove>
struct __sfinae_assign_base {};
template <>
struct __sfinae_assign_base<false, false> {
  __sfinae_assign_base() = default;
  __sfinae_assign_base(__sfinae_assign_base const&) = default;
  __sfinae_assign_base(__sfinae_assign_base &&) = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base const&) = delete;
  __sfinae_assign_base& operator=(__sfinae_assign_base&&) = delete;
};
template <>
struct __sfinae_assign_base<true, false> {
  __sfinae_assign_base() = default;
  __sfinae_assign_base(__sfinae_assign_base const&) = default;
  __sfinae_assign_base(__sfinae_assign_base &&) = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base const&) = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base&&) = delete;
};
template <>
struct __sfinae_assign_base<false, true> {
  __sfinae_assign_base() = default;
  __sfinae_assign_base(__sfinae_assign_base const&) = default;
  __sfinae_assign_base(__sfinae_assign_base &&) = default;
  __sfinae_assign_base& operator=(__sfinae_assign_base const&) = delete;
  __sfinae_assign_base& operator=(__sfinae_assign_base&&) = default;
};
#endif // _LIBCUDACXX_STD_VER > 14

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TUPLE_SFINAE_HELPERS_H
