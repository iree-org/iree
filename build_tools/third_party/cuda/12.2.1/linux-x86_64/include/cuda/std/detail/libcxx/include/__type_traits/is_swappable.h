//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SWAPPABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SWAPPABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/add_lvalue_reference.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_referenceable.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/nat.h"
#include "../__utility/declval.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp> struct __is_swappable;
template <class _Tp> struct __is_nothrow_swappable;

#ifndef _LIBCUDACXX_CXX03_LANG
template <class _Tp>
using __swap_result_t = __enable_if_t<is_move_constructible<_Tp>::value && is_move_assignable<_Tp>::value>;
#else
template <class>
using __swap_result_t = void;
#endif

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __swap_result_t<_Tp>
swap(_Tp& __x, _Tp& __y) _NOEXCEPT_(is_nothrow_move_constructible<_Tp>::value &&
                                    is_nothrow_move_assignable<_Tp>::value);

template<class _Tp, size_t _Np>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<__is_swappable<_Tp>::value>
swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) _NOEXCEPT_(__is_nothrow_swappable<_Tp>::value);

namespace __detail
{
// ALL generic swap overloads MUST already have a declaration available at this point.

template <class _Tp, class _Up = _Tp,
          bool _NotVoid = !is_void<_Tp>::value && !is_void<_Up>::value>
struct __swappable_with
{
    template <class _LHS, class _RHS>
    _LIBCUDACXX_INLINE_VISIBILITY static decltype(swap(_CUDA_VSTD::declval<_LHS>(), _CUDA_VSTD::declval<_RHS>()))
    __test_swap(int);
    template <class, class>
    _LIBCUDACXX_INLINE_VISIBILITY static __nat __test_swap(long);

    // Extra parens are needed for the C++03 definition of decltype.
    typedef decltype((__test_swap<_Tp, _Up>(0))) __swap1;
    typedef decltype((__test_swap<_Up, _Tp>(0))) __swap2;

    static const bool value = _IsNotSame<__swap1, __nat>::value
                           && _IsNotSame<__swap2, __nat>::value;
};

template <class _Tp, class _Up>
struct __swappable_with<_Tp, _Up,  false> : false_type {};

template <class _Tp, class _Up = _Tp, bool _Swappable = __swappable_with<_Tp, _Up>::value>
struct __nothrow_swappable_with {
  static const bool value =
#ifndef _LIBCUDACXX_HAS_NO_NOEXCEPT
      noexcept(swap(_CUDA_VSTD::declval<_Tp>(), _CUDA_VSTD::declval<_Up>()))
  &&  noexcept(swap(_CUDA_VSTD::declval<_Up>(), _CUDA_VSTD::declval<_Tp>()));
#else
      false;
#endif
};

template <class _Tp, class _Up>
struct __nothrow_swappable_with<_Tp, _Up, false> : false_type {};

} // namespace __detail

template <class _Tp>
struct __is_swappable
    : public integral_constant<bool, __detail::__swappable_with<_Tp&>::value>
{
};

template <class _Tp>
struct __is_nothrow_swappable
    : public integral_constant<bool, __detail::__nothrow_swappable_with<_Tp&>::value>
{
};

#if _LIBCUDACXX_STD_VER > 11

template <class _Tp, class _Up>
struct _LIBCUDACXX_TEMPLATE_VIS is_swappable_with
    : public integral_constant<bool, __detail::__swappable_with<_Tp, _Up>::value>
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_swappable
    : public __conditional_t<
        __libcpp_is_referenceable<_Tp>::value,
        is_swappable_with<
            __add_lvalue_reference_t<_Tp>,
            __add_lvalue_reference_t<_Tp> >,
        false_type
    >
{
};

template <class _Tp, class _Up>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_swappable_with
    : public integral_constant<bool, __detail::__nothrow_swappable_with<_Tp, _Up>::value>
{
};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_nothrow_swappable
    : public __conditional_t<
        __libcpp_is_referenceable<_Tp>::value,
        is_nothrow_swappable_with<
            __add_lvalue_reference_t<_Tp>,
            __add_lvalue_reference_t<_Tp> >,
        false_type
    >
{
};

template <class _Tp, class _Up>
_LIBCUDACXX_INLINE_VAR constexpr bool is_swappable_with_v = is_swappable_with<_Tp, _Up>::value;

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_swappable_v = is_swappable<_Tp>::value;

template <class _Tp, class _Up>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_swappable_with_v = is_nothrow_swappable_with<_Tp, _Up>::value;

template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_nothrow_swappable_v = is_nothrow_swappable<_Tp>::value;

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SWAPPABLE_H
