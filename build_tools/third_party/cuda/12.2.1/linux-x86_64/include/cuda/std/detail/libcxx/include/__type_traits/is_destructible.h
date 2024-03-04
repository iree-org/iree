//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_DESTRUCTIBLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_DESTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_function.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_all_extents.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_DESTRUCTIBLE) && !defined(_LIBCUDACXX_USE_IS_DESTRUCTIBLE_FALLBACK)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_destructible
   : public integral_constant<bool, _LIBCUDACXX_IS_DESTRUCTIBLE(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_destructible_v = _LIBCUDACXX_IS_DESTRUCTIBLE(_Tp);
#endif

#else // __has_builtin(__is_destructible)

//  if it's a reference, return true
//  if it's a function, return false
//  if it's   void,     return false
//  if it's an array of unknown bound, return false
//  Otherwise, return "declval<_Up&>().~_Up()" is well-formed
//    where _Up is remove_all_extents<_Tp>::type

template <class>
struct __is_destructible_apply { typedef int type; };

template <typename _Tp>
struct __is_destructor_wellformed {
    template <typename _Tp1>
    _LIBCUDACXX_INLINE_VISIBILITY static true_type  __test (
        typename __is_destructible_apply<decltype(_CUDA_VSTD::declval<_Tp1&>().~_Tp1())>::type
    );

    template <typename _Tp1>
    _LIBCUDACXX_INLINE_VISIBILITY static false_type __test (...);

    static const bool value = decltype(__test<_Tp>(12))::value;
};

template <class _Tp, bool>
struct __destructible_imp;

template <class _Tp>
struct __destructible_imp<_Tp, false>
   : public integral_constant<bool,
        __is_destructor_wellformed<__remove_all_extents_t<_Tp> >::value> {};

template <class _Tp>
struct __destructible_imp<_Tp, true>
    : public true_type {};

template <class _Tp, bool>
struct __destructible_false;

template <class _Tp>
struct __destructible_false<_Tp, false> : public __destructible_imp<_Tp, is_reference<_Tp>::value> {};

template <class _Tp>
struct __destructible_false<_Tp, true> : public false_type {};

template <class _Tp>
struct is_destructible : public __destructible_false<_Tp, is_function<_Tp>::value> {};

template <class _Tp>
struct is_destructible<_Tp[]> : public false_type {};

template <>
struct is_destructible<void> : public false_type {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_destructible_v = is_destructible<_Tp>::value;
#endif

#endif // __has_builtin(__is_destructible)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_DESTRUCTIBLE_H
