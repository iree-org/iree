//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_EMPTY_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_EMPTY_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_EMPTY) && !defined(_LIBCUDACXX_USE_IS_EMPTY_FALLBACK)

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_empty
    : public integral_constant<bool, _LIBCUDACXX_IS_EMPTY(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_empty_v = _LIBCUDACXX_IS_EMPTY(_Tp);
#endif

#else

template <class _Tp>
struct __is_empty1
    : public _Tp
{
    double __lx;
};

struct __is_empty2
{
    double __lx;
};

template <class _Tp, bool = is_class<_Tp>::value>
struct __libcpp_empty : public integral_constant<bool, sizeof(__is_empty1<_Tp>) == sizeof(__is_empty2)> {};

template <class _Tp> struct __libcpp_empty<_Tp, false> : public false_type {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_empty : public __libcpp_empty<_Tp> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_empty_v = is_empty<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_EMPTY) && !defined(_LIBCUDACXX_USE_IS_EMPTY_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_EMPTY_H
