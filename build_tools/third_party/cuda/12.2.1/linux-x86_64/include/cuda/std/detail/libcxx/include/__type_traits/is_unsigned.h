//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_arithmetic.h"
#include "../__type_traits/is_integral.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Before AppleClang 14, __is_unsigned returned true for enums with signed underlying type.
#if defined(_LIBCUDACXX_IS_UNSIGNED) && !defined(_LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK) && !(defined(_LIBCUDACXX_APPLE_CLANG_VER) && _LIBCUDACXX_APPLE_CLANG_VER < 1400)

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_unsigned
    : public integral_constant<bool, _LIBCUDACXX_IS_UNSIGNED(_Tp)>
    {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_unsigned_v = _LIBCUDACXX_IS_UNSIGNED(_Tp);
#endif

#else

template <class _Tp, bool = is_integral<_Tp>::value>
struct __libcpp_is_unsigned_impl : public _BoolConstant<(_Tp(0) < _Tp(-1))> {};

template <class _Tp>
struct __libcpp_is_unsigned_impl<_Tp, false> : public false_type {};  // floating point

template <class _Tp, bool = is_arithmetic<_Tp>::value>
struct __libcpp_is_unsigned : public __libcpp_is_unsigned_impl<_Tp> {};

template <class _Tp> struct __libcpp_is_unsigned<_Tp, false> : public false_type {};

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS is_unsigned : public __libcpp_is_unsigned<_Tp> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_unsigned_v = is_unsigned<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_UNSIGNED) && !defined(_LIBCUDACXX_USE_IS_UNSIGNED_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_UNSIGNED_H
