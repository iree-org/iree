//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SAME_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SAME_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_SAME) && !defined(_LIBCUDACXX_USE_IS_SAME_FALLBACK)

template <class _Tp, class _Up>
struct _LIBCUDACXX_TEMPLATE_VIS is_same : _BoolConstant<_LIBCUDACXX_IS_SAME(_Tp, _Up)> { };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Up>
_LIBCUDACXX_INLINE_VAR constexpr bool is_same_v = _LIBCUDACXX_IS_SAME(_Tp, _Up);
#endif

// _IsSame<T,U> has the same effect as is_same<T,U> but instantiates fewer types:
// is_same<A,B> and is_same<C,D> are guaranteed to be different types, but
// _IsSame<A,B> and _IsSame<C,D> are the same type (namely, false_type).
// Neither GCC nor Clang can mangle the __is_same builtin, so _IsSame
// mustn't be directly used anywhere that contributes to name-mangling
// (such as in a dependent return type).

template <class _Tp, class _Up>
using _IsSame = _BoolConstant<_LIBCUDACXX_IS_SAME(_Tp, _Up)>;

template <class _Tp, class _Up>
using _IsNotSame = _BoolConstant<!_LIBCUDACXX_IS_SAME(_Tp, _Up)>;

#else

template <class _Tp, class _Up> struct _LIBCUDACXX_TEMPLATE_VIS is_same           : public false_type {};
template <class _Tp>            struct _LIBCUDACXX_TEMPLATE_VIS is_same<_Tp, _Tp> : public true_type {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Up>
_LIBCUDACXX_INLINE_VAR constexpr bool is_same_v = is_same<_Tp, _Up>::value;
#endif

// _IsSame<T,U> has the same effect as is_same<T,U> but instantiates fewer types:
// is_same<A,B> and is_same<C,D> are guaranteed to be different types, but
// _IsSame<A,B> and _IsSame<C,D> are the same type (namely, false_type).
// Neither GCC nor Clang can mangle the __is_same builtin, so _IsSame
// mustn't be directly used anywhere that contributes to name-mangling
// (such as in a dependent return type).

template <class _Tp, class _Up>
using _IsSame = _BoolConstant<is_same<_Tp, _Up>::value>;

template <class _Tp, class _Up>
using _IsNotSame = _BoolConstant<!is_same<_Tp, _Up>::value>;

#endif // defined(_LIBCUDACXX_IS_SAME) && !defined(_LIBCUDACXX_USE_IS_SAME_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SAME_H
