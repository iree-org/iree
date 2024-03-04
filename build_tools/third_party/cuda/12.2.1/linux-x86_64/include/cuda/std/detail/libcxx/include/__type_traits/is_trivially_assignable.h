//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_ASSIGNABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_ASSIGNABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_scalar.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_TRIVIALLY_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_ASSIGNABLE_FALLBACK)

template <class _Tp, class _Arg>
struct is_trivially_assignable
    : integral_constant<bool, _LIBCUDACXX_IS_TRIVIALLY_ASSIGNABLE(_Tp, _Arg)>
{ };

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VAR constexpr bool is_trivially_assignable_v = _LIBCUDACXX_IS_TRIVIALLY_ASSIGNABLE(_Tp, _Arg);
#endif

#else

template <class _Tp, class _Arg>
struct is_trivially_assignable
    : public false_type {};

template <class _Tp>
struct is_trivially_assignable<_Tp&, _Tp>
    : integral_constant<bool, is_scalar<_Tp>::value> {};

template <class _Tp>
struct is_trivially_assignable<_Tp&, _Tp&>
    : integral_constant<bool, is_scalar<_Tp>::value> {};

template <class _Tp>
struct is_trivially_assignable<_Tp&, const _Tp&>
    : integral_constant<bool, is_scalar<_Tp>::value> {};

template <class _Tp>
struct is_trivially_assignable<_Tp&, _Tp&&>
    : integral_constant<bool, is_scalar<_Tp>::value> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp, class _Arg>
_LIBCUDACXX_INLINE_VAR constexpr bool is_trivially_assignable_v
    = is_trivially_assignable<_Tp, _Arg>::value;
#endif

#endif // defined(_LIBCUDACXX_IS_TRIVIALLY_ASSIGNABLE) && !defined(_LIBCUDACXX_USE_IS_TRIVIALLY_ASSIGNABLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_TRIVIALLY_ASSIGNABLE_H
