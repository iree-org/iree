//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_RANK_H
#define _LIBCUDACXX___TYPE_TRAITS_RANK_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_ARRAY_RANK) && !defined(_LIBCUDACXX_USE_ARRAY_RANK_FALLBACK) && 0

template <class _Tp>
struct rank : integral_constant<size_t, _LIBCUDACXX_ARRAY_RANK(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr size_t rank_v = _LIBCUDACXX_ARRAY_RANK(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS rank
    : public integral_constant<size_t, 0> {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS rank<_Tp[]>
    : public integral_constant<size_t, rank<_Tp>::value + 1> {};
template <class _Tp, size_t _Np> struct _LIBCUDACXX_TEMPLATE_VIS rank<_Tp[_Np]>
    : public integral_constant<size_t, rank<_Tp>::value + 1> {};

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr size_t rank_v = rank<_Tp>::value;
#endif

#endif // defined(_LIBCUDACXX_ARRAY_RANK) && !defined(_LIBCUDACXX_USE_ARRAY_RANK_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_RANK_H
