//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H
#define _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool>
struct _IfImpl;

template <>
struct _IfImpl<true> {
  template <class _IfRes, class _ElseRes>
  using _Select _LIBCUDACXX_NODEBUG_TYPE = _IfRes;
};

template <>
struct _IfImpl<false> {
  template <class _IfRes, class _ElseRes>
  using _Select _LIBCUDACXX_NODEBUG_TYPE = _ElseRes;
};

template <bool _Cond, class _IfRes, class _ElseRes>
using _If _LIBCUDACXX_NODEBUG_TYPE = typename _IfImpl<_Cond>::template _Select<_IfRes, _ElseRes>;

template <bool _Bp, class _If, class _Then>
    struct _LIBCUDACXX_TEMPLATE_VIS conditional {typedef _If type;};
template <class _If, class _Then>
    struct _LIBCUDACXX_TEMPLATE_VIS conditional<false, _If, _Then> {typedef _Then type;};

#if _LIBCUDACXX_STD_VER > 11
template <bool _Bp, class _IfRes, class _ElseRes>
using conditional_t = typename conditional<_Bp, _IfRes, _ElseRes>::type;
#endif

// Helper so we can use "conditional_t" in all language versions.
template <bool _Bp, class _If, class _Then> using __conditional_t = typename conditional<_Bp, _If, _Then>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_CONDITIONAL_H
