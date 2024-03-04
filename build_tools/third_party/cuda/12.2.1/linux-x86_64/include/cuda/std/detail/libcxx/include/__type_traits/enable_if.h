//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
#define _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <bool, class _Tp = void> struct _LIBCUDACXX_TEMPLATE_VIS enable_if {};
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS enable_if<true, _Tp> {typedef _Tp type;};

template <bool _Bp, class _Tp = void> using __enable_if_t _LIBCUDACXX_NODEBUG_TYPE = typename enable_if<_Bp, _Tp>::type;

#if _LIBCUDACXX_STD_VER > 11
template <bool _Bp, class _Tp = void> using enable_if_t = typename enable_if<_Bp, _Tp>::type;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
