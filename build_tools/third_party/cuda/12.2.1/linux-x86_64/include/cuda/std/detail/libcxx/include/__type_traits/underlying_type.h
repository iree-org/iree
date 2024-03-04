//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H
#define _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/is_enum.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_UNDERLYING_TYPE) && !defined(_LIBCUDACXX_USE_UNDERLYING_TYPE_FALLBACK)

template <class _Tp, bool = is_enum<_Tp>::value> struct __underlying_type_impl;

template <class _Tp>
struct __underlying_type_impl<_Tp, false> {};

template <class _Tp>
struct __underlying_type_impl<_Tp, true>
{
    typedef _LIBCUDACXX_UNDERLYING_TYPE(_Tp) type;
};

template <class _Tp>
struct underlying_type : __underlying_type_impl<_Tp, is_enum<_Tp>::value> {};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using underlying_type_t = typename underlying_type<_Tp>::type;
#endif

#else

template <class _Tp, bool _Support = false>
struct underlying_type
{
    static_assert(_Support, "The underyling_type trait requires compiler "
                            "support. Either no such support exists or "
                            "libc++ does not know how to use it.");
};

#endif // defined(_LIBCUDACXX_UNDERLYING_TYPE) && !defined(_LIBCUDACXX_USE_UNDERLYING_TYPE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_UNDERLYING_TYPE_H
