//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_TYPE_IDENTITY_H
#define _LIBCUDACXX___TYPE_TRAITS_TYPE_IDENTITY_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
struct __type_identity { typedef _Tp type; };

template <class _Tp>
using __type_identity_t _LIBCUDACXX_NODEBUG_TYPE = typename __type_identity<_Tp>::type;

#if _LIBCUDACXX_STD_VER > 11
template<class _Tp> struct type_identity { typedef _Tp type; };
template<class _Tp> using type_identity_t = typename type_identity<_Tp>::type;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_TYPE_IDENTITY_H
