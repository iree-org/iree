//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_SCOPED_ENUM_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_SCOPED_ENUM_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_enum.h"
#include "../__type_traits/underlying_type.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 20
template <class _Tp, bool = is_enum_v<_Tp> >
struct __is_scoped_enum_helper : false_type {};

template <class _Tp>
struct __is_scoped_enum_helper<_Tp, true>
    : public bool_constant<!is_convertible_v<_Tp, underlying_type_t<_Tp> > > {};

template <class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS is_scoped_enum
    : public __is_scoped_enum_helper<_Tp> {};

template <class _Tp>
inline constexpr bool is_scoped_enum_v = is_scoped_enum<_Tp>::value;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_SCOPED_ENUM_H
