//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_FINAL)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS
__libcpp_is_final : public integral_constant<bool, _LIBCUDACXX_IS_FINAL(_Tp)> {};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS
is_final : public integral_constant<bool, _LIBCUDACXX_IS_FINAL(_Tp)> {};
#endif

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_final_v = _LIBCUDACXX_IS_FINAL(_Tp);
#endif

#else

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS
__libcpp_is_final : public false_type {};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS
is_final :  public false_type {};
#endif

#if _LIBCUDACXX_STD_VER > 11 && !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool is_final_v = false;
#endif

#endif // defined(_LIBCUDACXX_IS_FINAL)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_FINAL_H
