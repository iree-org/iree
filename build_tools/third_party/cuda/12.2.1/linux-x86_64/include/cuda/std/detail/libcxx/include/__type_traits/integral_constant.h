//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_INTEGRAL_CONSTANT_H
#define _LIBCUDACXX___TYPE_TRAITS_INTEGRAL_CONSTANT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp, _Tp __v>
struct _LIBCUDACXX_TEMPLATE_VIS integral_constant
{
  static _LIBCUDACXX_CONSTEXPR const _Tp      value = __v;
  typedef _Tp               value_type;
  typedef integral_constant type;
  _LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR operator value_type() const _NOEXCEPT {return value;}
#if _LIBCUDACXX_STD_VER > 11
  _LIBCUDACXX_INLINE_VISIBILITY
  constexpr value_type operator ()() const _NOEXCEPT {return value;}
#endif
};

template <class _Tp, _Tp __v>
_LIBCUDACXX_CONSTEXPR const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true>  true_type;
typedef integral_constant<bool, false> false_type;

template <bool _Val>
using _BoolConstant _LIBCUDACXX_NODEBUG_TYPE = integral_constant<bool, _Val>;

#if _LIBCUDACXX_STD_VER > 11
template <bool __b>
using bool_constant = integral_constant<bool, __b>;
#endif

#if _LIBCUDACXX_STD_VER > 11
#define _LIBCUDACXX_BOOL_CONSTANT(__b) bool_constant<(__b)>
#else
#define _LIBCUDACXX_BOOL_CONSTANT(__b) integral_constant<bool,(__b)>
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_INTEGRAL_CONSTANT_H
