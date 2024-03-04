//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_same.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_IS_REFERENCEABLE) && !defined(_LIBCUDACXX_USE_IS_REFERENCEABLE_FALLBACK)

template <class _Tp>
struct __libcpp_is_referenceable
  : public integral_constant<bool, _LIBCUDACXX_IS_REFERENCEABLE(_Tp)>
  {};

#else
struct __libcpp_is_referenceable_impl {
  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static _Tp& __test(int);
  template <class _Tp>
  _LIBCUDACXX_INLINE_VISIBILITY static false_type __test(...);
};

template <class _Tp>
struct __libcpp_is_referenceable
    : integral_constant<bool, _IsNotSame<decltype(__libcpp_is_referenceable_impl::__test<_Tp>(0)), false_type>::value> {
};
#endif // defined(_LIBCUDACXX_IS_REFERENCEABLE) && !defined(_LIBCUDACXX_USE_IS_REFERENCEABLE_FALLBACK)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCEABLE_H
