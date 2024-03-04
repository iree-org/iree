//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H
#define _LIBCUDACXX___TYPE_TRAITS_IS_REFERENCE_WRAPPER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/remove_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp> class _LIBCUDACXX_TEMPLATE_VIS reference_wrapper;

template <class _Tp> struct __is_reference_wrapper_impl : public false_type {};
template <class _Tp> struct __is_reference_wrapper_impl<reference_wrapper<_Tp> > : public true_type {};
template <class _Tp> struct __is_reference_wrapper
    : public __is_reference_wrapper_impl<__remove_cv_t<_Tp> > {};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_ENABLE_IF_H
