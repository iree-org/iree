//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H
#define _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/add_lvalue_reference.h"
#include "../__type_traits/add_rvalue_reference.h"
#include "../__type_traits/copy_cv.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _From, class _To>
struct __copy_cvref
{
    using type = __copy_cv_t<_From, _To>;
};

template <class _From, class _To>
struct __copy_cvref<_From&, _To>
{
    using type = __add_lvalue_reference_t<__copy_cv_t<_From, _To> >;
};

template <class _From, class _To>
struct __copy_cvref<_From&&, _To>
{
    using type = __add_rvalue_reference_t<__copy_cv_t<_From, _To> >;
};

template <class _From, class _To>
using __copy_cvref_t = typename __copy_cvref<_From, _To>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H
