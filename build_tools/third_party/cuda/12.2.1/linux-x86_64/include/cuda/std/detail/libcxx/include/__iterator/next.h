// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_NEXT_H
#define _LIBCUDACXX___ITERATOR_NEXT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__iterator/advance.h"
#include "../__iterator/iterator_traits.h"
#include "../__type_traits/enable_if.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
__enable_if_t
<
    __is_cpp17_input_iterator<_InputIter>::value,
    _InputIter
>
next(_InputIter __x,
     typename iterator_traits<_InputIter>::difference_type __n = 1)
{
    _LIBCUDACXX_ASSERT(__n >= 0 || __is_cpp17_bidirectional_iterator<_InputIter>::value,
                       "Attempt to next(it, -n) on a non-bidi iterator");

    _CUDA_VSTD::advance(__x, __n);
    return __x;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_NEXT_H
