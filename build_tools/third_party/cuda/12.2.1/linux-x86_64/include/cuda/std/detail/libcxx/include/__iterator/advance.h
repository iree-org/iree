// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ADVANCE_H
#define _LIBCUDACXX___ITERATOR_ADVANCE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__assert"
#include "../__iterator/iterator_traits.h"
#include "../cstdlib"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
void __advance(_InputIter& __i,
             typename iterator_traits<_InputIter>::difference_type __n, input_iterator_tag)
{
    for (; __n > 0; --__n)
        ++__i;
}

template <class _BiDirIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
void __advance(_BiDirIter& __i,
             typename iterator_traits<_BiDirIter>::difference_type __n, bidirectional_iterator_tag)
{
    if (__n >= 0)
        for (; __n > 0; --__n)
            ++__i;
    else
        for (; __n < 0; ++__n)
            --__i;
}

template <class _RandIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
void __advance(_RandIter& __i,
             typename iterator_traits<_RandIter>::difference_type __n, random_access_iterator_tag)
{
   __i += __n;
}

template <class _InputIter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
void advance(_InputIter& __i,
             typename iterator_traits<_InputIter>::difference_type __n)
{
    _LIBCUDACXX_ASSERT(__n >= 0 || __is_cpp17_bidirectional_iterator<_InputIter>::value,
                       "Attempt to advance(it, -n) on a non-bidi iterator");
    __advance(__i, __n, typename iterator_traits<_InputIter>::iterator_category());
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ADVANCE_H
