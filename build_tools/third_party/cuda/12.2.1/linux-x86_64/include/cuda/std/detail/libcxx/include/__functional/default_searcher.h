// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H
#define _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

// #include "../__algorithm/search.h"
#include "../__functional/identity.h"
#include "../__functional/operations.h"
#include "../__iterator/iterator_traits.h"
#include "../__utility/pair.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _BinaryPredicate, class _ForwardIterator1, class _ForwardIterator2>
_LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
pair<_ForwardIterator1, _ForwardIterator1>
__search(_ForwardIterator1 __first1, _ForwardIterator1 __last1,
         _ForwardIterator2 __first2, _ForwardIterator2 __last2, _BinaryPredicate __pred,
         forward_iterator_tag, forward_iterator_tag)
{
    if (__first2 == __last2)
        return _CUDA_VSTD::make_pair(__first1, __first1);  // Everything matches an empty sequence
    while (true)
    {
        // Find first element in sequence 1 that matchs *__first2, with a mininum of loop checks
        while (true)
        {
            if (__first1 == __last1)  // return __last1 if no element matches *__first2
                return _CUDA_VSTD::make_pair(__last1, __last1);
            if (__pred(*__first1, *__first2))
                break;
            ++__first1;
        }
        // *__first1 matches *__first2, now match elements after here
        _ForwardIterator1 __m1 = __first1;
        _ForwardIterator2 __m2 = __first2;
        while (true)
        {
            if (++__m2 == __last2)  // If pattern exhausted, __first1 is the answer (works for 1 element pattern)
                return _CUDA_VSTD::make_pair(__first1, __m1);
            if (++__m1 == __last1)  // Otherwise if source exhaused, pattern not found
                return _CUDA_VSTD::make_pair(__last1, __last1);
            if (!__pred(*__m1, *__m2))  // if there is a mismatch, restart with a new __first1
            {
                ++__first1;
                break;
            }  // else there is a match, check next elements
        }
    }
}

template <class _BinaryPredicate, class _RandomAccessIterator1, class _RandomAccessIterator2>
_LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_AFTER_CXX11
pair<_RandomAccessIterator1, _RandomAccessIterator1>
__search(_RandomAccessIterator1 __first1, _RandomAccessIterator1 __last1,
         _RandomAccessIterator2 __first2, _RandomAccessIterator2 __last2, _BinaryPredicate __pred,
           random_access_iterator_tag, random_access_iterator_tag)
{
    typedef typename iterator_traits<_RandomAccessIterator1>::difference_type _Diff1;
    typedef typename iterator_traits<_RandomAccessIterator2>::difference_type _Diff2;
    // Take advantage of knowing source and pattern lengths.  Stop short when source is smaller than pattern
    const _Diff2 __len2 = __last2 - __first2;
    if (__len2 == 0)
        return _CUDA_VSTD::make_pair(__first1, __first1);
    const _Diff1 __len1 = __last1 - __first1;
    if (__len1 < __len2)
        return _CUDA_VSTD::make_pair(__last1, __last1);
    const _RandomAccessIterator1 __s = __last1 - (__len2 - 1);  // Start of pattern match can't go beyond here

    while (true)
    {
        while (true)
        {
            if (__first1 == __s)
                return _CUDA_VSTD::make_pair(__last1, __last1);
            if (__pred(*__first1, *__first2))
                break;
            ++__first1;
        }

        _RandomAccessIterator1 __m1 = __first1;
        _RandomAccessIterator2 __m2 = __first2;
         while (true)
         {
             if (++__m2 == __last2)
                 return _CUDA_VSTD::make_pair(__first1, __first1 + __len2);
             ++__m1;          // no need to check range on __m1 because __s guarantees we have enough source
             if (!__pred(*__m1, *__m2))
             {
                 ++__first1;
                 break;
             }
         }
    }
}

#ifndef __cuda_std__

#if _LIBCUDACXX_STD_VER > 14

// default searcher
template<class _ForwardIterator, class _BinaryPredicate = equal_to<>>
class _LIBCUDACXX_TEMPLATE_VIS default_searcher {
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    default_searcher(_ForwardIterator __f, _ForwardIterator __l,
                       _BinaryPredicate __p = _BinaryPredicate())
        : __first_(__f), __last_(__l), __pred_(__p) {}

    template <typename _ForwardIterator2>
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17
    pair<_ForwardIterator2, _ForwardIterator2>
    operator () (_ForwardIterator2 __f, _ForwardIterator2 __l) const
    {
        return _CUDA_VSTD::__search(__f, __l, __first_, __last_, __pred_,
            typename _CUDA_VSTD::iterator_traits<_ForwardIterator>::iterator_category(),
            typename _CUDA_VSTD::iterator_traits<_ForwardIterator2>::iterator_category());
    }

private:
    _ForwardIterator __first_;
    _ForwardIterator __last_;
    _BinaryPredicate __pred_;
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(default_searcher);

#endif // _LIBCUDACXX_STD_VER > 14
#endif // __cuda_std__

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_DEFAULT_SEARCHER_H
