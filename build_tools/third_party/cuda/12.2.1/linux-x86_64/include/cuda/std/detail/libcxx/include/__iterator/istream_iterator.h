// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/iterator.h"
#include "../__memory/addressof.h"
#include "../cstddef"
#include "../iosfwd"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char,
          class _Traits = char_traits<_CharT>, class _Distance = ptrdiff_t>
class _LIBCUDACXX_TEMPLATE_VIS istream_iterator
#if _LIBCUDACXX_STD_VER <= 14 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<input_iterator_tag, _Tp, _Distance, const _Tp*, const _Tp&>
#endif
{
_LIBCUDACXX_SUPPRESS_DEPRECATED_POP
public:
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef basic_istream<_CharT,_Traits> istream_type;
private:
    istream_type* __in_stream_;
    _Tp __value_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR istream_iterator() : __in_stream_(0), __value_() {}
    _LIBCUDACXX_INLINE_VISIBILITY istream_iterator(istream_type& __s) : __in_stream_(_CUDA_VSTD::addressof(__s))
        {
            if (!(*__in_stream_ >> __value_))
                __in_stream_ = 0;
        }

    _LIBCUDACXX_INLINE_VISIBILITY const _Tp& operator*() const {return __value_;}
    _LIBCUDACXX_INLINE_VISIBILITY const _Tp* operator->() const {return _CUDA_VSTD::addressof((operator*()));}
    _LIBCUDACXX_INLINE_VISIBILITY istream_iterator& operator++()
        {
            if (!(*__in_stream_ >> __value_))
                __in_stream_ = 0;
            return *this;
        }
    _LIBCUDACXX_INLINE_VISIBILITY istream_iterator  operator++(int)
        {istream_iterator __t(*this); ++(*this); return __t;}

    template <class _Up, class _CharU, class _TraitsU, class _DistanceU>
    friend _LIBCUDACXX_INLINE_VISIBILITY
    bool
    operator==(const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __x,
               const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __y);

    template <class _Up, class _CharU, class _TraitsU, class _DistanceU>
    friend _LIBCUDACXX_INLINE_VISIBILITY
    bool
    operator==(const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __x,
               const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __y);
};

template <class _Tp, class _CharT, class _Traits, class _Distance>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool
operator==(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
           const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
    return __x.__in_stream_ == __y.__in_stream_;
}

template <class _Tp, class _CharT, class _Traits, class _Distance>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool
operator!=(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
           const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
    return !(__x == __y);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ISTREAM_ITERATOR_H
