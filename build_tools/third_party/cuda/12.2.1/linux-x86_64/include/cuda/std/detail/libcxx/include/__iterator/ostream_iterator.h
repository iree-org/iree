// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/iterator.h"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../cstddef"
#include "../iosfwd"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char, class _Traits = char_traits<_CharT> >
class _LIBCUDACXX_TEMPLATE_VIS ostream_iterator
#if _LIBCUDACXX_STD_VER <= 14 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif
{
_LIBCUDACXX_SUPPRESS_DEPRECATED_POP
public:
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef basic_ostream<_CharT,_Traits> ostream_type;
private:
    ostream_type* __out_stream_;
    const char_type* __delim_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator(ostream_type& __s) _NOEXCEPT
        : __out_stream_(_CUDA_VSTD::addressof(__s)), __delim_(0) {}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator(ostream_type& __s, const _CharT* __delimiter) _NOEXCEPT
        : __out_stream_(_CUDA_VSTD::addressof(__s)), __delim_(__delimiter) {}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator=(const _Tp& __value_)
        {
            *__out_stream_ << __value_;
            if (__delim_)
                *__out_stream_ << __delim_;
            return *this;
        }

    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator*()     {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator++()    {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY ostream_iterator& operator++(int) {return *this;}
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_OSTREAM_ITERATOR_H
