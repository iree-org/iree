// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_OSTREAMBUF_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_OSTREAMBUF_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/iterator.h"
#include "../__iterator/iterator_traits.h"
#include "../cstddef"
#include "../iosfwd"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_SUPPRESS_DEPRECATED_PUSH
template <class _CharT, class _Traits>
class _LIBCUDACXX_TEMPLATE_VIS ostreambuf_iterator
#if _LIBCUDACXX_STD_VER <= 14 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<output_iterator_tag, void, void, void, void>
#endif
{
_LIBCUDACXX_SUPPRESS_DEPRECATED_POP
public:
    typedef _CharT                          char_type;
    typedef _Traits                         traits_type;
    typedef basic_streambuf<_CharT,_Traits> streambuf_type;
    typedef basic_ostream<_CharT,_Traits>   ostream_type;
private:
    streambuf_type* __sbuf_;
public:
    _LIBCUDACXX_INLINE_VISIBILITY ostreambuf_iterator(ostream_type& __s) _NOEXCEPT
        : __sbuf_(__s.rdbuf()) {}
    _LIBCUDACXX_INLINE_VISIBILITY ostreambuf_iterator(streambuf_type* __s) _NOEXCEPT
        : __sbuf_(__s) {}
    _LIBCUDACXX_INLINE_VISIBILITY ostreambuf_iterator& operator=(_CharT __c)
        {
            if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sputc(__c), traits_type::eof()))
                __sbuf_ = 0;
            return *this;
        }
    _LIBCUDACXX_INLINE_VISIBILITY ostreambuf_iterator& operator*()     {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY ostreambuf_iterator& operator++()    {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY ostreambuf_iterator& operator++(int) {return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY bool failed() const _NOEXCEPT {return __sbuf_ == 0;}

    template <class _Ch, class _Tr>
    friend
    _LIBCUDACXX_HIDDEN _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER
    ostreambuf_iterator<_Ch, _Tr>
    __pad_and_output(ostreambuf_iterator<_Ch, _Tr> __s,
                     const _Ch* __ob, const _Ch* __op, const _Ch* __oe,
                     ios_base& __iob, _Ch __fl);
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_OSTREAMBUF_ITERATOR_H
