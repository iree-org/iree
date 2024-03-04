// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ISTREAMBUF_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_ISTREAMBUF_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__iterator/iterator.h"
#include "../__iterator/iterator_traits.h"
#include "../iosfwd"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_LIBCUDACXX_SUPPRESS_DEPRECATED_PUSH
template<class _CharT, class _Traits>
class _LIBCUDACXX_TEMPLATE_VIS istreambuf_iterator
#if _LIBCUDACXX_STD_VER <= 14 || !defined(_LIBCUDACXX_ABI_NO_ITERATOR_BASES)
    : public iterator<input_iterator_tag, _CharT,
                      typename _Traits::off_type, _CharT*,
                      _CharT>
#endif
{
_LIBCUDACXX_SUPPRESS_DEPRECATED_POP
public:
    typedef _CharT                          char_type;
    typedef _Traits                         traits_type;
    typedef typename _Traits::int_type      int_type;
    typedef basic_streambuf<_CharT,_Traits> streambuf_type;
    typedef basic_istream<_CharT,_Traits>   istream_type;
private:
    mutable streambuf_type* __sbuf_;

    class __proxy
    {
        char_type __keep_;
        streambuf_type* __sbuf_;
        _LIBCUDACXX_INLINE_VISIBILITY __proxy(char_type __c, streambuf_type* __s)
            : __keep_(__c), __sbuf_(__s) {}
        friend class istreambuf_iterator;
    public:
        _LIBCUDACXX_INLINE_VISIBILITY char_type operator*() const {return __keep_;}
    };

    _LIBCUDACXX_INLINE_VISIBILITY
    bool __test_for_eof() const
    {
        if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sgetc(), traits_type::eof()))
            __sbuf_ = 0;
        return __sbuf_ == 0;
    }
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR istreambuf_iterator() _NOEXCEPT : __sbuf_(0) {}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator(istream_type& __s) _NOEXCEPT
        : __sbuf_(__s.rdbuf()) {}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator(streambuf_type* __s) _NOEXCEPT
        : __sbuf_(__s) {}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator(const __proxy& __p) _NOEXCEPT
        : __sbuf_(__p.__sbuf_) {}

    _LIBCUDACXX_INLINE_VISIBILITY char_type  operator*() const
        {return static_cast<char_type>(__sbuf_->sgetc());}
    _LIBCUDACXX_INLINE_VISIBILITY istreambuf_iterator& operator++()
        {
            __sbuf_->sbumpc();
            return *this;
        }
    _LIBCUDACXX_INLINE_VISIBILITY __proxy              operator++(int)
        {
            return __proxy(__sbuf_->sbumpc(), __sbuf_);
        }

    _LIBCUDACXX_INLINE_VISIBILITY bool equal(const istreambuf_iterator& __b) const
        {return __test_for_eof() == __b.__test_for_eof();}
};

template <class _CharT, class _Traits>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool operator==(const istreambuf_iterator<_CharT,_Traits>& __a,
                const istreambuf_iterator<_CharT,_Traits>& __b)
                {return __a.equal(__b);}

template <class _CharT, class _Traits>
inline _LIBCUDACXX_INLINE_VISIBILITY
bool operator!=(const istreambuf_iterator<_CharT,_Traits>& __a,
                const istreambuf_iterator<_CharT,_Traits>& __b)
                {return !__a.equal(__b);}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ISTREAMBUF_ITERATOR_H
