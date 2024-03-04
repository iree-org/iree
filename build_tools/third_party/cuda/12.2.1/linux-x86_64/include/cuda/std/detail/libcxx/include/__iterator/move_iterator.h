// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H
#define _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__iterator/iterator_traits.h"
#include "../__type_traits/conditional.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_reference.h"
#include "../__utility/move.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD


template <class _Iter>
class _LIBCUDACXX_TEMPLATE_VIS move_iterator
{
private:
    _Iter __i;
public:
    typedef _Iter                                            iterator_type;
    typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;
    typedef typename iterator_traits<iterator_type>::value_type value_type;
    typedef typename iterator_traits<iterator_type>::difference_type difference_type;
    typedef iterator_type pointer;
#ifndef _LIBCUDACXX_CXX03_LANG
    typedef typename iterator_traits<iterator_type>::reference __reference;
    typedef __conditional_t<
            is_reference<__reference>::value,
            __libcpp_remove_reference_t<__reference>&&,
            __reference
        > reference;
#else
    typedef typename iterator_traits<iterator_type>::reference reference;
#endif

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator() : __i() {}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    explicit move_iterator(_Iter __x) : __i(__x) {}
    template <class _Up>
      _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
      move_iterator(const move_iterator<_Up>& __u) : __i(__u.base()) {}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14 _Iter base() const {return __i;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    reference operator*() const { return static_cast<reference>(*__i); }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    pointer  operator->() const { return __i;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator& operator++() {++__i; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator  operator++(int) {move_iterator __tmp(*this); ++__i; return __tmp;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator& operator--() {--__i; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator  operator--(int) {move_iterator __tmp(*this); --__i; return __tmp;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator  operator+ (difference_type __n) const {return move_iterator(__i + __n);}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator& operator+=(difference_type __n) {__i += __n; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator  operator- (difference_type __n) const {return move_iterator(__i - __n);}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    move_iterator& operator-=(difference_type __n) {__i -= __n; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
    reference operator[](difference_type __n) const { return _CUDA_VSTD::move(__i[__n]); }
};

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
bool
operator==(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
bool
operator<(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
bool
operator!=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() != __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
bool
operator>(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() > __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
bool
operator>=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() >= __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
bool
operator<=(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() <= __y.base();
}

#ifndef _LIBCUDACXX_CXX03_LANG
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
auto
operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
-> decltype(__x.base() - __y.base())
{
    return __x.base() - __y.base();
}
#else
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY
typename move_iterator<_Iter1>::difference_type
operator-(const move_iterator<_Iter1>& __x, const move_iterator<_Iter2>& __y)
{
    return __x.base() - __y.base();
}
#endif

template <class _Iter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
move_iterator<_Iter>
operator+(typename move_iterator<_Iter>::difference_type __n, const move_iterator<_Iter>& __x)
{
    return move_iterator<_Iter>(__x.base() + __n);
}

template <class _Iter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX14
move_iterator<_Iter>
make_move_iterator(_Iter __i)
{
    return move_iterator<_Iter>(__i);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_MOVE_ITERATOR_H
