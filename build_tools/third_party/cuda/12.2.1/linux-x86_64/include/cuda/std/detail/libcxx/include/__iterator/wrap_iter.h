// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_WRAP_ITER_H
#define _LIBCUDACXX___ITERATOR_WRAP_ITER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__debug"
#include "../__iterator/iterator_traits.h"
#include "../__memory/addressof.h"
#include "../__memory/pointer_traits.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/is_trivially_copy_assignable.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD


// __wrap_iter

template <class _Iter> class __wrap_iter;

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator==(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator<(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator!=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator>(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator>=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator<=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

#ifndef _LIBCUDACXX_CXX03_LANG
template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
auto
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
-> decltype(__x.base() - __y.base());
#else
template <class _Iter1, class _Iter2>
_LIBCUDACXX_INLINE_VISIBILITY
typename __wrap_iter<_Iter1>::difference_type
operator-(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;
#endif

template <class _Iter>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
__wrap_iter<_Iter>
operator+(typename __wrap_iter<_Iter>::difference_type, __wrap_iter<_Iter>) _NOEXCEPT;

template <class _Ip, class _Op> _Op _LIBCUDACXX_INLINE_VISIBILITY copy(_Ip, _Ip, _Op);
template <class _B1, class _B2> _B2 _LIBCUDACXX_INLINE_VISIBILITY copy_backward(_B1, _B1, _B2);
template <class _Ip, class _Op> _Op _LIBCUDACXX_INLINE_VISIBILITY move(_Ip, _Ip, _Op);
template <class _B1, class _B2> _B2 _LIBCUDACXX_INLINE_VISIBILITY move_backward(_B1, _B1, _B2);

#if _LIBCUDACXX_DEBUG_LEVEL < 2

template <class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
__enable_if_t
<
    is_trivially_copy_assignable<_Tp>::value,
    _Tp*
>
__unwrap_iter(__wrap_iter<_Tp*>);

#else

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
__enable_if_t
<
    is_trivially_copy_assignable<_Tp>::value,
    __wrap_iter<_Tp*>
>
__unwrap_iter(__wrap_iter<_Tp*> __i);

#endif

template <class _Iter>
class __wrap_iter
{
public:
    typedef _Iter                                                      iterator_type;
    typedef typename iterator_traits<iterator_type>::iterator_category iterator_category;
    typedef typename iterator_traits<iterator_type>::value_type        value_type;
    typedef typename iterator_traits<iterator_type>::difference_type   difference_type;
    typedef typename iterator_traits<iterator_type>::pointer           pointer;
    typedef typename iterator_traits<iterator_type>::reference         reference;
private:
    iterator_type __i;
public:
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter() _NOEXCEPT
#if _LIBCUDACXX_STD_VER > 11
                : __i{}
#endif
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        __get_db()->__insert_i(this);
#endif
    }
    template <class _Up> _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
        __wrap_iter(const __wrap_iter<_Up>& __u,
            __enable_if_t<is_convertible<_Up, iterator_type>::value>* = 0) _NOEXCEPT
            : __i(__u.base())
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        __get_db()->__iterator_copy(this, &__u);
#endif
    }
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
    __wrap_iter(const __wrap_iter& __x)
        : __i(__x.base())
    {
        __get_db()->__iterator_copy(this, &__x);
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
    __wrap_iter& operator=(const __wrap_iter& __x)
    {
        if (this != &__x)
        {
            __get_db()->__iterator_copy(this, &__x);
            __i = __x.__i;
        }
        return *this;
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
    ~__wrap_iter()
    {
        __get_db()->__erase_i(this);
    }
#endif
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG reference operator*() const _NOEXCEPT
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        _LIBCUDACXX_ASSERT(__get_const_db()->__dereferenceable(this),
                       "Attempted to dereference a non-dereferenceable iterator");
#endif
        return *__i;
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG pointer  operator->() const _NOEXCEPT
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        _LIBCUDACXX_ASSERT(__get_const_db()->__dereferenceable(this),
                       "Attempted to dereference a non-dereferenceable iterator");
#endif
        return (pointer)_CUDA_VSTD::addressof(*__i);
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter& operator++() _NOEXCEPT
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        _LIBCUDACXX_ASSERT(__get_const_db()->__dereferenceable(this),
                       "Attempted to increment non-incrementable iterator");
#endif
        ++__i;
        return *this;
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter  operator++(int) _NOEXCEPT
        {__wrap_iter __tmp(*this); ++(*this); return __tmp;}

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter& operator--() _NOEXCEPT
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        _LIBCUDACXX_ASSERT(__get_const_db()->__decrementable(this),
                       "Attempted to decrement non-decrementable iterator");
#endif
        --__i;
        return *this;
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter  operator--(int) _NOEXCEPT
        {__wrap_iter __tmp(*this); --(*this); return __tmp;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter  operator+ (difference_type __n) const _NOEXCEPT
        {__wrap_iter __w(*this); __w += __n; return __w;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter& operator+=(difference_type __n) _NOEXCEPT
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        _LIBCUDACXX_ASSERT(__get_const_db()->__addable(this, __n),
                   "Attempted to add/subtract iterator outside of valid range");
#endif
        __i += __n;
        return *this;
    }
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter  operator- (difference_type __n) const _NOEXCEPT
        {return *this + (-__n);}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter& operator-=(difference_type __n) _NOEXCEPT
        {*this += -__n; return *this;}
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG reference    operator[](difference_type __n) const _NOEXCEPT
    {
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
        _LIBCUDACXX_ASSERT(__get_const_db()->__subscriptable(this, __n),
                   "Attempted to subscript iterator outside of valid range");
#endif
        return __i[__n];
    }

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG iterator_type base() const _NOEXCEPT {return __i;}

private:
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter(const void* __p, iterator_type __x) : __i(__x)
    {
        __get_db()->__insert_ic(this, __p);
    }
#else
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG __wrap_iter(iterator_type __x) _NOEXCEPT : __i(__x) {}
#endif

    template <class _Up> friend class __wrap_iter;
    template <class _CharT, class _Traits, class _Alloc> friend class basic_string;
    template <class _Tp, class _Alloc> friend class vector;
    template <class _Tp, size_t> friend class span;

    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    bool
    operator==(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    bool
    operator<(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    bool
    operator!=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    bool
    operator>(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    bool
    operator>=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    bool
    operator<=(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;

#ifndef _LIBCUDACXX_CXX03_LANG
    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    auto
    operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
    -> decltype(__x.base() - __y.base());
#else
    template <class _Iter1, class _Iter2>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    typename __wrap_iter<_Iter1>::difference_type
    operator-(const __wrap_iter<_Iter1>&, const __wrap_iter<_Iter2>&) _NOEXCEPT;
#endif

    template <class _Iter1>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    __wrap_iter<_Iter1>
    operator+(typename __wrap_iter<_Iter1>::difference_type, __wrap_iter<_Iter1>) _NOEXCEPT;

    template <class _Ip, class _Op> _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER friend _Op copy(_Ip, _Ip, _Op);
    template <class _B1, class _B2> _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER friend _B2 copy_backward(_B1, _B1, _B2);
    template <class _Ip, class _Op> _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER friend _Op move(_Ip, _Ip, _Op);
    template <class _B1, class _B2> _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER friend _B2 move_backward(_B1, _B1, _B2);

#if _LIBCUDACXX_DEBUG_LEVEL < 2
    template <class _Tp>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_CONSTEXPR_IF_NODEBUG friend
    __enable_if_t
    <
        is_trivially_copy_assignable<_Tp>::value,
        _Tp*
    >
    __unwrap_iter(__wrap_iter<_Tp*>);
#else
  template <class _Tp>
  inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
  __enable_if_t
  <
      is_trivially_copy_assignable<_Tp>::value,
      __wrap_iter<_Tp*>
  >
  __unwrap_iter(__wrap_iter<_Tp*> __i);
#endif
};

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator==(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
    return __x.base() == __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator<(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
    _LIBCUDACXX_ASSERT(__get_const_db()->__less_than_comparable(&__x, &__y),
                   "Attempted to compare incomparable iterators");
#endif
    return __x.base() < __y.base();
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
    return !(__x == __y);
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
    return __y < __x;
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
    return !(__x < __y);
}

template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
    return !(__y < __x);
}

template <class _Iter1>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator!=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT
{
    return !(__x == __y);
}

template <class _Iter1>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator>(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT
{
    return __y < __x;
}

template <class _Iter1>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator>=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT
{
    return !(__x < __y);
}

template <class _Iter1>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
bool
operator<=(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter1>& __y) _NOEXCEPT
{
    return !(__y < __x);
}

#ifndef _LIBCUDACXX_CXX03_LANG
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
auto
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
-> decltype(__x.base() - __y.base())
{
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
    _LIBCUDACXX_ASSERT(__get_const_db()->__less_than_comparable(&__x, &__y),
                   "Attempted to subtract incompatible iterators");
#endif
    return __x.base() - __y.base();
}
#else
template <class _Iter1, class _Iter2>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
typename __wrap_iter<_Iter1>::difference_type
operator-(const __wrap_iter<_Iter1>& __x, const __wrap_iter<_Iter2>& __y) _NOEXCEPT
{
#ifdef _LIBCUDACXX_ENABLE_DEBUG_MODE
    _LIBCUDACXX_ASSERT(__get_const_db()->__less_than_comparable(&__x, &__y),
                   "Attempted to subtract incompatible iterators");
#endif
    return __x.base() - __y.base();
}
#endif

template <class _Iter>
inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_IF_NODEBUG
__wrap_iter<_Iter>
operator+(typename __wrap_iter<_Iter>::difference_type __n,
          __wrap_iter<_Iter> __x) _NOEXCEPT
{
    __x += __n;
    return __x;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_WRAP_ITER_H
