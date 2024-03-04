// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_ITERATOR_TRAITS_H
#define _LIBCUDACXX___ITERATOR_ITERATOR_TRAITS_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__fwd/pair.h"
#include "../__type_traits/add_const.h"
#include "../__type_traits/is_convertible.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/void_t.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct _LIBCUDACXX_TEMPLATE_VIS input_iterator_tag {};
struct _LIBCUDACXX_TEMPLATE_VIS output_iterator_tag {};
struct _LIBCUDACXX_TEMPLATE_VIS forward_iterator_tag       : public input_iterator_tag {};
struct _LIBCUDACXX_TEMPLATE_VIS bidirectional_iterator_tag : public forward_iterator_tag {};
struct _LIBCUDACXX_TEMPLATE_VIS random_access_iterator_tag : public bidirectional_iterator_tag {};

template <class _Tp>
struct __has_iterator_typedefs
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Up>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_INLINE_VISIBILITY static __two __test(...);
    template <class _Up>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_INLINE_VISIBILITY static char __test(
        __void_t<typename _Up::iterator_category>* = 0,
    	__void_t<typename _Up::difference_type>* = 0,
    	__void_t<typename _Up::value_type>* = 0,
    	__void_t<typename _Up::reference>* = 0,
    	__void_t<typename _Up::pointer>* = 0);
public:
    static const bool value = sizeof(__test<_Tp>(0,0,0,0,0)) == 1;
};


template <class _Tp>
struct __has_iterator_category
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Up>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_INLINE_VISIBILITY static __two __test(...);
    template <class _Up>
    _LIBCUDACXX_EXECUTION_SPACE_SPECIFIER _LIBCUDACXX_INLINE_VISIBILITY static char __test(typename _Up::iterator_category* = 0);
public:
    static const bool value = sizeof(__test<_Tp>(0)) == 1;
};

template <class _Iter, bool> struct __iterator_traits_impl {};

template <class _Iter>
struct __iterator_traits_impl<_Iter, true>
{
    typedef typename _Iter::difference_type   difference_type;
    typedef typename _Iter::value_type        value_type;
    typedef typename _Iter::pointer           pointer;
    typedef typename _Iter::reference         reference;
    typedef typename _Iter::iterator_category iterator_category;
};

template <class _Iter, bool> struct __iterator_traits {};

template <class _Iter>
struct __iterator_traits<_Iter, true>
    :  __iterator_traits_impl
      <
        _Iter,
        is_convertible<typename _Iter::iterator_category, input_iterator_tag>::value ||
        is_convertible<typename _Iter::iterator_category, output_iterator_tag>::value
      >
{};

// iterator_traits<Iterator> will only have the nested types if Iterator::iterator_category
//    exists.  Else iterator_traits<Iterator> will be an empty class.  This is a
//    conforming extension which allows some programs to compile and behave as
//    the client expects instead of failing at compile time.

template <class _Iter>
struct _LIBCUDACXX_TEMPLATE_VIS iterator_traits
    : __iterator_traits<_Iter, __has_iterator_typedefs<_Iter>::value> {};

template<class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS iterator_traits<_Tp*>
{
    typedef ptrdiff_t difference_type;
    typedef __remove_cv_t<_Tp> value_type;
    typedef _Tp* pointer;
    typedef _Tp& reference;
    typedef random_access_iterator_tag iterator_category;
};

template <class _Tp, class _Up, bool = __has_iterator_category<iterator_traits<_Tp> >::value>
struct __has_iterator_category_convertible_to
    : public integral_constant<bool, is_convertible<typename iterator_traits<_Tp>::iterator_category, _Up>::value>
{};

template <class _Tp, class _Up>
struct __has_iterator_category_convertible_to<_Tp, _Up, false> : public false_type {};

template <class _Tp>
struct __is_cpp17_input_iterator : public __has_iterator_category_convertible_to<_Tp, input_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_forward_iterator : public __has_iterator_category_convertible_to<_Tp, forward_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_bidirectional_iterator : public __has_iterator_category_convertible_to<_Tp, bidirectional_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_random_access_iterator : public __has_iterator_category_convertible_to<_Tp, random_access_iterator_tag> {};

template <class _Tp>
struct __is_cpp17_exactly_input_iterator
    : public integral_constant<bool,
         __has_iterator_category_convertible_to<_Tp, input_iterator_tag>::value &&
        !__has_iterator_category_convertible_to<_Tp, forward_iterator_tag>::value> {};

#ifndef _LIBCUDACXX_HAS_NO_DEDUCTION_GUIDES
template<class _InputIterator>
using __iter_value_type = typename iterator_traits<_InputIterator>::value_type;

template<class _InputIterator>
using __iter_key_type = remove_const_t<typename iterator_traits<_InputIterator>::value_type::first_type>;

template<class _InputIterator>
using __iter_mapped_type = typename iterator_traits<_InputIterator>::value_type::second_type;

template<class _InputIterator>
using __iter_to_alloc_type = pair<
    add_const_t<typename iterator_traits<_InputIterator>::value_type::first_type>,
    typename iterator_traits<_InputIterator>::value_type::second_type>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_ITERATOR_TRAITS_H
