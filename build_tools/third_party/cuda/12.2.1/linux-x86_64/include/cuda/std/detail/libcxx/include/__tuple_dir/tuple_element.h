//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H
#define _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__tuple_dir/tuple_indices.h"
#include "../__tuple_dir/tuple_types.h"
#include "../__type_traits/add_const.h"
#include "../__type_traits/add_cv.h"
#include "../__type_traits/add_volatile.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ip, class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS tuple_element;

template <size_t _Ip, class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, const _Tp>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename add_const<typename tuple_element<_Ip, _Tp>::type>::type type;
};

template <size_t _Ip, class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, volatile _Tp>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename add_volatile<typename tuple_element<_Ip, _Tp>::type>::type type;
};

template <size_t _Ip, class _Tp>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, const volatile _Tp>
{
    typedef _LIBCUDACXX_NODEBUG_TYPE typename add_cv<typename tuple_element<_Ip, _Tp>::type>::type type;
};

#ifndef _LIBCUDACXX_CXX03_LANG

#ifdef _LIBCUDACXX_COMPILER_MSVC

namespace __indexer_detail {

template <size_t _Idx, class ..._Types>
struct _nth_of;

template <class _Head, class ..._Tail>
struct _nth_of<0, _Head, _Tail...> {
    using type = _Head;
};

template <size_t _Idx, class _Head, class ..._Tail>
struct _nth_of<_Idx, _Head, _Tail...> {
    using type = typename _nth_of<_Idx-1, _Tail...>::type;
};

template <size_t _Idx, class ..._Types>
struct nth_of {
    static_assert(_Idx < sizeof...(_Types), "");
    using _impl = _nth_of<_Idx, _Types...>;
    using type = typename _impl::type;
};

} // namespace __indexer_detail

template <size_t _Idx, class ..._Types>
using __type_pack_element _LIBCUDACXX_NODEBUG_TYPE = typename __indexer_detail::nth_of<_Idx, _Types...>::type;

#elif !__has_builtin(__type_pack_element)

namespace __indexer_detail {

template <size_t _Idx, class _Tp>
struct __indexed { using type _LIBCUDACXX_NODEBUG_TYPE = _Tp; };

template <class _Types, class _Indexes> struct __indexer;

template <class ..._Types, size_t ..._Idx>
struct __indexer<__tuple_types<_Types...>, __tuple_indices<_Idx...>>
    : __indexed<_Idx, _Types>...
{};

template <size_t _Idx, class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
__indexed<_Idx, _Tp> __at_index(__indexed<_Idx, _Tp> const&);

} // namespace __indexer_detail

template <size_t _Idx, class ..._Types>
using __type_pack_element _LIBCUDACXX_NODEBUG_TYPE = typename decltype(
    __indexer_detail::__at_index<_Idx>(
        __indexer_detail::__indexer<
            __tuple_types<_Types...>,
            typename __make_tuple_indices<sizeof...(_Types)>::type
        >{})
  )::type;
#endif

template <size_t _Ip, class ..._Types>
struct _LIBCUDACXX_TEMPLATE_VIS tuple_element<_Ip, __tuple_types<_Types...> >
{
    static_assert(_Ip < sizeof...(_Types), "tuple_element index out of range");
    typedef _LIBCUDACXX_NODEBUG_TYPE __type_pack_element<_Ip, _Types...> type;
};

#if _LIBCUDACXX_STD_VER > 11
template <size_t _Ip, class ..._Tp>
using tuple_element_t _LIBCUDACXX_NODEBUG_TYPE = typename tuple_element <_Ip, _Tp...>::type;
#endif

#endif // _LIBCUDACXX_CXX03_LANG

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TUPLE_TUPLE_ELEMENT_H
