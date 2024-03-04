// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ITERATOR_SIZE_H
#define _LIBCUDACXX___ITERATOR_SIZE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/common_type.h"
#include "../__type_traits/make_signed.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11
template <class _Cont>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr auto size(const _Cont& __c)
_NOEXCEPT_(noexcept(__c.size()))
-> decltype        (__c.size())
{ return            __c.size(); }

template <class _Tp, size_t _Sz>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr size_t size(const _Tp (&)[_Sz]) noexcept { return _Sz; }
#endif // _LIBCUDACXX_STD_VER > 11

#if _LIBCUDACXX_STD_VER > 17
template <class _Cont>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr auto ssize(const _Cont& __c)
_NOEXCEPT_(noexcept(static_cast<common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>>(__c.size())))
->                              common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>
{ return            static_cast<common_type_t<ptrdiff_t, make_signed_t<decltype(__c.size())>>>(__c.size()); }

// GCC complains about the implicit conversion from ptrdiff_t to size_t in
// the array bound.
# if defined(__GNUC__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wsign-conversion"
# endif
template <class _Tp, ptrdiff_t _Sz>
_LIBCUDACXX_INLINE_VISIBILITY
constexpr ptrdiff_t ssize(const _Tp (&)[_Sz]) noexcept { return _Sz; }
# if defined(__GNUC__)
#   pragma GCC diagnostic pop
# endif
#endif  // _LIBCUDACXX_STD_VER > 17

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ITERATOR_SIZE_H
