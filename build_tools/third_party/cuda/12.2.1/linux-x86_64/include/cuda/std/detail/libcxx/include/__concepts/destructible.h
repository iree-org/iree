//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
#define _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/is_object.h"
#include "../__type_traits/is_nothrow_destructible.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 11 && _GNUC_VER < 1100

template<class _Tp>
_LIBCUDACXX_CONCEPT destructible = is_nothrow_destructible_v<_Tp>;

#else
// [concept.destructible]
#if _LIBCUDACXX_STD_VER > 17

template<class _Tp>
constexpr bool __destructible_impl = false;

template<class _Tp>
    requires requires(_Tp& __t) { { __t.~_Tp() } noexcept; }
constexpr bool __destructible_impl<_Tp> = true;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Tp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __destructible_impl_req_,
  requires(_Tp& __t)(
    requires(is_object_v<_Tp>),
    requires(noexcept(__t.~_Tp()))
  ));

template<class _Tp>
constexpr bool __destructible_impl = _LIBCUDACXX_FRAGMENT(__destructible_impl_req_, _Tp);
#endif // _LIBCUDACXX_STD_VER > 11

#if _LIBCUDACXX_STD_VER > 11
template<class _Tp>
constexpr bool __destructible = __destructible_impl<_Tp>;

template<class _Tp>
constexpr bool __destructible<_Tp&> = true;

template<class _Tp>
constexpr bool __destructible<_Tp&&> = true;

template<class _Tp, size_t _Nm>
constexpr bool __destructible<_Tp[_Nm]> = __destructible<_Tp>;

template<class _Tp>
_LIBCUDACXX_CONCEPT destructible = __destructible<_Tp>;

#endif // _LIBCUDACXX_STD_VER > 11
#endif // _GNUC_VER < 1100

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_DESTRUCTIBLE_H
