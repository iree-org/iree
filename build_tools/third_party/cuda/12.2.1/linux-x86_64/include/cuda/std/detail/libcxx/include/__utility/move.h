// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_MOVE_H
#define _LIBCUDACXX___UTILITY_MOVE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/conditional.h"
#include "../__type_traits/is_copy_constructible.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/remove_reference.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR __libcpp_remove_reference_t<_Tp>&&
move(_Tp&& __t) _NOEXCEPT {
  typedef _LIBCUDACXX_NODEBUG_TYPE __libcpp_remove_reference_t<_Tp> _Up;
  return static_cast<_Up&&>(__t);
}

template <class _Tp>
using __move_if_noexcept_result_t =
    __conditional_t<!is_nothrow_move_constructible<_Tp>::value && is_copy_constructible<_Tp>::value, const _Tp&, _Tp&&>;

template <class _Tp>
_LIBCUDACXX_NODISCARD_EXT inline _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX11 __move_if_noexcept_result_t<_Tp>
move_if_noexcept(_Tp& __x) _NOEXCEPT {
  return _CUDA_VSTD::move(__x);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_MOVE_H
