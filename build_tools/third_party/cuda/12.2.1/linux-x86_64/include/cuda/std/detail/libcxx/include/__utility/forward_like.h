// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_FORWARD_LIKE_H
#define _LIBCUDACXX___UTILITY_FORWARD_LIKE_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/conditional.h"
#include "../__type_traits/is_const.h"
#include "../__type_traits/is_reference.h"
#include "../__type_traits/remove_reference.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 20

template <class _Ap, class _Bp>
using _CopyConst = _If<is_const_v<_Ap>, const _Bp, _Bp>;

template <class _Ap, class _Bp>
using _OverrideRef = _If<is_rvalue_reference_v<_Ap>, remove_reference_t<_Bp>&&, _Bp&>;

template <class _Ap, class _Bp>
using _ForwardLike = _OverrideRef<_Ap&&, _CopyConst<remove_reference_t<_Ap>, remove_reference_t<_Bp>>>;

template <class _Tp, class _Up>
_LIBCUDACXX_NODISCARD_ATTRIBUTE _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY
constexpr auto forward_like(_Up&& __ux) noexcept -> _ForwardLike<_Tp, _Up> {
  return static_cast<_ForwardLike<_Tp, _Up>>(__ux);
}

#endif // _LIBCUDACXX_STD_VER > 20

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_FORWARD_LIKE_H
