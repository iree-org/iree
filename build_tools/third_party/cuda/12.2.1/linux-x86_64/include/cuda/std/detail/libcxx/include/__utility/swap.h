//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_SWAP_H
#define _LIBCUDACXX___UTILITY_SWAP_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/enable_if.h"
#include "../__type_traits/is_move_assignable.h"
#include "../__type_traits/is_move_constructible.h"
#include "../__type_traits/is_nothrow_move_assignable.h"
#include "../__type_traits/is_nothrow_move_constructible.h"
#include "../__type_traits/is_swappable.h"
#include "../__utility/move.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Tp>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __swap_result_t<_Tp>
swap(_Tp& __x, _Tp& __y) _NOEXCEPT_(is_nothrow_move_constructible<_Tp>::value &&
                                    is_nothrow_move_assignable<_Tp>::value) {
  _Tp __t(_CUDA_VSTD::move(__x));
  __x = _CUDA_VSTD::move(__y);
  __y = _CUDA_VSTD::move(__t);
}

template <class _Tp, size_t _Np>
inline _LIBCUDACXX_INLINE_VISIBILITY
_LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<__is_swappable<_Tp>::value>
swap(_Tp (&__a)[_Np], _Tp (&__b)[_Np]) _NOEXCEPT_(__is_nothrow_swappable<_Tp>::value) {
  for (size_t __i = 0; __i != _Np; ++__i) {
    swap(__a[__i], __b[__i]);
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_SWAP_H
