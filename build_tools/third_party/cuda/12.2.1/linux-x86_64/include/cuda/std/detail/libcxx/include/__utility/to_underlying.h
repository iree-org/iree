// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___UTILITY_TO_UNDERLYING_H
#define _LIBCUDACXX___UTILITY_TO_UNDERLYING_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/underlying_type.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#ifndef _LIBCUDACXX_CXX03_LANG
template <class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY constexpr typename underlying_type<_Tp>::type
__to_underlying(_Tp __val) noexcept {
  return static_cast<typename underlying_type<_Tp>::type>(__val);
}
#endif // !_LIBCUDACXX_CXX03_LANG

#if _LIBCUDACXX_STD_VER > 20
template <class _Tp>
_LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY constexpr underlying_type_t<_Tp>
to_underlying(_Tp __val) noexcept {
  return _CUDA_VSTD::__to_underlying(__val);
}
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___UTILITY_TO_UNDERLYING_H
