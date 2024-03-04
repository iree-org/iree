// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_IDENTITY_H
#define _LIBCUDACXX___FUNCTIONAL_IDENTITY_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__utility/forward.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __identity {
  template <class _Tp>
  _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR _Tp&& operator()(_Tp&& __t) const _NOEXCEPT {
    return _CUDA_VSTD::forward<_Tp>(__t);
  }

  using is_transparent = void;
};

#if _LIBCUDACXX_STD_VER > 11

struct identity {
    template<class _Tp>
    _LIBCUDACXX_NODISCARD_EXT _LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp&& operator()(_Tp&& __t) const noexcept
    {
        return _CUDA_VSTD::forward<_Tp>(__t);
    }

    using is_transparent = void;
};
#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___FUNCTIONAL_IDENTITY_H
