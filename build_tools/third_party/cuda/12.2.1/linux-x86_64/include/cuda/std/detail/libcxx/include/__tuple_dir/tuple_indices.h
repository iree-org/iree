//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_MAKE_TUPLE_INDICES_H
#define _LIBCUDACXX___TUPLE_MAKE_TUPLE_INDICES_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__utility/integer_sequence.h"
#include "../cstddef"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

#ifndef _LIBCUDACXX_CXX03_LANG

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ep, size_t _Sp = 0>
struct __make_tuple_indices
{
    static_assert(_Sp <= _Ep, "__make_tuple_indices input error");
    typedef __make_indices_imp<_Ep, _Sp> type;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX_CXX03_LANG

#endif // _LIBCUDACXX___TUPLE_MAKE_TUPLE_INDICES_H
