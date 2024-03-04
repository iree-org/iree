//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_RESULT_OF_H
#define _LIBCUDACXX___TYPE_TRAITS_RESULT_OF_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__functional/invoke.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// result_of

#if _LIBCUDACXX_STD_VER <= 17 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS)
template <class _Callable> class _LIBCUDACXX_DEPRECATED_IN_CXX17 result_of;

template <class _Fp, class ..._Args>
class _LIBCUDACXX_TEMPLATE_VIS result_of<_Fp(_Args...)>
    : public __invoke_of<_Fp, _Args...>
{
};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using result_of_t _LIBCUDACXX_DEPRECATED_IN_CXX17 = typename result_of<_Tp>::type;
#endif // _LIBCUDACXX_STD_VER > 11
#endif // _LIBCUDACXX_STD_VER <= 17 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_TYPE_TRAITS)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_RESULT_OF_H
