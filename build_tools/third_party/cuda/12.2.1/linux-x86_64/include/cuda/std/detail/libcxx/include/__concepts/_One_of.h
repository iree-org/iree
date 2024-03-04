//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_ONE_OF_H
#define _LIBCUDACXX___CONCEPTS_ONE_OF_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/disjunction.h"
#include "../__type_traits/is_same.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 14
template <class _Ty, class... _Others>
_LIBCUDACXX_CONCEPT _One_of = (is_same_v<_Ty, _Others> || ...);
#elif _LIBCUDACXX_STD_VER > 11
template <class _Ty, class... _Others>
_LIBCUDACXX_CONCEPT _One_of = _LIBCUDACXX_TRAIT(disjunction, is_same<_Ty, _Others>...);
#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_ONE_OF_H