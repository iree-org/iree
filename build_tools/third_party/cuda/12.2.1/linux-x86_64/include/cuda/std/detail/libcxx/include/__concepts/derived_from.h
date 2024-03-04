//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_DERIVED_FROM_H
#define _LIBCUDACXX___CONCEPTS_DERIVED_FROM_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/add_pointer.h"
#include "../__type_traits/is_base_of.h"
#include "../__type_traits/is_convertible.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.derived]

template<class _Dp, class _Bp>
concept derived_from =
  is_base_of_v<_Bp, _Dp> &&
  is_convertible_v<const volatile _Dp*, const volatile _Bp*>;

#elif _LIBCUDACXX_STD_VER > 11

template<class _Dp, class _Bp>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __derived_from_,
  requires()(
    requires(_LIBCUDACXX_TRAIT(is_base_of, _Bp, _Dp)),
    requires(_LIBCUDACXX_TRAIT(is_convertible, add_pointer_t<const volatile _Dp>, add_pointer_t<const volatile _Bp>))
  ));

template<class _Dp, class _Bp>
_LIBCUDACXX_CONCEPT derived_from = _LIBCUDACXX_FRAGMENT(__derived_from_, _Dp, _Bp);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_DERIVED_FROM_H
