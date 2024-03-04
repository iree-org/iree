//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_CONVERTIBLE_TO_H
#define _LIBCUDACXX___CONCEPTS_CONVERTIBLE_TO_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/is_convertible.h"
#include "../__utility/declval.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _LIBCUDACXX_STD_VER > 17

// [concept.convertible]

template<class _From, class _To>
concept convertible_to =
  is_convertible_v<_From, _To> &&
  requires {
    static_cast<_To>(_CUDA_VSTD::declval<_From>());
  };

#elif _LIBCUDACXX_STD_VER > 11

// We cannot put this conversion check with the other constraint, as types with deleted operator will break here
template<class _From, class _To>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __test_conversion_,
  requires()(
    static_cast<_To>(_CUDA_VSTD::declval<_From>())
  ));

template<class _From, class _To>
_LIBCUDACXX_CONCEPT __test_conversion = _LIBCUDACXX_FRAGMENT(__test_conversion_, _From, _To);

template<class _From, class _To>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __convertible_to_,
  requires()(
    requires(_LIBCUDACXX_TRAIT(is_convertible, _From, _To)),
    requires(__test_conversion<_From, _To>)
  ));

template<class _From, class _To>
_LIBCUDACXX_CONCEPT convertible_to = _LIBCUDACXX_FRAGMENT(__convertible_to_, _From, _To);

#endif // _LIBCUDACXX_STD_VER > 11

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_CONVERTIBLE_TO_H
