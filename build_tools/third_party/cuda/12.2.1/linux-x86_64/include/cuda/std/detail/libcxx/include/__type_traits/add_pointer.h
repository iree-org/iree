//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_ADD_POINTER_H
#define _LIBCUDACXX___TYPE_TRAITS_ADD_POINTER_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#include "../__type_traits/is_referenceable.h"
#include "../__type_traits/is_same.h"
#include "../__type_traits/is_void.h"
#include "../__type_traits/remove_cv.h"
#include "../__type_traits/remove_reference.h"

#if defined(_LIBCUDACXX_USE_PRAGMA_GCC_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_LIBCUDACXX_ADD_POINTER) && !defined(_LIBCUDACXX_USE_ADD_POINTER_FALLBACK)

template <class _Tp>
using __add_pointer_t = _LIBCUDACXX_ADD_POINTER(_Tp);

#else
template <class _Tp,
          bool = __libcpp_is_referenceable<_Tp>::value || is_void<_Tp>::value>
struct __add_pointer_impl {
  typedef _LIBCUDACXX_NODEBUG_TYPE __libcpp_remove_reference_t<_Tp>* type;
};
template <class _Tp> struct __add_pointer_impl<_Tp, false>
    {typedef _LIBCUDACXX_NODEBUG_TYPE _Tp type;};

template <class _Tp>
using __add_pointer_t = typename __add_pointer_impl<_Tp>::type;

#endif // defined(_LIBCUDACXX_ADD_POINTER) && !defined(_LIBCUDACXX_USE_ADD_POINTER_FALLBACK)

template <class _Tp>
struct add_pointer {
  using type _LIBCUDACXX_NODEBUG_TYPE = __add_pointer_t<_Tp>;
};

#if _LIBCUDACXX_STD_VER > 11
template <class _Tp> using add_pointer_t = __add_pointer_t<_Tp>;
#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_ADD_POINTER_H
