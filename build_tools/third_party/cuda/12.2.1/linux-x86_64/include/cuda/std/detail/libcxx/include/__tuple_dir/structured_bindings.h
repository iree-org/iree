//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H
#define _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H

#ifdef __cuda_std__

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmismatched-tags"
#endif

#if !defined(__CUDACC_RTC__)
// Fetch utility to get primary template for ::std::tuple_size necessary for the specialization of
// ::std::tuple_size<cuda::std::tuple> to enable structured bindings.
// See https://github.com/NVIDIA/libcudacxx/issues/316
#include <utility>
#endif

#include "../__fwd/array.h"
#include "../__fwd/pair.h"
#include "../__fwd/tuple.h"
#include "../__tuple_dir/tuple_element.h"
#include "../__tuple_dir/tuple_size.h"
#include "../__type_traits/integral_constant.h"

// This is a workaround for the fact that structured bindings require that the specializations of
// `tuple_size` and `tuple_element` reside in namespace std (https://eel.is/c++draft/dcl.struct.bind#4).
// See https://github.com/NVIDIA/libcudacxx/issues/316 for a short discussion
#if _LIBCUDACXX_STD_VER > 14
namespace std {
#if defined(__CUDACC_RTC__)
    template <class... _Tp>
    struct tuple_size;

    template<size_t _Ip, class... _Tp>
    struct tuple_element;
#endif

    template <class _Tp, size_t _Size>
    struct tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
    {};

    template <class _Tp, size_t _Size>
    struct tuple_size<const _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
    {};

    template <class _Tp, size_t _Size>
    struct tuple_size<volatile _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
    {};

    template <class _Tp, size_t _Size>
    struct tuple_size<const volatile _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::array<_Tp, _Size>>
    {};

    template<size_t _Ip, class _Tp, size_t _Size>
    struct tuple_element<_Ip, _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::array<_Tp, _Size>>
    {};

    template<size_t _Ip, class _Tp, size_t _Size>
    struct tuple_element<_Ip, const _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_element<_Ip, const _CUDA_VSTD::array<_Tp, _Size>>
    {};

    template<size_t _Ip, class _Tp, size_t _Size>
    struct tuple_element<_Ip, volatile _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_element<_Ip, volatile _CUDA_VSTD::array<_Tp, _Size>>
    {};

    template<size_t _Ip, class _Tp, size_t _Size>
    struct tuple_element<_Ip, const volatile _CUDA_VSTD::array<_Tp, _Size>>
      : _CUDA_VSTD::tuple_element<_Ip, const volatile _CUDA_VSTD::array<_Tp, _Size>>
    {};

    template <class _Tp, class _Up>
    struct tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template <class _Tp, class _Up>
    struct tuple_size<const _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template <class _Tp, class _Up>
    struct tuple_size<volatile _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template <class _Tp, class _Up>
    struct tuple_size<const volatile _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template<size_t _Ip, class _Tp, class _Up>
    struct tuple_element<_Ip, _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template<size_t _Ip, class _Tp, class _Up>
    struct tuple_element<_Ip, const _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_element<_Ip, const _CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template<size_t _Ip, class _Tp, class _Up>
    struct tuple_element<_Ip, volatile _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_element<_Ip, volatile _CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template<size_t _Ip, class _Tp, class _Up>
    struct tuple_element<_Ip, const volatile _CUDA_VSTD::pair<_Tp, _Up>>
      : _CUDA_VSTD::tuple_element<_Ip, const volatile _CUDA_VSTD::pair<_Tp, _Up>>
    {};

    template <class... _Tp>
    struct tuple_size<_CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
    {};

    template <class... _Tp>
    struct tuple_size<const _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
    {};

    template <class... _Tp>
    struct tuple_size<volatile _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
    {};

    template <class... _Tp>
    struct tuple_size<const volatile _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_size<_CUDA_VSTD::tuple<_Tp...>>
    {};

    template<size_t _Ip, class... _Tp>
    struct tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_element<_Ip, _CUDA_VSTD::tuple<_Tp...>>
    {};

    template<size_t _Ip, class... _Tp>
    struct tuple_element<_Ip, const _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_element<_Ip, const _CUDA_VSTD::tuple<_Tp...>>
    {};

    template<size_t _Ip, class... _Tp>
    struct tuple_element<_Ip, volatile _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_element<_Ip, volatile _CUDA_VSTD::tuple<_Tp...>>
    {};

    template<size_t _Ip, class... _Tp>
    struct tuple_element<_Ip, const volatile _CUDA_VSTD::tuple<_Tp...>>
      : _CUDA_VSTD::tuple_element<_Ip, const volatile _CUDA_VSTD::tuple<_Tp...>>
    {};
}
#endif // _LIBCUDACXX_STD_VER > 14

#if defined(__clang__)
#pragma clang diagnostic pop
# endif

#endif // __cuda_std__

#endif // _LIBCUDACXX___TUPLE_STRUCTURED_BINDINGS_H
