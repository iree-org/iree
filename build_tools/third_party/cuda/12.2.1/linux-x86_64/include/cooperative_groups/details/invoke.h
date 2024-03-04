/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#ifndef _CG_INVOKE_H
#define _CG_INVOKE_H

#include "info.h"
#include "helpers.h"

#if defined(_CG_CPP11_FEATURES)

_CG_BEGIN_NAMESPACE

namespace details {

    template <typename Group>
    struct _elect_group_supported : _CG_STL_NAMESPACE::false_type {};
#ifdef _CG_HAS_INSTR_ELECT
    template<>
    struct _elect_group_supported<coalesced_group> : _CG_STL_NAMESPACE::true_type {};
    template<unsigned int Size, typename Parent>
    struct _elect_group_supported<thread_block_tile<Size, Parent>> :
        _CG_STL_NAMESPACE::integral_constant<bool, (Size <= 32)> {};
#endif

    template <typename Group>
    struct elect_group_supported : public _elect_group_supported<details::remove_qual<Group>> {};

    template<typename Group>
    _CG_STATIC_QUALIFIER bool elect_one(const Group& group, unsigned int mask, unsigned int& leader_lane) {
        int is_leader = 0;
#ifdef _CG_HAS_INSTR_ELECT
        asm("{\n\t"
          " .reg .pred p;\n\t"
          "  elect.sync %0|p, %2;\n\t"
          " @p mov.s32 %1, 1;\n\t"
          "}"
          : "+r"(leader_lane), "+r"(is_leader) : "r" (mask));
#endif
        return is_leader;
    }

    template<bool UseElect>
    struct invoke_one_impl {};

    template<>
    struct invoke_one_impl<true> {
        template<typename Group, typename Fn, typename... Args>
        _CG_STATIC_QUALIFIER void invoke_one(const Group& group, Fn&& fn, Args&&... args) {
            auto mask = details::_coalesced_group_data_access::get_mask(group);
            unsigned int leader_lane = 0;

            if (elect_one(group, mask, leader_lane)) {
                _CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...);
            }
        }

        template<typename Group, typename Fn, typename... Args>
        _CG_STATIC_QUALIFIER auto invoke_one_broadcast(const Group& group, Fn&& fn, Args&&... args)
                -> typename _CG_STL_NAMESPACE::remove_reference<
                    decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...))>::type {

            using ResultType = decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...));
            details::remove_qual<ResultType> result;
            auto mask = details::_coalesced_group_data_access::get_mask(group);
            unsigned int leader_lane = 0;

            if (elect_one(group, mask, leader_lane)) {
                result = _CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...);
            }

            // Need to use low level api instead of group.shfl, because elect_one returns lane id, not group rank.
            return tile::shuffle_dispatch<ResultType>::shfl(result, mask, leader_lane, 32);
        }
    };

    template<>
    struct invoke_one_impl<false> {
        template<typename Group, typename Fn, typename... Args>
        _CG_STATIC_QUALIFIER void invoke_one(const Group& group, Fn&& fn, Args&&... args) {
            if (group.thread_rank() == 0) {
                _CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...);
            }
        }

        template<typename Group, typename Fn, typename... Args>
        _CG_STATIC_QUALIFIER auto invoke_one_broadcast(const Group& group, Fn&& fn, Args&&... args)
                -> typename _CG_STL_NAMESPACE::remove_reference<
                    decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...))>::type {

            using ResultType = decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...));
            details::remove_qual<ResultType> result;

            if (group.thread_rank() == 0) {
                result = _CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...);
            }

            return group.shfl(result, 0);
        }
    };


}; // namespace details

template<typename Group, typename Fn, typename... Args>
_CG_QUALIFIER void invoke_one(const Group& group, Fn&& fn, Args&&... args) {
    using impl = details::invoke_one_impl<details::elect_group_supported<Group>::value>;
    impl::invoke_one(group, _CG_STL_NAMESPACE::forward<Fn>(fn), _CG_STL_NAMESPACE::forward<Args>(args)...);
}

template<typename Fn, typename... Args>
_CG_QUALIFIER auto invoke_one_broadcast(const coalesced_group& group, Fn&& fn, Args&&... args)
        -> typename _CG_STL_NAMESPACE::remove_reference<
            decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...))>::type {

    using ResultType = decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...));
    static_assert(!_CG_STL_NAMESPACE::is_same<ResultType, void>::value,
                  "For invocables returning void invoke_one should be used instead");
    using impl = details::invoke_one_impl<details::elect_group_supported<coalesced_group>::value>;
    return impl::invoke_one_broadcast(group,
                                      _CG_STL_NAMESPACE::forward<Fn>(fn),
                                      _CG_STL_NAMESPACE::forward<Args>(args)...);
}

template<unsigned int Size, typename Parent, typename Fn, typename... Args>
_CG_QUALIFIER auto invoke_one_broadcast(const thread_block_tile<Size, Parent>& group, Fn&& fn, Args&&... args)
        -> typename _CG_STL_NAMESPACE::remove_reference<
            decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...))>::type {

    using ResultType = decltype(_CG_STL_NAMESPACE::forward<Fn>(fn)(_CG_STL_NAMESPACE::forward<Args>(args)...));
    static_assert(!_CG_STL_NAMESPACE::is_same<ResultType, void>::value,
                  "For invocables returning void invoke_one should be used instead");
    using impl = details::invoke_one_impl<details::elect_group_supported<thread_block_tile<Size, Parent>>::value>;
    return impl::invoke_one_broadcast(group,
                                      _CG_STL_NAMESPACE::forward<Fn>(fn),
                                      _CG_STL_NAMESPACE::forward<Args>(args)...);
}

_CG_END_NAMESPACE

#endif //_CG_CPP11_FEATURES

#endif // _CG_INVOKE_H
