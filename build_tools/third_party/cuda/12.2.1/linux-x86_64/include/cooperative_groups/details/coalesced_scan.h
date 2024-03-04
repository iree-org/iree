/* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
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
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
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

#ifndef _CG_COALESCED_SCAN_H_
#define _CG_COALESCED_SCAN_H_

#include "info.h"
#include "helpers.h"
#include "cooperative_groups.h"
#include "partitioning.h"
#include "functional.h"

_CG_BEGIN_NAMESPACE

namespace details {

template <typename TyGroup, typename TyVal, typename TyOp>
_CG_QUALIFIER auto inclusive_scan_contiguous(const TyGroup& group, TyVal&& val, TyOp&& op) -> decltype(op(val, val)) {
    auto out = val;
    for (int mask = 1; mask < group.size(); mask <<= 1) {
        auto tmp = group.shfl_up(out, mask);
        if (mask <= group.thread_rank()) {
            out = op(out, tmp);
        }
    }

    return out;
}

template <typename TyGroup, typename TyVal, typename TyOp>
_CG_QUALIFIER auto inclusive_scan_non_contiguous(const TyGroup& group, TyVal&& val, TyOp&& op) -> decltype(op(val, val)) {
    const unsigned int groupSize = group.size();
    auto out = val;

    const unsigned int mask = details::_coalesced_group_data_access::get_mask(group);
    unsigned int lanemask = details::lanemask32_lt() & mask;
    unsigned int srcLane = details::laneid();

    const unsigned int base = __ffs(mask)-1; /* lane with rank == 0 */
    const unsigned int rank = __popc(lanemask);

    for (unsigned int i = 1, j = 1; i < groupSize; i <<= 1) {
        if (i <= rank) {
            srcLane -= j;
            j = i; /* maximum possible lane */

            unsigned int begLane = base + rank - i; /* minimum possible lane */

            /*  Next source lane is in the range [ begLane .. srcLane ]
                *  If begLane < srcLane then do a binary search.
                */
            while (begLane < srcLane) {
                const unsigned int halfLane = (begLane + srcLane) >> 1;
                const unsigned int halfMask = lanemask >> halfLane;
                const unsigned int d = __popc(halfMask);
                if (d < i) {
                    srcLane = halfLane - 1; /* halfLane too large */
                }
                else if ((i < d) || !(halfMask & 0x01)) {
                    begLane = halfLane + 1; /* halfLane too small */
                }
                else {
                    begLane = srcLane = halfLane; /* happen to hit */
                }
            }
        }

        auto tmp = details::tile::shuffle_dispatch<TyVal>::shfl(out, mask, srcLane, 32);
        if (i <= rank) {
            out = op(out, tmp);
        }
    }
    return out;
}

template <unsigned int TySize, typename ParentT, typename TyVal, typename TyOp>
_CG_QUALIFIER auto coalesced_inclusive_scan(const __single_warp_thread_block_tile<TySize, ParentT>& group,
                                            TyVal&& val,
                                            TyOp&& op) -> decltype(op(val, val)) {
    return inclusive_scan_contiguous(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp>(op));
}

template <typename TyVal, typename TyOp>
_CG_QUALIFIER auto coalesced_inclusive_scan(const coalesced_group& group, TyVal&& val, TyOp&& op) -> decltype(op(val, val)) {
    if (group.size() == 32) {
        return inclusive_scan_contiguous(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp>(op));
    }
    else {
        return inclusive_scan_non_contiguous(group, _CG_STL_NAMESPACE::forward<TyVal>(val), _CG_STL_NAMESPACE::forward<TyOp>(op));
    }
}

template <bool IntegralOptimized>
struct scan_choose_convertion;

template<>
struct scan_choose_convertion<true> {
    template <typename TyGroup, typename TyRes, typename TyVal>
    _CG_STATIC_QUALIFIER details::remove_qual<TyVal> convert_inclusive_to_exclusive(const TyGroup& group, TyRes& result, TyVal&& val) {
        return result - val;
    }
};

template<>
struct scan_choose_convertion<false> {
    template <typename TyGroup, typename TyRes, typename TyVal>
    _CG_STATIC_QUALIFIER details::remove_qual<TyVal> convert_inclusive_to_exclusive(const TyGroup& group, TyRes& result, TyVal&& val) {
        auto ret = group.shfl_up(result, 1);
        if (group.thread_rank() == 0) {
            return {};
        }
        else {
            return ret;
        }
    }
};

template <typename TyGroup, typename TyRes, typename TyVal, typename TyFn>
_CG_QUALIFIER auto convert_inclusive_to_exclusive(const TyGroup& group, TyRes& result, TyVal&& val, TyFn&& op) -> decltype(op(val, val)) {
    using conversion = scan_choose_convertion<_CG_STL_NAMESPACE::is_same<remove_qual<TyFn>, cooperative_groups::plus<remove_qual<TyVal>>>::value
                                 && _CG_STL_NAMESPACE::is_integral<remove_qual<TyVal>>::value>;
    return conversion::convert_inclusive_to_exclusive(group, result, _CG_STL_NAMESPACE::forward<TyVal>(val));
}

} // details

_CG_END_NAMESPACE

#endif // _CG_COALESCED_SCAN_H_