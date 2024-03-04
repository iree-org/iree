/*
 * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
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

#ifndef _CG_PARTITIONING_H
#define _CG_PARTITIONING_H

#include "info.h"
#include "helpers.h"

_CG_BEGIN_NAMESPACE

namespace details {

    template <typename TyGroup>
    _CG_STATIC_QUALIFIER coalesced_group _binary_partition(const TyGroup &tile, bool pred) {
        const unsigned int fullMask = ~0u;

        unsigned int thisMask = _coalesced_group_data_access::get_mask(tile);
        unsigned int predMask = pred ? 0 : fullMask;
        unsigned int setMask = __ballot_sync(thisMask, pred);

        if (setMask == thisMask || setMask == 0) {
            coalesced_group subTile = _coalesced_group_data_access::construct_from_mask<coalesced_group>(thisMask);
            _coalesced_group_data_access::modify_meta_group(subTile, 0, 1);
            return subTile;
        }
        else {
            unsigned int subMask = thisMask & (setMask ^ predMask);
            coalesced_group subTile = _coalesced_group_data_access::construct_from_mask<coalesced_group>(subMask);
            _coalesced_group_data_access::modify_meta_group(subTile, pred, 2);
            return subTile;
        }
    }

#ifdef _CG_HAS_MATCH_COLLECTIVE
    template <typename TyGroup, typename TyPredicate>
    _CG_STATIC_QUALIFIER coalesced_group _labeled_partition(const TyGroup &tile, TyPredicate pred) {
        unsigned int thisMask = _coalesced_group_data_access::get_mask(tile);
        unsigned int thisBias = __ffs(thisMask) - 1; // Subtract 1 to index properly from [1-32]
        unsigned int subMask = __match_any_sync(thisMask, pred);

        coalesced_group subTile = _coalesced_group_data_access::construct_from_mask<coalesced_group>(subMask);

        int leaderLaneId = subTile.shfl(details::laneid(), 0);

        bool isLeader = !subTile.thread_rank();
        unsigned int leaderMask = __ballot_sync(thisMask, isLeader);
        unsigned int tileRank = __fns(leaderMask, leaderLaneId, 0) - thisBias;

        _coalesced_group_data_access::modify_meta_group(subTile, tileRank, __popc(leaderMask));

        return subTile;
    }
#endif
}; // namespace details

_CG_STATIC_QUALIFIER coalesced_group binary_partition(const coalesced_group &tile, bool pred) {
    return details::_binary_partition(tile, pred);
}

template <unsigned int Size, typename ParentT>
_CG_STATIC_QUALIFIER coalesced_group binary_partition(const thread_block_tile<Size, ParentT> &tile, bool pred) {
#ifdef _CG_CPP11_FEATURES
    static_assert(Size <= 32, "Binary partition is available only for tiles of size smaller or equal to 32");
#endif
    return details::_binary_partition(tile, pred);
}


#if defined(_CG_HAS_MATCH_COLLECTIVE) && defined(_CG_CPP11_FEATURES)
template <typename TyPredicate>
_CG_STATIC_QUALIFIER coalesced_group labeled_partition(const coalesced_group &tile, TyPredicate pred) {
    static_assert(_CG_STL_NAMESPACE::is_integral<TyPredicate>::value, "labeled_partition predicate must be an integral type");
    return details::_labeled_partition(tile, pred);
}

template <typename TyPredicate, unsigned int Size, typename ParentT>
_CG_STATIC_QUALIFIER coalesced_group labeled_partition(const thread_block_tile<Size, ParentT> &tile, TyPredicate pred) {
    static_assert(_CG_STL_NAMESPACE::is_integral<TyPredicate>::value, "labeled_partition predicate must be an integral type");
    static_assert(Size <= 32, "Labeled partition is available only for tiles of size smaller or equal to 32");
    return details::_labeled_partition(tile, pred);
}
#endif

_CG_END_NAMESPACE

#endif // _CG_PARTITIONING_H
