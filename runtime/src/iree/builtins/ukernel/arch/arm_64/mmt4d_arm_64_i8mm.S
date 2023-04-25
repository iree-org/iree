#include "iree/builtins/ukernel/arch/arm_64/assembly.h"
#include "iree/builtins/ukernel/exported_bits.h"

// Parameters:
//   x0: iree_uk_int32_t* out_tile
//   x1: const iree_uk_int8_t* lhs_panel
//   x2: const iree_uk_int8_t* rhs_panel
//   w3: iree_uk_int32_t K. Note: K>=1, as the K==0 case was handled as an early-return.
//   w4: iree_uk_uint32_t flags
//   x5: (UNUSED) params - relevant params K and flags already passed above.

BEGIN_FUNCTION iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm

        // Save callee-saved NEON registers
        stp d8, d9, [sp, -64]!
        stp d10, d11, [sp, 16]
        stp d12, d13, [sp, 32]
        stp d14, d15, [sp, 48]

        // Do we accumulate into or clear the accumulator tile?
        tbnz w4, IREE_UK_FLAG_MMT4D_ACCUMULATE_BIT_POS, 1f
        
    0:
        // No-accumulate case. Clear the 8x8 accumulator tile.
        movi v16.16b, 0
        movi v17.16b, 0
        movi v18.16b, 0
        movi v19.16b, 0
        movi v20.16b, 0
        movi v21.16b, 0
        movi v22.16b, 0
        movi v23.16b, 0
        movi v24.16b, 0
        movi v25.16b, 0
        movi v26.16b, 0
        movi v27.16b, 0
        movi v28.16b, 0
        movi v29.16b, 0
        movi v30.16b, 0
        movi v31.16b, 0
        b 2f
  
    1:
        // Accumulate case. Load the 8x8 accumulator tile from row-major
        // out_tile and swizzle it into 2x2 tiled layout.
        //
        // Load rows 0--3.
        ldp q0, q1, [x0, 0]
        ldp q2, q3, [x0, 32]
        ldp q4, q5, [x0, 64]
        ldp q6, q7, [x0, 96]
        // Load rows 4--7.
        ldp q8, q9, [x0, 128]
        ldp q10, q11, [x0, 160]
        ldp q12, q13, [x0, 192]
        ldp q14, q15, [x0, 224]
        // Swizzle in 2x2 tiles for smmla, rows 0--1.
        zip1 v16.2d, v0.2d, v2.2d
        zip2 v17.2d, v0.2d, v2.2d
        zip1 v18.2d, v1.2d, v3.2d
        zip2 v19.2d, v1.2d, v3.2d
        // Swizzle in 2x2 tiles for smmla, rows 2--3.
        zip1 v20.2d, v4.2d, v6.2d
        zip2 v21.2d, v4.2d, v6.2d
        zip1 v22.2d, v5.2d, v7.2d
        zip2 v23.2d, v5.2d, v7.2d
        // Swizzle in 2x2 tiles for smmla, rows 4--5.
        zip1 v24.2d, v8.2d, v10.2d
        zip2 v25.2d, v8.2d, v10.2d
        zip1 v26.2d, v9.2d, v11.2d
        zip2 v27.2d, v9.2d, v11.2d
        // Swizzle in 2x2 tiles for smmla, rows 6--7.
        zip1 v28.2d, v12.2d, v14.2d
        zip2 v29.2d, v12.2d, v14.2d
        zip1 v30.2d, v13.2d, v15.2d
        zip2 v31.2d, v13.2d, v15.2d
    
    2:

        // Start of math work. If K==1, jump over the whole main loop.
        subs w3, w3, 1
        b.eq 6f

    3:
        // Prologue of main loop, 2x partially unrolled, for when K>=2.
        //
        // Decrement the loop counter K.
        subs w3, w3, 2
        // Pre-load data for first loop iteration
        //
        // Load 8x8 LHS tile
        ldp q0, q1, [x1], 32
        ldp q2, q3, [x1], 32
        // Load 8x8 RHS tile
        ldp q4, q5, [x2], 32
        ldp q6, q7, [x2], 32
        // Load 8x8 LHS tile
        ldp q8, q9, [x1], 32
        ldp q10, q11, [x1], 32
        // Load 8x8 RHS tile...
        ldp q12, q13, [x2], 32
        // ...second half loads is kept inside the loop below.
        //
        // Multiply-accumulate, rows 0--1.
        smmla v16.4s, v0.16b, v4.16b
        smmla v17.4s, v0.16b, v5.16b
        smmla v18.4s, v0.16b, v6.16b
        smmla v19.4s, v0.16b, v7.16b

        // If K==2, jump to the epilogue.
        b.le 5f

    4:
        // Body of main loop, 2x partially unrolled, for when K>2.
        //
        // Multiply-accumulate, rows 2--3.
        smmla v20.4s, v1.16b, v4.16b
        smmla v21.4s, v1.16b, v5.16b
        smmla v22.4s, v1.16b, v6.16b
        smmla v23.4s, v1.16b, v7.16b
        ldp q14, q15, [x2], 32
        // Multiply-accumulate, rows 4--5.
        smmla v24.4s, v2.16b, v4.16b
        smmla v25.4s, v2.16b, v5.16b
        smmla v26.4s, v2.16b, v6.16b
        smmla v27.4s, v2.16b, v7.16b
        ldp q0, q1, [x1], 32
        // Multiply-accumulate, rows 6--7.
        smmla v28.4s, v3.16b, v4.16b
        smmla v29.4s, v3.16b, v5.16b
        smmla v30.4s, v3.16b, v6.16b
        smmla v31.4s, v3.16b, v7.16b
        ldp q2, q3, [x1], 32
        // Multiply-accumulate, rows 0--1.
        smmla v16.4s, v8.16b, v12.16b
        smmla v17.4s, v8.16b, v13.16b
        smmla v18.4s, v8.16b, v14.16b
        smmla v19.4s, v8.16b, v15.16b
        ldp q4, q5, [x2], 32
        // Multiply-accumulate, rows 2--3.
        smmla v20.4s, v9.16b, v12.16b
        smmla v21.4s, v9.16b, v13.16b
        smmla v22.4s, v9.16b, v14.16b
        smmla v23.4s, v9.16b, v15.16b
        ldp q6, q7, [x2], 32
        // Multiply-accumulate, rows 4--5.
        smmla v24.4s, v10.16b, v12.16b
        smmla v25.4s, v10.16b, v13.16b
        smmla v26.4s, v10.16b, v14.16b
        smmla v27.4s, v10.16b, v15.16b
        ldp q8, q9, [x1], 32
        // Multiply-accumulate, rows 6--7.
        smmla v28.4s, v11.16b, v12.16b
        smmla v29.4s, v11.16b, v13.16b
        smmla v30.4s, v11.16b, v14.16b
        smmla v31.4s, v11.16b, v15.16b
        ldp q10, q11, [x1], 32
        // Multiply-accumulate, rows 0--1.
        smmla v16.4s, v0.16b, v4.16b
        smmla v17.4s, v0.16b, v5.16b
        ldp q12, q13, [x2], 32
        smmla v18.4s, v0.16b, v6.16b
        subs w3, w3, 2
        smmla v19.4s, v0.16b, v7.16b
        b.gt 4b

    5:
        // Epilogue of main loop, 2x partially unrolled, for when K>2.
        //
        // Load last chunk of last RHS tile.
        ldp q14, q15, [x2], 32

        // Multiply-accumulate, rows 2--3.
        smmla v20.4s, v1.16b, v4.16b
        smmla v21.4s, v1.16b, v5.16b
        smmla v22.4s, v1.16b, v6.16b
        smmla v23.4s, v1.16b, v7.16b
        // Multiply-accumulate, rows 4--5.
        smmla v24.4s, v2.16b, v4.16b
        smmla v25.4s, v2.16b, v5.16b
        smmla v26.4s, v2.16b, v6.16b
        smmla v27.4s, v2.16b, v7.16b
        // Multiply-accumulate, rows 6--7.
        smmla v28.4s, v3.16b, v4.16b
        smmla v29.4s, v3.16b, v5.16b
        smmla v30.4s, v3.16b, v6.16b
        smmla v31.4s, v3.16b, v7.16b

        // Multiply-accumulate, rows 0--1.
        smmla v16.4s, v8.16b, v12.16b
        smmla v17.4s, v8.16b, v13.16b
        smmla v18.4s, v8.16b, v14.16b
        smmla v19.4s, v8.16b, v15.16b
        // Multiply-accumulate, rows 2--3.
        smmla v20.4s, v9.16b, v12.16b
        smmla v21.4s, v9.16b, v13.16b
        smmla v22.4s, v9.16b, v14.16b
        smmla v23.4s, v9.16b, v15.16b
        // Multiply-accumulate, rows 4--5.
        smmla v24.4s, v10.16b, v12.16b
        smmla v25.4s, v10.16b, v13.16b
        smmla v26.4s, v10.16b, v14.16b
        smmla v27.4s, v10.16b, v15.16b
        // Multiply-accumulate, rows 6--7.
        smmla v28.4s, v11.16b, v12.16b
        smmla v29.4s, v11.16b, v13.16b
        smmla v30.4s, v11.16b, v14.16b
        smmla v31.4s, v11.16b, v15.16b

        // Finished accumulating? Then jump to final store.
        b.lt 7f
        // Fall through.

    6:
        // Accumulate for a single K-value - used for either the K==1 case or
        // final value of K for odd K>1.

        // Load 8x8 LHS tile
        ldp q0, q1, [x1, 0]
        ldp q2, q3, [x1, 32]
        add x1, x1, 64
        // Load 8x8 RHS tile
        ldp q4, q5, [x2, 0]
        ldp q6, q7, [x2, 32]
        add x2, x2, 64
        // Multiply-accumulate, rows 0--1.
        smmla v16.4s, v0.16b, v4.16b
        smmla v17.4s, v0.16b, v5.16b
        smmla v18.4s, v0.16b, v6.16b
        smmla v19.4s, v0.16b, v7.16b
        // Multiply-accumulate, rows 2--3.
        smmla v20.4s, v1.16b, v4.16b
        smmla v21.4s, v1.16b, v5.16b
        smmla v22.4s, v1.16b, v6.16b
        smmla v23.4s, v1.16b, v7.16b
        // Multiply-accumulate, rows 4--5.
        smmla v24.4s, v2.16b, v4.16b
        smmla v25.4s, v2.16b, v5.16b
        smmla v26.4s, v2.16b, v6.16b
        smmla v27.4s, v2.16b, v7.16b
        // Multiply-accumulate, rows 6--7.
        smmla v28.4s, v3.16b, v4.16b
        smmla v29.4s, v3.16b, v5.16b
        smmla v30.4s, v3.16b, v6.16b
        smmla v31.4s, v3.16b, v7.16b

    7:
        // Done accumulating.
        //
        // Swizzle back to row-major and store to destination.
        //
        // Swizzle back to row-major, rows 0--1.
        uzp1 v0.2d, v16.2d, v17.2d
        uzp1 v1.2d, v18.2d, v19.2d
        uzp2 v2.2d, v16.2d, v17.2d
        uzp2 v3.2d, v18.2d, v19.2d
        // Swizzle back to row-major, rows 2--3.
        uzp1 v4.2d, v20.2d, v21.2d
        uzp1 v5.2d, v22.2d, v23.2d
        uzp2 v6.2d, v20.2d, v21.2d
        uzp2 v7.2d, v22.2d, v23.2d
        // Swizzle back to row-major, rows 4--5.
        uzp1 v8.2d, v24.2d, v25.2d
        uzp1 v9.2d, v26.2d, v27.2d
        uzp2 v10.2d, v24.2d, v25.2d
        uzp2 v11.2d, v26.2d, v27.2d
        // Swizzle back to row-major, rows 6--7.
        uzp1 v12.2d, v28.2d, v29.2d
        uzp1 v13.2d, v30.2d, v31.2d
        uzp2 v14.2d, v28.2d, v29.2d
        uzp2 v15.2d, v30.2d, v31.2d
        // Store rows 0--3 to destination.
        stp q0, q1, [x0, 0]
        stp q2, q3, [x0, 32]
        stp q4, q5, [x0, 64]
        stp q6, q7, [x0, 96]
        stp q8, q9, [x0, 128]
        stp q10, q11, [x0, 160]
        stp q12, q13, [x0, 192]
        stp q14, q15, [x0, 224]

        // Restore callee-saved NEON registers
        ldp d14, d15, [sp, 48]
        ldp d12, d13, [sp, 32]
        ldp d10, d11, [sp, 16]
        ldp d8, d9, [sp], 64
        ret

END_FUNCTION iree_uk_mmt4d_tile_i8i8i32_8x8x8_arm_64_i8mm

ALLOW_NON_EXECUTABLE_STACK
