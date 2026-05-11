; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1201 -mattr='+wavefrontsize32' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 5
	.text
	.globl	matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32
	.p2align	8
	.type	matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32,@function
matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32:
	v_lshlrev_b32_e32 v4, 3, v0
	s_load_b128 s[12:15], s[0:1], 0x0
	v_or_b32_e32 v1, 0x100, v0
	v_lshlrev_b32_e32 v5, 4, v0
	s_lshl_b32 s3, ttmp9, 7
	v_and_b32_e32 v183, 0x78, v4
	v_or_b32_e32 v2, 0x200, v0
	v_lshrrev_b32_e32 v180, 3, v1
	s_lshl_b32 s2, ttmp9, 4
	v_or_b32_e32 v3, 0x300, v0
	s_and_b32 s3, s3, 0x780
	s_and_b32 s2, s2, 0x700
	v_and_b32_e32 v184, 0x70, v5
	v_or_b32_e32 v5, s3, v183
	v_lshrrev_b32_e32 v179, 3, v0
	v_lshrrev_b32_e32 v181, 3, v2
	v_lshrrev_b32_e32 v192, 4, v1
	v_or_b32_e32 v1, s2, v180
	v_lshrrev_b32_e32 v182, 3, v3
	v_lshrrev_b32_e32 v191, 4, v0
	v_lshlrev_b32_e32 v197, 1, v5
	v_or_b32_e32 v4, s2, v179
	v_lshl_or_b32 v196, v1, 12, v184
	v_or_b32_e32 v1, s2, v181
	v_lshrrev_b32_e32 v194, 4, v2
	v_or_b32_e32 v2, s2, v182
	v_lshrrev_b32_e32 v195, 4, v3
	v_lshl_or_b32 v3, v191, 12, v197
	s_mov_b32 s7, 0x31027000
	s_mov_b32 s6, 0x800000
	s_wait_kmcnt 0x0
	s_and_b32 s15, s15, 0xffff
	v_lshl_or_b32 v193, v4, 12, v184
	v_lshl_or_b32 v4, v192, 12, v197
	s_and_b32 s13, s13, 0xffff
	v_lshl_or_b32 v198, v1, 12, v184
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_mov_b32 s8, s14
	s_mov_b32 s9, s15
	v_lshl_or_b32 v199, v2, 12, v184
	s_mov_b32 s4, s12
	s_mov_b32 s5, s13
	v_lshl_or_b32 v1, v194, 12, v197
	s_clause 0x1
	buffer_load_b128 v[137:140], v3, s[8:11], null offen
	buffer_load_b128 v[141:144], v4, s[8:11], null offen
	s_clause 0x1
	buffer_load_b128 v[145:148], v198, s[4:7], null offen
	buffer_load_b128 v[149:152], v199, s[4:7], null offen
	buffer_load_b128 v[153:156], v1, s[8:11], null offen
	v_lshl_or_b32 v1, v195, 12, v197
	s_clause 0x5
	buffer_load_b128 v[129:132], v193, s[4:7], null offen
	buffer_load_b128 v[133:136], v196, s[4:7], null offen
	buffer_load_b128 v[157:160], v193, s[4:7], null offen offset:524288
	buffer_load_b128 v[161:164], v193, s[4:7], null offen offset:655360
	buffer_load_b128 v[165:168], v193, s[4:7], null offen offset:786432
	buffer_load_b128 v[169:172], v193, s[4:7], null offen offset:917504
	buffer_load_b128 v[173:176], v1, s[8:11], null offen
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v178, 15, v0
	v_lshrrev_b32_e32 v177, 1, v0
	v_lshlrev_b32_e32 v187, 1, v0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v8, v1 :: v_dual_and_b32 v185, 0xcf, v0
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v10, v1
	v_and_b32_e32 v177, 8, v177
	v_and_or_b32 v178, v187, 64, v178
	v_or_b32_e32 v186, 48, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_mul_u32_u24_e32 v185, 0x44, v185
	v_mad_u32_u24 v202, 0x88, v179, v184
	v_mad_u32_u24 v203, 0x88, v180, v184
	v_mad_u32_u24 v204, 0x88, v181, v184
	v_mad_u32_u24 v205, 0x88, v182, v184
	v_lshlrev_b32_e32 v179, 1, v183
	v_mul_u32_u24_e32 v181, 0x108, v194
	v_lshlrev_b32_e32 v183, 1, v177
	v_mul_u32_u24_e32 v184, 0x84, v177
	v_lshlrev_b32_e32 v187, 1, v178
	v_mul_u32_u24_e32 v188, 0x108, v191
	v_mul_u32_u24_e32 v186, 0x44, v186
	v_mul_u32_u24_e32 v180, 0x108, v192
	v_mul_u32_u24_e32 v182, 0x108, v195
	v_add3_u32 v212, v181, v179, 0x8800
	v_lshl_add_u32 v181, v184, 1, v187
	v_lshl_add_u32 v185, v185, 1, v183
	v_mov_b32_e32 v2, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_dual_mov_b32 v12, v1 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v14, v1 :: v_dual_mov_b32 v11, v1
	v_dual_mov_b32 v16, v1 :: v_dual_mov_b32 v13, v1
	v_dual_mov_b32 v18, v1 :: v_dual_mov_b32 v15, v1
	v_dual_mov_b32 v20, v1 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v22, v1 :: v_dual_mov_b32 v19, v1
	v_dual_mov_b32 v24, v1 :: v_dual_mov_b32 v21, v1
	v_dual_mov_b32 v26, v1 :: v_dual_mov_b32 v23, v1
	v_dual_mov_b32 v28, v1 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v30, v1 :: v_dual_mov_b32 v27, v1
	v_dual_mov_b32 v32, v1 :: v_dual_mov_b32 v29, v1
	v_dual_mov_b32 v34, v1 :: v_dual_mov_b32 v31, v1
	v_dual_mov_b32 v36, v1 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v38, v1 :: v_dual_mov_b32 v35, v1
	v_dual_mov_b32 v40, v1 :: v_dual_mov_b32 v37, v1
	v_dual_mov_b32 v42, v1 :: v_dual_mov_b32 v39, v1
	v_dual_mov_b32 v44, v1 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v46, v1 :: v_dual_mov_b32 v43, v1
	v_dual_mov_b32 v48, v1 :: v_dual_mov_b32 v45, v1
	v_dual_mov_b32 v50, v1 :: v_dual_mov_b32 v47, v1
	v_dual_mov_b32 v52, v1 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v54, v1 :: v_dual_mov_b32 v51, v1
	v_dual_mov_b32 v56, v1 :: v_dual_mov_b32 v53, v1
	v_dual_mov_b32 v58, v1 :: v_dual_mov_b32 v55, v1
	v_dual_mov_b32 v60, v1 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v62, v1 :: v_dual_mov_b32 v59, v1
	v_dual_mov_b32 v64, v1 :: v_dual_mov_b32 v61, v1
	v_dual_mov_b32 v66, v1 :: v_dual_mov_b32 v63, v1
	v_dual_mov_b32 v68, v1 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v70, v1 :: v_dual_mov_b32 v67, v1
	v_dual_mov_b32 v72, v1 :: v_dual_mov_b32 v69, v1
	v_dual_mov_b32 v74, v1 :: v_dual_mov_b32 v71, v1
	v_dual_mov_b32 v76, v1 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v78, v1 :: v_dual_mov_b32 v75, v1
	v_dual_mov_b32 v80, v1 :: v_dual_mov_b32 v77, v1
	v_dual_mov_b32 v82, v1 :: v_dual_mov_b32 v79, v1
	v_dual_mov_b32 v84, v1 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v86, v1 :: v_dual_mov_b32 v83, v1
	v_dual_mov_b32 v88, v1 :: v_dual_mov_b32 v85, v1
	v_dual_mov_b32 v90, v1 :: v_dual_mov_b32 v87, v1
	v_dual_mov_b32 v92, v1 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v94, v1 :: v_dual_mov_b32 v91, v1
	v_dual_mov_b32 v96, v1 :: v_dual_mov_b32 v93, v1
	v_dual_mov_b32 v98, v1 :: v_dual_mov_b32 v95, v1
	v_dual_mov_b32 v100, v1 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v102, v1 :: v_dual_mov_b32 v99, v1
	v_dual_mov_b32 v104, v1 :: v_dual_mov_b32 v101, v1
	v_dual_mov_b32 v106, v1 :: v_dual_mov_b32 v103, v1
	v_dual_mov_b32 v108, v1 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v110, v1 :: v_dual_mov_b32 v107, v1
	v_dual_mov_b32 v112, v1 :: v_dual_mov_b32 v109, v1
	v_dual_mov_b32 v114, v1 :: v_dual_mov_b32 v111, v1
	v_dual_mov_b32 v116, v1 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v118, v1 :: v_dual_mov_b32 v115, v1
	v_dual_mov_b32 v120, v1 :: v_dual_mov_b32 v117, v1
	v_dual_mov_b32 v122, v1 :: v_dual_mov_b32 v119, v1
	v_dual_mov_b32 v124, v1 :: v_dual_mov_b32 v121, v1
	v_dual_mov_b32 v126, v1 :: v_dual_mov_b32 v123, v1
	v_dual_mov_b32 v128, v1 :: v_dual_mov_b32 v125, v1
	v_dual_mov_b32 v127, v1 :: v_dual_add_nc_u32 v206, 0x4400, v202
	v_add_nc_u32_e32 v208, 0x6600, v202
	v_add_nc_u32_e32 v207, 0x5500, v202
	v_add_nc_u32_e32 v209, 0x7700, v202
	v_add3_u32 v210, v188, v179, 0x8800
	v_add3_u32 v211, v180, v179, 0x8800
	v_add3_u32 v213, v182, v179, 0x8800
	v_lshl_add_u32 v180, v186, 1, v183
	v_add_nc_u32_e32 v179, 0x8800, v181
	v_add_nc_u32_e32 v190, 0x9880, v181
	v_add_nc_u32_e32 v201, 0xa900, v181
	v_add_nc_u32_e32 v200, 0xb980, v181
	v_add_nc_u32_e32 v188, 0x880, v185
	v_add_nc_u32_e32 v187, 0x8a0, v185
	v_add_nc_u32_e32 v183, 0x8c0, v185
	v_add_nc_u32_e32 v184, 0x8e0, v185
	v_add_nc_u32_e32 v189, 0x1100, v185
	v_add_nc_u32_e32 v186, 0x1120, v185
	v_add_nc_u32_e32 v181, 0x1140, v185
	v_add_nc_u32_e32 v182, 0x1160, v185
	s_mov_b32 s16, 0
	s_wait_loadcnt 0x6
	ds_store_2addr_b64 v202, v[129:130], v[131:132] offset1:1
	ds_store_2addr_b64 v210, v[137:138], v[139:140] offset1:1
	s_wait_loadcnt 0x5
	ds_store_2addr_b64 v203, v[133:134], v[135:136] offset1:1
	ds_store_2addr_b64 v211, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v204, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v212, v[153:154], v[155:156] offset1:1
	ds_store_2addr_b64 v205, v[149:150], v[151:152] offset1:1
	s_wait_loadcnt 0x4
	ds_store_2addr_b64 v206, v[157:158], v[159:160] offset1:1
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v207, v[161:162], v[163:164] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v208, v[165:166], v[167:168] offset1:1
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v209, v[169:170], v[171:172] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v213, v[173:174], v[175:176] offset1:1
.LBB0_1:
	s_add_co_i32 s12, s16, 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s13, s12, 4
	s_lshl_b32 s14, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v129, s13, v191
	v_or_b32_e32 v130, s13, v192
	v_or_b32_e32 v131, s13, v194
	v_or_b32_e32 v132, s13, v195
	v_add_nc_u32_e32 v133, s14, v198
	v_add_nc_u32_e32 v134, s14, v199
	v_add_nc_u32_e32 v141, s14, v196
	v_add_nc_u32_e32 v153, s14, v193
	v_lshl_or_b32 v129, v129, 12, v197
	v_lshl_or_b32 v130, v130, 12, v197
	v_lshl_or_b32 v131, v131, 12, v197
	v_lshl_or_b32 v132, v132, 12, v197
	s_clause 0x7
	buffer_load_b128 v[137:140], v133, s[4:7], null offen
	buffer_load_b128 v[133:136], v134, s[4:7], null offen
	buffer_load_b128 v[157:160], v141, s[4:7], null offen
	buffer_load_b128 v[173:176], v153, s[4:7], null offen
	buffer_load_b128 v[141:144], v153, s[4:7], null offen offset:524288
	buffer_load_b128 v[145:148], v153, s[4:7], null offen offset:655360
	buffer_load_b128 v[149:152], v153, s[4:7], null offen offset:786432
	buffer_load_b128 v[153:156], v153, s[4:7], null offen offset:917504
	s_clause 0x3
	buffer_load_b128 v[169:172], v129, s[8:11], null offen
	buffer_load_b128 v[165:168], v130, s[8:11], null offen
	buffer_load_b128 v[161:164], v131, s[8:11], null offen
	buffer_load_b128 v[129:132], v132, s[8:11], null offen
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_u16_d16 v214, v179
	ds_load_u16_d16 v215, v179 offset:528
	ds_load_u16_d16 v216, v179 offset:1056
	ds_load_u16_d16 v217, v179 offset:1584
	ds_load_2addr_b64 v[222:225], v188 offset1:1
	ds_load_2addr_b64 v[230:233], v180 offset1:1
	ds_load_2addr_b64 v[226:229], v189 offset1:1
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v214, v179 offset:264
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v215, v179 offset:792
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v216, v179 offset:1320
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v217, v179 offset:1848
	ds_load_2addr_b64 v[218:221], v185 offset1:1
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[89:96], v[222:225], v[214:217], v[89:96]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[218:221], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[57:64], v[226:229], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	ds_load_u16_d16 v216, v179 offset:1120
	ds_load_u16_d16 v214, v179 offset:64
	ds_load_u16_d16 v215, v179 offset:592
	ds_load_u16_d16 v217, v179 offset:1648
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v216, v179 offset:1384
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v214, v179 offset:328
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v215, v179 offset:856
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v217, v179 offset:1912
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[105:112], v[218:221], v[214:217], v[105:112]
	v_wmma_f32_16x16x16_f16 v[73:80], v[222:225], v[214:217], v[73:80]
	v_wmma_f32_16x16x16_f16 v[41:48], v[226:229], v[214:217], v[41:48]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[214:217], v[9:16]
	ds_load_u16_d16 v214, v179 offset:32
	ds_load_u16_d16 v215, v179 offset:560
	ds_load_u16_d16 v234, v179 offset:96
	ds_load_u16_d16 v235, v179 offset:624
	ds_load_u16_d16 v216, v179 offset:1088
	ds_load_u16_d16 v236, v179 offset:1152
	ds_load_u16_d16 v217, v179 offset:1616
	ds_load_u16_d16 v237, v179 offset:1680
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v214, v179 offset:296
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v234, v179 offset:360
	ds_load_u16_d16_hi v215, v179 offset:824
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v235, v179 offset:888
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v216, v179 offset:1352
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v236, v179 offset:1416
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v217, v179 offset:1880
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v237, v179 offset:1944
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[113:120], v[218:221], v[214:217], v[113:120]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[218:221], v[234:237], v[97:104]
	v_wmma_f32_16x16x16_f16 v[81:88], v[222:225], v[214:217], v[81:88]
	v_wmma_f32_16x16x16_f16 v[65:72], v[222:225], v[234:237], v[65:72]
	v_wmma_f32_16x16x16_f16 v[49:56], v[226:229], v[214:217], v[49:56]
	v_wmma_f32_16x16x16_f16 v[33:40], v[226:229], v[234:237], v[33:40]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[214:217], v[17:24]
	ds_load_u16_d16 v214, v190
	ds_load_u16_d16 v215, v179 offset:4752
	ds_load_u16_d16 v218, v190 offset:32
	ds_load_u16_d16 v219, v179 offset:4784
	ds_load_u16_d16 v222, v190 offset:64
	ds_load_u16_d16 v223, v179 offset:4816
	ds_load_u16_d16 v226, v190 offset:96
	ds_load_u16_d16 v227, v179 offset:4848
	ds_load_u16_d16 v216, v179 offset:5280
	ds_load_u16_d16 v220, v179 offset:5312
	ds_load_u16_d16 v224, v179 offset:5344
	ds_load_u16_d16 v228, v179 offset:5376
	ds_load_u16_d16 v217, v179 offset:5808
	ds_load_u16_d16 v221, v179 offset:5840
	ds_load_u16_d16 v225, v179 offset:5872
	ds_load_u16_d16 v229, v179 offset:5904
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[234:237], v[1:8]
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v214, v179 offset:4488
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v222, v179 offset:4552
	ds_load_u16_d16_hi v218, v179 offset:4520
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v226, v179 offset:4584
	ds_load_u16_d16_hi v215, v179 offset:5016
	ds_load_u16_d16_hi v219, v179 offset:5048
	ds_load_u16_d16_hi v223, v179 offset:5080
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:5112
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:5544
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:5576
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:5608
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:5640
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:6072
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:6104
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:6136
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:6168
	ds_load_2addr_b64 v[230:233], v185 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[218:221], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[222:225], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[226:229], v[97:104]
	ds_load_2addr_b64 v[230:233], v187 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[214:217], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[218:221], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[222:225], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[226:229], v[65:72]
	ds_load_2addr_b64 v[230:233], v186 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[218:221], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[222:225], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[226:229], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[218:221], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[222:225], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[226:229], v[1:8]
	ds_load_u16_d16 v214, v201
	ds_load_u16_d16 v215, v179 offset:8976
	ds_load_u16_d16 v218, v201 offset:32
	ds_load_u16_d16 v219, v179 offset:9008
	ds_load_u16_d16 v222, v201 offset:64
	ds_load_u16_d16 v223, v179 offset:9040
	ds_load_u16_d16 v226, v201 offset:96
	ds_load_u16_d16 v227, v179 offset:9072
	ds_load_u16_d16 v216, v179 offset:9504
	ds_load_u16_d16 v224, v179 offset:9568
	ds_load_u16_d16 v220, v179 offset:9536
	ds_load_u16_d16 v228, v179 offset:9600
	ds_load_u16_d16 v217, v179 offset:10032
	ds_load_u16_d16 v225, v179 offset:10096
	ds_load_u16_d16 v221, v179 offset:10064
	ds_load_u16_d16 v229, v179 offset:10128
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v214, v179 offset:8712
	s_wait_dscnt 0xe
	ds_load_u16_d16_hi v218, v179 offset:8744
	s_wait_dscnt 0xd
	ds_load_u16_d16_hi v222, v179 offset:8776
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v226, v179 offset:8808
	ds_load_u16_d16_hi v215, v179 offset:9240
	ds_load_u16_d16_hi v219, v179 offset:9272
	ds_load_u16_d16_hi v223, v179 offset:9304
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:9336
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:9768
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:9832
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:9800
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:9864
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:10296
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:10360
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:10328
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:10392
	ds_load_2addr_b64 v[230:233], v185 offset0:8 offset1:9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[218:221], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[222:225], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[226:229], v[97:104]
	ds_load_2addr_b64 v[230:233], v183 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[214:217], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[218:221], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[222:225], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[226:229], v[65:72]
	ds_load_2addr_b64 v[230:233], v181 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[218:221], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[222:225], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[226:229], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:8 offset1:9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[218:221], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[222:225], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[226:229], v[1:8]
	ds_load_u16_d16 v217, v179 offset:14352
	ds_load_u16_d16 v214, v200 offset:96
	ds_load_u16_d16 v218, v200
	ds_load_u16_d16 v219, v179 offset:13200
	ds_load_u16_d16 v222, v200 offset:32
	ds_load_u16_d16 v223, v179 offset:13232
	ds_load_u16_d16 v226, v200 offset:64
	ds_load_u16_d16 v227, v179 offset:13264
	ds_load_u16_d16 v215, v179 offset:13296
	ds_load_u16_d16 v220, v179 offset:13728
	ds_load_u16_d16 v224, v179 offset:13760
	ds_load_u16_d16 v228, v179 offset:13792
	ds_load_u16_d16 v216, v179 offset:13824
	ds_load_u16_d16 v221, v179 offset:14256
	ds_load_u16_d16 v225, v179 offset:14288
	ds_load_u16_d16 v229, v179 offset:14320
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:14616
	s_wait_dscnt 0xe
	ds_load_u16_d16_hi v218, v179 offset:12936
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v226, v179 offset:13000
	ds_load_u16_d16_hi v222, v179 offset:12968
	ds_load_u16_d16_hi v214, v179 offset:13032
	ds_load_u16_d16_hi v219, v179 offset:13464
	ds_load_u16_d16_hi v223, v179 offset:13496
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:13528
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v215, v179 offset:13560
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:13992
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:14024
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:14056
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:14088
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:14520
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:14552
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:14584
	ds_load_2addr_b64 v[230:233], v185 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[218:221], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[222:225], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[226:229], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[214:217], v[97:104]
	ds_load_2addr_b64 v[230:233], v184 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[218:221], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[222:225], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[226:229], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[214:217], v[65:72]
	ds_load_2addr_b64 v[230:233], v182 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[218:221], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[222:225], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[226:229], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[214:217], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[218:221], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[222:225], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[226:229], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[214:217], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_lt_u32 s16, 0x78
	s_mov_b32 s16, s12
	s_wait_loadcnt 0x8
	ds_store_2addr_b64 v202, v[173:174], v[175:176] offset1:1
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v210, v[169:170], v[171:172] offset1:1
	ds_store_2addr_b64 v203, v[157:158], v[159:160] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v211, v[165:166], v[167:168] offset1:1
	ds_store_2addr_b64 v204, v[137:138], v[139:140] offset1:1
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v212, v[161:162], v[163:164] offset1:1
	ds_store_2addr_b64 v205, v[133:134], v[135:136] offset1:1
	ds_store_2addr_b64 v206, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v207, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v208, v[149:150], v[151:152] offset1:1
	ds_store_2addr_b64 v209, v[153:154], v[155:156] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v213, v[129:130], v[131:132] offset1:1
	s_cbranch_scc1 .LBB0_1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v0, 0xc0, v0
	v_or_b32_e32 v178, s3, v178
	s_wait_kmcnt 0x0
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x31027000
	v_or3_b32 v0, s2, v0, v177
	v_lshlrev_b32_e32 v177, 2, v178
	s_mov_b32 s2, 0x1000000
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v0, v0, 13, v177
	s_barrier_wait -1
	ds_load_u16_d16 v137, v179
	ds_load_u16_d16 v129, v179 offset:32
	ds_load_u16_d16 v141, v179 offset:64
	ds_load_u16_d16 v133, v179 offset:96
	ds_load_u16_d16 v138, v179 offset:528
	ds_load_u16_d16 v142, v179 offset:592
	ds_load_u16_d16 v146, v179 offset:8976
	ds_load_u16_d16 v150, v179 offset:9008
	ds_load_u16_d16 v154, v179 offset:9040
	ds_load_u16_d16 v158, v179 offset:9072
	ds_load_u16_d16 v147, v179 offset:9504
	ds_load_u16_d16 v155, v179 offset:9568
	ds_load_u16_d16 v148, v179 offset:10032
	ds_load_u16_d16 v156, v179 offset:10096
	ds_load_u16_d16 v151, v179 offset:9536
	ds_load_u16_d16 v159, v179 offset:9600
	ds_load_u16_d16 v145, v201
	ds_load_u16_d16 v149, v201 offset:32
	ds_load_u16_d16 v153, v201 offset:64
	ds_load_u16_d16 v157, v201 offset:96
	ds_load_u16_d16 v161, v200
	ds_load_u16_d16 v165, v200 offset:32
	ds_load_u16_d16 v169, v200 offset:64
	ds_load_u16_d16 v173, v200 offset:96
	ds_load_u16_d16 v152, v179 offset:10064
	ds_load_u16_d16 v160, v179 offset:10128
	ds_load_u16_d16 v130, v179 offset:560
	ds_load_u16_d16 v134, v179 offset:624
	ds_load_u16_d16 v162, v179 offset:13200
	ds_load_u16_d16 v166, v179 offset:13232
	ds_load_u16_d16 v170, v179 offset:13264
	ds_load_u16_d16 v174, v179 offset:13296
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v146, v179 offset:9240
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v150, v179 offset:9272
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v154, v179 offset:9304
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v158, v179 offset:9336
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v147, v179 offset:9768
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v155, v179 offset:9832
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v151, v179 offset:9800
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v159, v179 offset:9864
	ds_load_u16_d16_hi v148, v179 offset:10296
	ds_load_u16_d16_hi v156, v179 offset:10360
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v161, v179 offset:12936
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v169, v179 offset:13000
	s_wait_dscnt 0x13
	ds_load_u16_d16_hi v152, v179 offset:10328
	s_wait_dscnt 0x13
	ds_load_u16_d16_hi v160, v179 offset:10392
	ds_load_u16_d16_hi v165, v179 offset:12968
	ds_load_u16_d16_hi v173, v179 offset:13032
	ds_load_u16_d16 v139, v179 offset:1056
	ds_load_u16_d16 v143, v179 offset:1120
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v162, v179 offset:13464
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v166, v179 offset:13496
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v170, v179 offset:13528
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v174, v179 offset:13560
	ds_load_u16_d16 v163, v179 offset:13728
	ds_load_u16_d16 v167, v179 offset:13760
	ds_load_u16_d16 v171, v179 offset:13792
	ds_load_u16_d16 v175, v179 offset:13824
	ds_load_u16_d16_hi v137, v179 offset:264
	ds_load_u16_d16_hi v129, v179 offset:296
	ds_load_u16_d16_hi v141, v179 offset:328
	ds_load_u16_d16_hi v133, v179 offset:360
	ds_load_u16_d16_hi v138, v179 offset:792
	ds_load_u16_d16_hi v142, v179 offset:856
	ds_load_u16_d16 v176, v179 offset:14352
	ds_load_u16_d16_hi v130, v179 offset:824
	ds_load_u16_d16_hi v134, v179 offset:888
	ds_load_u16_d16 v131, v179 offset:1088
	ds_load_u16_d16 v135, v179 offset:1152
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v139, v179 offset:1320
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v143, v179 offset:1384
	ds_load_u16_d16 v140, v179 offset:1584
	ds_load_u16_d16 v144, v179 offset:1648
	ds_load_u16_d16 v191, v190
	ds_load_u16_d16 v195, v190 offset:32
	ds_load_u16_d16 v199, v190 offset:64
	ds_load_u16_d16 v203, v190 offset:96
	ds_load_u16_d16 v132, v179 offset:1616
	ds_load_u16_d16 v136, v179 offset:1680
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v163, v179 offset:13992
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v167, v179 offset:14024
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v171, v179 offset:14056
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v175, v179 offset:14088
	ds_load_u16_d16 v164, v179 offset:14256
	ds_load_u16_d16 v168, v179 offset:14288
	ds_load_u16_d16 v172, v179 offset:14320
	ds_load_u16_d16 v192, v179 offset:4752
	ds_load_u16_d16 v196, v179 offset:4784
	ds_load_u16_d16 v200, v179 offset:4816
	ds_load_u16_d16 v204, v179 offset:4848
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v176, v179 offset:14616
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v131, v179 offset:1352
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v135, v179 offset:1416
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v140, v179 offset:1848
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v144, v179 offset:1912
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v191, v179 offset:4488
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v199, v179 offset:4552
	ds_load_u16_d16 v193, v179 offset:5280
	ds_load_u16_d16 v197, v179 offset:5312
	ds_load_u16_d16 v201, v179 offset:5344
	ds_load_u16_d16 v205, v179 offset:5376
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v132, v179 offset:1880
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v136, v179 offset:1944
	ds_load_u16_d16_hi v195, v179 offset:4520
	ds_load_u16_d16_hi v203, v179 offset:4584
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v164, v179 offset:14520
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v168, v179 offset:14552
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v172, v179 offset:14584
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v192, v179 offset:5016
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v196, v179 offset:5048
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v200, v179 offset:5080
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v204, v179 offset:5112
	ds_load_2addr_b64 v[207:210], v185 offset1:1
	ds_load_2addr_b64 v[211:214], v188 offset1:1
	ds_load_2addr_b64 v[215:218], v189 offset1:1
	ds_load_2addr_b64 v[219:222], v180 offset1:1
	ds_load_2addr_b64 v[223:226], v185 offset0:4 offset1:5
	ds_load_2addr_b64 v[227:230], v185 offset0:8 offset1:9
	ds_load_2addr_b64 v[231:234], v185 offset0:12 offset1:13
	ds_load_2addr_b64 v[187:190], v187 offset1:1
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v193, v179 offset:5544
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v197, v179 offset:5576
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v201, v179 offset:5608
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v205, v179 offset:5640
	ds_load_u16_d16 v194, v179 offset:5808
	ds_load_u16_d16 v198, v179 offset:5840
	ds_load_u16_d16 v202, v179 offset:5872
	ds_load_u16_d16 v206, v179 offset:5904
	ds_load_2addr_b64 v[235:238], v183 offset1:1
	ds_load_2addr_b64 v[239:242], v184 offset1:1
	ds_load_2addr_b64 v[183:186], v186 offset1:1
	ds_load_2addr_b64 v[243:246], v180 offset0:4 offset1:5
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x16_f16 v[121:128], v[207:210], v[137:140], v[121:128]
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x16_f16 v[89:96], v[211:214], v[137:140], v[89:96]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x16_f16 v[57:64], v[215:218], v[137:140], v[57:64]
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x16_f16 v[25:32], v[219:222], v[137:140], v[25:32]
	ds_load_2addr_b64 v[137:140], v181 offset1:1
	ds_load_2addr_b64 v[247:250], v182 offset1:1
	v_wmma_f32_16x16x16_f16 v[105:112], v[207:210], v[141:144], v[105:112]
	v_wmma_f32_16x16x16_f16 v[73:80], v[211:214], v[141:144], v[73:80]
	v_wmma_f32_16x16x16_f16 v[41:48], v[215:218], v[141:144], v[41:48]
	v_wmma_f32_16x16x16_f16 v[9:16], v[219:222], v[141:144], v[9:16]
	ds_load_2addr_b64 v[141:144], v180 offset0:8 offset1:9
	ds_load_2addr_b64 v[251:254], v180 offset0:12 offset1:13
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v194, v179 offset:6072
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v198, v179 offset:6104
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v202, v179 offset:6136
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v206, v179 offset:6168
	ds_load_u16_d16_hi v145, v179 offset:8712
	ds_load_u16_d16_hi v149, v179 offset:8744
	ds_load_u16_d16_hi v153, v179 offset:8776
	ds_load_u16_d16_hi v157, v179 offset:8808
	v_wmma_f32_16x16x16_f16 v[113:120], v[207:210], v[129:132], v[113:120]
	v_wmma_f32_16x16x16_f16 v[97:104], v[207:210], v[133:136], v[97:104]
	v_wmma_f32_16x16x16_f16 v[81:88], v[211:214], v[129:132], v[81:88]
	v_wmma_f32_16x16x16_f16 v[65:72], v[211:214], v[133:136], v[65:72]
	v_wmma_f32_16x16x16_f16 v[49:56], v[215:218], v[129:132], v[49:56]
	v_wmma_f32_16x16x16_f16 v[33:40], v[215:218], v[133:136], v[33:40]
	v_wmma_f32_16x16x16_f16 v[17:24], v[219:222], v[129:132], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[219:222], v[133:136], v[1:8]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_f16 v[121:128], v[223:226], v[191:194], v[121:128]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_f16 v[113:120], v[223:226], v[195:198], v[113:120]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_f16 v[105:112], v[223:226], v[199:202], v[105:112]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_f16 v[97:104], v[223:226], v[203:206], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[187:190], v[191:194], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[187:190], v[195:198], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[187:190], v[199:202], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[187:190], v[203:206], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[183:186], v[191:194], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[183:186], v[195:198], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[183:186], v[199:202], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[183:186], v[203:206], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[243:246], v[191:194], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[243:246], v[195:198], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[243:246], v[199:202], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[243:246], v[203:206], v[1:8]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[121:128], v[227:230], v[145:148], v[121:128]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_f16 v[113:120], v[227:230], v[149:152], v[113:120]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[105:112], v[227:230], v[153:156], v[105:112]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[227:230], v[157:160], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[235:238], v[145:148], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[235:238], v[149:152], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[235:238], v[153:156], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[235:238], v[157:160], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[137:140], v[145:148], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[137:140], v[149:152], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[137:140], v[153:156], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[137:140], v[157:160], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[141:144], v[145:148], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[141:144], v[149:152], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[141:144], v[153:156], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[141:144], v[157:160], v[1:8]
	v_wmma_f32_16x16x16_f16 v[121:128], v[231:234], v[161:164], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[231:234], v[165:168], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[231:234], v[169:172], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[231:234], v[173:176], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[239:242], v[161:164], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[239:242], v[165:168], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[239:242], v[169:172], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[239:242], v[173:176], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[247:250], v[161:164], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[247:250], v[165:168], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[247:250], v[169:172], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[247:250], v[173:176], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[251:254], v[161:164], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[251:254], v[165:168], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[251:254], v[169:172], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[251:254], v[173:176], v[1:8]
	s_clause 0x1f
	buffer_store_b32 v121, v0, s[0:3], null offen
	buffer_store_b32 v113, v0, s[0:3], null offen offset:64
	buffer_store_b32 v105, v0, s[0:3], null offen offset:128
	buffer_store_b32 v97, v0, s[0:3], null offen offset:192
	buffer_store_b32 v122, v0, s[0:3], null offen offset:8192
	buffer_store_b32 v114, v0, s[0:3], null offen offset:8256
	buffer_store_b32 v106, v0, s[0:3], null offen offset:8320
	buffer_store_b32 v98, v0, s[0:3], null offen offset:8384
	buffer_store_b32 v123, v0, s[0:3], null offen offset:16384
	buffer_store_b32 v115, v0, s[0:3], null offen offset:16448
	buffer_store_b32 v107, v0, s[0:3], null offen offset:16512
	buffer_store_b32 v99, v0, s[0:3], null offen offset:16576
	buffer_store_b32 v124, v0, s[0:3], null offen offset:24576
	buffer_store_b32 v116, v0, s[0:3], null offen offset:24640
	buffer_store_b32 v108, v0, s[0:3], null offen offset:24704
	buffer_store_b32 v100, v0, s[0:3], null offen offset:24768
	buffer_store_b32 v125, v0, s[0:3], null offen offset:32768
	buffer_store_b32 v117, v0, s[0:3], null offen offset:32832
	buffer_store_b32 v109, v0, s[0:3], null offen offset:32896
	buffer_store_b32 v101, v0, s[0:3], null offen offset:32960
	buffer_store_b32 v126, v0, s[0:3], null offen offset:40960
	buffer_store_b32 v118, v0, s[0:3], null offen offset:41024
	buffer_store_b32 v110, v0, s[0:3], null offen offset:41088
	buffer_store_b32 v102, v0, s[0:3], null offen offset:41152
	buffer_store_b32 v127, v0, s[0:3], null offen offset:49152
	buffer_store_b32 v119, v0, s[0:3], null offen offset:49216
	buffer_store_b32 v111, v0, s[0:3], null offen offset:49280
	buffer_store_b32 v103, v0, s[0:3], null offen offset:49344
	buffer_store_b32 v128, v0, s[0:3], null offen offset:57344
	buffer_store_b32 v120, v0, s[0:3], null offen offset:57408
	buffer_store_b32 v112, v0, s[0:3], null offen offset:57472
	buffer_store_b32 v104, v0, s[0:3], null offen offset:57536
	s_clause 0x1f
	buffer_store_b32 v89, v0, s[0:3], null offen offset:131072
	buffer_store_b32 v81, v0, s[0:3], null offen offset:131136
	buffer_store_b32 v73, v0, s[0:3], null offen offset:131200
	buffer_store_b32 v65, v0, s[0:3], null offen offset:131264
	buffer_store_b32 v90, v0, s[0:3], null offen offset:139264
	buffer_store_b32 v82, v0, s[0:3], null offen offset:139328
	buffer_store_b32 v74, v0, s[0:3], null offen offset:139392
	buffer_store_b32 v66, v0, s[0:3], null offen offset:139456
	buffer_store_b32 v91, v0, s[0:3], null offen offset:147456
	buffer_store_b32 v83, v0, s[0:3], null offen offset:147520
	buffer_store_b32 v75, v0, s[0:3], null offen offset:147584
	buffer_store_b32 v67, v0, s[0:3], null offen offset:147648
	buffer_store_b32 v92, v0, s[0:3], null offen offset:155648
	buffer_store_b32 v84, v0, s[0:3], null offen offset:155712
	buffer_store_b32 v76, v0, s[0:3], null offen offset:155776
	buffer_store_b32 v68, v0, s[0:3], null offen offset:155840
	buffer_store_b32 v93, v0, s[0:3], null offen offset:163840
	buffer_store_b32 v85, v0, s[0:3], null offen offset:163904
	buffer_store_b32 v77, v0, s[0:3], null offen offset:163968
	buffer_store_b32 v69, v0, s[0:3], null offen offset:164032
	buffer_store_b32 v94, v0, s[0:3], null offen offset:172032
	buffer_store_b32 v86, v0, s[0:3], null offen offset:172096
	buffer_store_b32 v78, v0, s[0:3], null offen offset:172160
	buffer_store_b32 v70, v0, s[0:3], null offen offset:172224
	buffer_store_b32 v95, v0, s[0:3], null offen offset:180224
	buffer_store_b32 v87, v0, s[0:3], null offen offset:180288
	buffer_store_b32 v79, v0, s[0:3], null offen offset:180352
	buffer_store_b32 v71, v0, s[0:3], null offen offset:180416
	buffer_store_b32 v96, v0, s[0:3], null offen offset:188416
	buffer_store_b32 v88, v0, s[0:3], null offen offset:188480
	buffer_store_b32 v80, v0, s[0:3], null offen offset:188544
	buffer_store_b32 v72, v0, s[0:3], null offen offset:188608
	s_clause 0x1f
	buffer_store_b32 v57, v0, s[0:3], null offen offset:262144
	buffer_store_b32 v49, v0, s[0:3], null offen offset:262208
	buffer_store_b32 v41, v0, s[0:3], null offen offset:262272
	buffer_store_b32 v33, v0, s[0:3], null offen offset:262336
	buffer_store_b32 v58, v0, s[0:3], null offen offset:270336
	buffer_store_b32 v50, v0, s[0:3], null offen offset:270400
	buffer_store_b32 v42, v0, s[0:3], null offen offset:270464
	buffer_store_b32 v34, v0, s[0:3], null offen offset:270528
	buffer_store_b32 v59, v0, s[0:3], null offen offset:278528
	buffer_store_b32 v51, v0, s[0:3], null offen offset:278592
	buffer_store_b32 v43, v0, s[0:3], null offen offset:278656
	buffer_store_b32 v35, v0, s[0:3], null offen offset:278720
	buffer_store_b32 v60, v0, s[0:3], null offen offset:286720
	buffer_store_b32 v52, v0, s[0:3], null offen offset:286784
	buffer_store_b32 v44, v0, s[0:3], null offen offset:286848
	buffer_store_b32 v36, v0, s[0:3], null offen offset:286912
	buffer_store_b32 v61, v0, s[0:3], null offen offset:294912
	buffer_store_b32 v53, v0, s[0:3], null offen offset:294976
	buffer_store_b32 v45, v0, s[0:3], null offen offset:295040
	buffer_store_b32 v37, v0, s[0:3], null offen offset:295104
	buffer_store_b32 v62, v0, s[0:3], null offen offset:303104
	buffer_store_b32 v54, v0, s[0:3], null offen offset:303168
	buffer_store_b32 v46, v0, s[0:3], null offen offset:303232
	buffer_store_b32 v38, v0, s[0:3], null offen offset:303296
	buffer_store_b32 v63, v0, s[0:3], null offen offset:311296
	buffer_store_b32 v55, v0, s[0:3], null offen offset:311360
	buffer_store_b32 v47, v0, s[0:3], null offen offset:311424
	buffer_store_b32 v39, v0, s[0:3], null offen offset:311488
	buffer_store_b32 v64, v0, s[0:3], null offen offset:319488
	buffer_store_b32 v56, v0, s[0:3], null offen offset:319552
	buffer_store_b32 v48, v0, s[0:3], null offen offset:319616
	buffer_store_b32 v40, v0, s[0:3], null offen offset:319680
	s_clause 0x1f
	buffer_store_b32 v25, v0, s[0:3], null offen offset:393216
	buffer_store_b32 v17, v0, s[0:3], null offen offset:393280
	buffer_store_b32 v9, v0, s[0:3], null offen offset:393344
	buffer_store_b32 v1, v0, s[0:3], null offen offset:393408
	buffer_store_b32 v26, v0, s[0:3], null offen offset:401408
	buffer_store_b32 v18, v0, s[0:3], null offen offset:401472
	buffer_store_b32 v10, v0, s[0:3], null offen offset:401536
	buffer_store_b32 v2, v0, s[0:3], null offen offset:401600
	buffer_store_b32 v27, v0, s[0:3], null offen offset:409600
	buffer_store_b32 v19, v0, s[0:3], null offen offset:409664
	buffer_store_b32 v11, v0, s[0:3], null offen offset:409728
	buffer_store_b32 v3, v0, s[0:3], null offen offset:409792
	buffer_store_b32 v28, v0, s[0:3], null offen offset:417792
	buffer_store_b32 v20, v0, s[0:3], null offen offset:417856
	buffer_store_b32 v12, v0, s[0:3], null offen offset:417920
	buffer_store_b32 v4, v0, s[0:3], null offen offset:417984
	buffer_store_b32 v29, v0, s[0:3], null offen offset:425984
	buffer_store_b32 v21, v0, s[0:3], null offen offset:426048
	buffer_store_b32 v13, v0, s[0:3], null offen offset:426112
	buffer_store_b32 v5, v0, s[0:3], null offen offset:426176
	buffer_store_b32 v30, v0, s[0:3], null offen offset:434176
	buffer_store_b32 v22, v0, s[0:3], null offen offset:434240
	buffer_store_b32 v14, v0, s[0:3], null offen offset:434304
	buffer_store_b32 v6, v0, s[0:3], null offen offset:434368
	buffer_store_b32 v31, v0, s[0:3], null offen offset:442368
	buffer_store_b32 v23, v0, s[0:3], null offen offset:442432
	buffer_store_b32 v15, v0, s[0:3], null offen offset:442496
	buffer_store_b32 v7, v0, s[0:3], null offen offset:442560
	buffer_store_b32 v32, v0, s[0:3], null offen offset:450560
	buffer_store_b32 v24, v0, s[0:3], null offen offset:450624
	buffer_store_b32 v16, v0, s[0:3], null offen offset:450688
	buffer_store_b32 v8, v0, s[0:3], null offen offset:450752
	s_barrier_signal -1
	s_barrier_wait -1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32, .Lfunc_end0-matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32
		.amdhsa_group_segment_fixed_size 51712
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 255
		.amdhsa_next_free_sgpr 17
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 57
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.num_vgpr, 255
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.num_agpr, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.numbered_sgpr, 17
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.num_named_barrier, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.private_seg_size, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.uses_vcc, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.uses_flat_scratch, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.has_dyn_sized_stack, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.has_recursion, 0
	.set .Lmatmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.has_indirect_call, 0
	.globl	matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32
	.p2align	8
	.type	matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32,@function
matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32:
	v_lshlrev_b32_e32 v4, 3, v0
	s_load_b128 s[12:15], s[0:1], 0x0
	v_or_b32_e32 v1, 0x100, v0
	v_lshlrev_b32_e32 v5, 4, v0
	s_lshl_b32 s3, ttmp9, 7
	v_and_b32_e32 v183, 0x78, v4
	v_or_b32_e32 v2, 0x200, v0
	v_lshrrev_b32_e32 v180, 3, v1
	s_lshl_b32 s2, ttmp9, 5
	v_or_b32_e32 v3, 0x300, v0
	s_and_b32 s3, s3, 0x380
	s_and_b32 s2, s2, 0x700
	v_and_b32_e32 v184, 0x70, v5
	v_or_b32_e32 v5, s3, v183
	v_lshrrev_b32_e32 v179, 3, v0
	v_lshrrev_b32_e32 v181, 3, v2
	v_lshrrev_b32_e32 v192, 4, v1
	v_or_b32_e32 v1, s2, v180
	v_lshrrev_b32_e32 v182, 3, v3
	v_lshrrev_b32_e32 v191, 4, v0
	v_lshlrev_b32_e32 v197, 1, v5
	v_or_b32_e32 v4, s2, v179
	v_lshl_or_b32 v196, v1, 13, v184
	v_or_b32_e32 v1, s2, v181
	v_lshrrev_b32_e32 v194, 4, v2
	v_or_b32_e32 v2, s2, v182
	v_lshrrev_b32_e32 v195, 4, v3
	v_lshl_or_b32 v3, v191, 11, v197
	s_mov_b32 s7, 0x31027000
	s_mov_b32 s10, 0x800000
	s_wait_kmcnt 0x0
	s_and_b32 s15, s15, 0xffff
	v_lshl_or_b32 v193, v4, 13, v184
	v_lshl_or_b32 v4, v192, 11, v197
	s_and_b32 s13, s13, 0xffff
	v_lshl_or_b32 v198, v1, 13, v184
	s_mov_b32 s18, s10
	s_mov_b32 s19, s7
	s_mov_b32 s16, s14
	s_mov_b32 s17, s15
	v_lshl_or_b32 v199, v2, 13, v184
	s_mov_b32 s6, 0x1000000
	s_mov_b32 s4, s12
	s_mov_b32 s5, s13
	v_lshl_or_b32 v1, v194, 11, v197
	s_clause 0x1
	buffer_load_b128 v[137:140], v3, s[16:19], null offen
	buffer_load_b128 v[141:144], v4, s[16:19], null offen
	s_clause 0x1
	buffer_load_b128 v[145:148], v198, s[4:7], null offen
	buffer_load_b128 v[149:152], v199, s[4:7], null offen
	buffer_load_b128 v[153:156], v1, s[16:19], null offen
	v_lshl_or_b32 v1, v195, 11, v197
	s_clause 0x5
	buffer_load_b128 v[129:132], v193, s[4:7], null offen
	buffer_load_b128 v[133:136], v196, s[4:7], null offen
	buffer_load_b128 v[157:160], v193, s[4:7], null offen offset:1048576
	buffer_load_b128 v[161:164], v193, s[4:7], null offen offset:1310720
	buffer_load_b128 v[165:168], v193, s[4:7], null offen offset:1572864
	buffer_load_b128 v[169:172], v193, s[4:7], null offen offset:1835008
	buffer_load_b128 v[173:176], v1, s[16:19], null offen
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v178, 15, v0
	v_lshrrev_b32_e32 v177, 1, v0
	v_lshlrev_b32_e32 v187, 1, v0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v8, v1 :: v_dual_and_b32 v185, 0xcf, v0
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v10, v1
	v_and_b32_e32 v177, 8, v177
	v_and_or_b32 v178, v187, 64, v178
	v_or_b32_e32 v186, 48, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_mul_u32_u24_e32 v185, 0x44, v185
	v_mad_u32_u24 v202, 0x88, v179, v184
	v_mad_u32_u24 v203, 0x88, v180, v184
	v_mad_u32_u24 v204, 0x88, v181, v184
	v_mad_u32_u24 v205, 0x88, v182, v184
	v_lshlrev_b32_e32 v179, 1, v183
	v_mul_u32_u24_e32 v181, 0x108, v194
	v_lshlrev_b32_e32 v183, 1, v177
	v_mul_u32_u24_e32 v184, 0x84, v177
	v_lshlrev_b32_e32 v187, 1, v178
	v_mul_u32_u24_e32 v188, 0x108, v191
	v_mul_u32_u24_e32 v186, 0x44, v186
	v_mul_u32_u24_e32 v180, 0x108, v192
	v_mul_u32_u24_e32 v182, 0x108, v195
	v_add3_u32 v212, v181, v179, 0x8800
	v_lshl_add_u32 v181, v184, 1, v187
	v_lshl_add_u32 v185, v185, 1, v183
	v_mov_b32_e32 v2, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_dual_mov_b32 v12, v1 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v14, v1 :: v_dual_mov_b32 v11, v1
	v_dual_mov_b32 v16, v1 :: v_dual_mov_b32 v13, v1
	v_dual_mov_b32 v18, v1 :: v_dual_mov_b32 v15, v1
	v_dual_mov_b32 v20, v1 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v22, v1 :: v_dual_mov_b32 v19, v1
	v_dual_mov_b32 v24, v1 :: v_dual_mov_b32 v21, v1
	v_dual_mov_b32 v26, v1 :: v_dual_mov_b32 v23, v1
	v_dual_mov_b32 v28, v1 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v30, v1 :: v_dual_mov_b32 v27, v1
	v_dual_mov_b32 v32, v1 :: v_dual_mov_b32 v29, v1
	v_dual_mov_b32 v34, v1 :: v_dual_mov_b32 v31, v1
	v_dual_mov_b32 v36, v1 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v38, v1 :: v_dual_mov_b32 v35, v1
	v_dual_mov_b32 v40, v1 :: v_dual_mov_b32 v37, v1
	v_dual_mov_b32 v42, v1 :: v_dual_mov_b32 v39, v1
	v_dual_mov_b32 v44, v1 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v46, v1 :: v_dual_mov_b32 v43, v1
	v_dual_mov_b32 v48, v1 :: v_dual_mov_b32 v45, v1
	v_dual_mov_b32 v50, v1 :: v_dual_mov_b32 v47, v1
	v_dual_mov_b32 v52, v1 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v54, v1 :: v_dual_mov_b32 v51, v1
	v_dual_mov_b32 v56, v1 :: v_dual_mov_b32 v53, v1
	v_dual_mov_b32 v58, v1 :: v_dual_mov_b32 v55, v1
	v_dual_mov_b32 v60, v1 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v62, v1 :: v_dual_mov_b32 v59, v1
	v_dual_mov_b32 v64, v1 :: v_dual_mov_b32 v61, v1
	v_dual_mov_b32 v66, v1 :: v_dual_mov_b32 v63, v1
	v_dual_mov_b32 v68, v1 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v70, v1 :: v_dual_mov_b32 v67, v1
	v_dual_mov_b32 v72, v1 :: v_dual_mov_b32 v69, v1
	v_dual_mov_b32 v74, v1 :: v_dual_mov_b32 v71, v1
	v_dual_mov_b32 v76, v1 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v78, v1 :: v_dual_mov_b32 v75, v1
	v_dual_mov_b32 v80, v1 :: v_dual_mov_b32 v77, v1
	v_dual_mov_b32 v82, v1 :: v_dual_mov_b32 v79, v1
	v_dual_mov_b32 v84, v1 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v86, v1 :: v_dual_mov_b32 v83, v1
	v_dual_mov_b32 v88, v1 :: v_dual_mov_b32 v85, v1
	v_dual_mov_b32 v90, v1 :: v_dual_mov_b32 v87, v1
	v_dual_mov_b32 v92, v1 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v94, v1 :: v_dual_mov_b32 v91, v1
	v_dual_mov_b32 v96, v1 :: v_dual_mov_b32 v93, v1
	v_dual_mov_b32 v98, v1 :: v_dual_mov_b32 v95, v1
	v_dual_mov_b32 v100, v1 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v102, v1 :: v_dual_mov_b32 v99, v1
	v_dual_mov_b32 v104, v1 :: v_dual_mov_b32 v101, v1
	v_dual_mov_b32 v106, v1 :: v_dual_mov_b32 v103, v1
	v_dual_mov_b32 v108, v1 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v110, v1 :: v_dual_mov_b32 v107, v1
	v_dual_mov_b32 v112, v1 :: v_dual_mov_b32 v109, v1
	v_dual_mov_b32 v114, v1 :: v_dual_mov_b32 v111, v1
	v_dual_mov_b32 v116, v1 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v118, v1 :: v_dual_mov_b32 v115, v1
	v_dual_mov_b32 v120, v1 :: v_dual_mov_b32 v117, v1
	v_dual_mov_b32 v122, v1 :: v_dual_mov_b32 v119, v1
	v_dual_mov_b32 v124, v1 :: v_dual_mov_b32 v121, v1
	v_dual_mov_b32 v126, v1 :: v_dual_mov_b32 v123, v1
	v_dual_mov_b32 v128, v1 :: v_dual_mov_b32 v125, v1
	v_dual_mov_b32 v127, v1 :: v_dual_add_nc_u32 v206, 0x4400, v202
	v_add_nc_u32_e32 v208, 0x6600, v202
	v_add_nc_u32_e32 v207, 0x5500, v202
	v_add_nc_u32_e32 v209, 0x7700, v202
	v_add3_u32 v210, v188, v179, 0x8800
	v_add3_u32 v211, v180, v179, 0x8800
	v_add3_u32 v213, v182, v179, 0x8800
	v_lshl_add_u32 v180, v186, 1, v183
	v_add_nc_u32_e32 v179, 0x8800, v181
	v_add_nc_u32_e32 v190, 0x9880, v181
	v_add_nc_u32_e32 v201, 0xa900, v181
	v_add_nc_u32_e32 v200, 0xb980, v181
	v_add_nc_u32_e32 v188, 0x880, v185
	v_add_nc_u32_e32 v187, 0x8a0, v185
	v_add_nc_u32_e32 v183, 0x8c0, v185
	v_add_nc_u32_e32 v184, 0x8e0, v185
	v_add_nc_u32_e32 v189, 0x1100, v185
	v_add_nc_u32_e32 v186, 0x1120, v185
	v_add_nc_u32_e32 v181, 0x1140, v185
	v_add_nc_u32_e32 v182, 0x1160, v185
	s_mov_b32 s16, 0
	s_mov_b32 s11, s7
	s_mov_b32 s8, s14
	s_mov_b32 s9, s15
	s_wait_loadcnt 0x6
	ds_store_2addr_b64 v202, v[129:130], v[131:132] offset1:1
	ds_store_2addr_b64 v210, v[137:138], v[139:140] offset1:1
	s_wait_loadcnt 0x5
	ds_store_2addr_b64 v203, v[133:134], v[135:136] offset1:1
	ds_store_2addr_b64 v211, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v204, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v212, v[153:154], v[155:156] offset1:1
	ds_store_2addr_b64 v205, v[149:150], v[151:152] offset1:1
	s_wait_loadcnt 0x4
	ds_store_2addr_b64 v206, v[157:158], v[159:160] offset1:1
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v207, v[161:162], v[163:164] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v208, v[165:166], v[167:168] offset1:1
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v209, v[169:170], v[171:172] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v213, v[173:174], v[175:176] offset1:1
.LBB1_1:
	s_add_co_i32 s12, s16, 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s13, s12, 4
	s_lshl_b32 s14, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v129, s13, v191
	v_or_b32_e32 v130, s13, v192
	v_or_b32_e32 v131, s13, v194
	v_or_b32_e32 v132, s13, v195
	v_add_nc_u32_e32 v133, s14, v198
	v_add_nc_u32_e32 v134, s14, v199
	v_add_nc_u32_e32 v141, s14, v196
	v_add_nc_u32_e32 v153, s14, v193
	v_lshl_or_b32 v129, v129, 11, v197
	v_lshl_or_b32 v130, v130, 11, v197
	v_lshl_or_b32 v131, v131, 11, v197
	v_lshl_or_b32 v132, v132, 11, v197
	s_clause 0x7
	buffer_load_b128 v[137:140], v133, s[4:7], null offen
	buffer_load_b128 v[133:136], v134, s[4:7], null offen
	buffer_load_b128 v[157:160], v141, s[4:7], null offen
	buffer_load_b128 v[173:176], v153, s[4:7], null offen
	buffer_load_b128 v[141:144], v153, s[4:7], null offen offset:1048576
	buffer_load_b128 v[145:148], v153, s[4:7], null offen offset:1310720
	buffer_load_b128 v[149:152], v153, s[4:7], null offen offset:1572864
	buffer_load_b128 v[153:156], v153, s[4:7], null offen offset:1835008
	s_clause 0x3
	buffer_load_b128 v[169:172], v129, s[8:11], null offen
	buffer_load_b128 v[165:168], v130, s[8:11], null offen
	buffer_load_b128 v[161:164], v131, s[8:11], null offen
	buffer_load_b128 v[129:132], v132, s[8:11], null offen
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_u16_d16 v214, v179
	ds_load_u16_d16 v215, v179 offset:528
	ds_load_u16_d16 v216, v179 offset:1056
	ds_load_u16_d16 v217, v179 offset:1584
	ds_load_2addr_b64 v[222:225], v188 offset1:1
	ds_load_2addr_b64 v[230:233], v180 offset1:1
	ds_load_2addr_b64 v[226:229], v189 offset1:1
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v214, v179 offset:264
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v215, v179 offset:792
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v216, v179 offset:1320
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v217, v179 offset:1848
	ds_load_2addr_b64 v[218:221], v185 offset1:1
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[89:96], v[222:225], v[214:217], v[89:96]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[218:221], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[57:64], v[226:229], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	ds_load_u16_d16 v216, v179 offset:1120
	ds_load_u16_d16 v214, v179 offset:64
	ds_load_u16_d16 v215, v179 offset:592
	ds_load_u16_d16 v217, v179 offset:1648
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v216, v179 offset:1384
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v214, v179 offset:328
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v215, v179 offset:856
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v217, v179 offset:1912
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[105:112], v[218:221], v[214:217], v[105:112]
	v_wmma_f32_16x16x16_f16 v[73:80], v[222:225], v[214:217], v[73:80]
	v_wmma_f32_16x16x16_f16 v[41:48], v[226:229], v[214:217], v[41:48]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[214:217], v[9:16]
	ds_load_u16_d16 v214, v179 offset:32
	ds_load_u16_d16 v215, v179 offset:560
	ds_load_u16_d16 v234, v179 offset:96
	ds_load_u16_d16 v235, v179 offset:624
	ds_load_u16_d16 v216, v179 offset:1088
	ds_load_u16_d16 v236, v179 offset:1152
	ds_load_u16_d16 v217, v179 offset:1616
	ds_load_u16_d16 v237, v179 offset:1680
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v214, v179 offset:296
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v234, v179 offset:360
	ds_load_u16_d16_hi v215, v179 offset:824
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v235, v179 offset:888
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v216, v179 offset:1352
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v236, v179 offset:1416
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v217, v179 offset:1880
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v237, v179 offset:1944
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[113:120], v[218:221], v[214:217], v[113:120]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[218:221], v[234:237], v[97:104]
	v_wmma_f32_16x16x16_f16 v[81:88], v[222:225], v[214:217], v[81:88]
	v_wmma_f32_16x16x16_f16 v[65:72], v[222:225], v[234:237], v[65:72]
	v_wmma_f32_16x16x16_f16 v[49:56], v[226:229], v[214:217], v[49:56]
	v_wmma_f32_16x16x16_f16 v[33:40], v[226:229], v[234:237], v[33:40]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[214:217], v[17:24]
	ds_load_u16_d16 v214, v190
	ds_load_u16_d16 v215, v179 offset:4752
	ds_load_u16_d16 v218, v190 offset:32
	ds_load_u16_d16 v219, v179 offset:4784
	ds_load_u16_d16 v222, v190 offset:64
	ds_load_u16_d16 v223, v179 offset:4816
	ds_load_u16_d16 v226, v190 offset:96
	ds_load_u16_d16 v227, v179 offset:4848
	ds_load_u16_d16 v216, v179 offset:5280
	ds_load_u16_d16 v220, v179 offset:5312
	ds_load_u16_d16 v224, v179 offset:5344
	ds_load_u16_d16 v228, v179 offset:5376
	ds_load_u16_d16 v217, v179 offset:5808
	ds_load_u16_d16 v221, v179 offset:5840
	ds_load_u16_d16 v225, v179 offset:5872
	ds_load_u16_d16 v229, v179 offset:5904
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[234:237], v[1:8]
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v214, v179 offset:4488
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v222, v179 offset:4552
	ds_load_u16_d16_hi v218, v179 offset:4520
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v226, v179 offset:4584
	ds_load_u16_d16_hi v215, v179 offset:5016
	ds_load_u16_d16_hi v219, v179 offset:5048
	ds_load_u16_d16_hi v223, v179 offset:5080
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:5112
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:5544
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:5576
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:5608
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:5640
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:6072
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:6104
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:6136
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:6168
	ds_load_2addr_b64 v[230:233], v185 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[218:221], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[222:225], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[226:229], v[97:104]
	ds_load_2addr_b64 v[230:233], v187 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[214:217], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[218:221], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[222:225], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[226:229], v[65:72]
	ds_load_2addr_b64 v[230:233], v186 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[218:221], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[222:225], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[226:229], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[218:221], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[222:225], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[226:229], v[1:8]
	ds_load_u16_d16 v214, v201
	ds_load_u16_d16 v215, v179 offset:8976
	ds_load_u16_d16 v218, v201 offset:32
	ds_load_u16_d16 v219, v179 offset:9008
	ds_load_u16_d16 v222, v201 offset:64
	ds_load_u16_d16 v223, v179 offset:9040
	ds_load_u16_d16 v226, v201 offset:96
	ds_load_u16_d16 v227, v179 offset:9072
	ds_load_u16_d16 v216, v179 offset:9504
	ds_load_u16_d16 v224, v179 offset:9568
	ds_load_u16_d16 v220, v179 offset:9536
	ds_load_u16_d16 v228, v179 offset:9600
	ds_load_u16_d16 v217, v179 offset:10032
	ds_load_u16_d16 v225, v179 offset:10096
	ds_load_u16_d16 v221, v179 offset:10064
	ds_load_u16_d16 v229, v179 offset:10128
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v214, v179 offset:8712
	s_wait_dscnt 0xe
	ds_load_u16_d16_hi v218, v179 offset:8744
	s_wait_dscnt 0xd
	ds_load_u16_d16_hi v222, v179 offset:8776
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v226, v179 offset:8808
	ds_load_u16_d16_hi v215, v179 offset:9240
	ds_load_u16_d16_hi v219, v179 offset:9272
	ds_load_u16_d16_hi v223, v179 offset:9304
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:9336
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:9768
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:9832
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:9800
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:9864
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:10296
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:10360
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:10328
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:10392
	ds_load_2addr_b64 v[230:233], v185 offset0:8 offset1:9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[218:221], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[222:225], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[226:229], v[97:104]
	ds_load_2addr_b64 v[230:233], v183 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[214:217], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[218:221], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[222:225], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[226:229], v[65:72]
	ds_load_2addr_b64 v[230:233], v181 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[218:221], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[222:225], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[226:229], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:8 offset1:9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[218:221], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[222:225], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[226:229], v[1:8]
	ds_load_u16_d16 v217, v179 offset:14352
	ds_load_u16_d16 v214, v200 offset:96
	ds_load_u16_d16 v218, v200
	ds_load_u16_d16 v219, v179 offset:13200
	ds_load_u16_d16 v222, v200 offset:32
	ds_load_u16_d16 v223, v179 offset:13232
	ds_load_u16_d16 v226, v200 offset:64
	ds_load_u16_d16 v227, v179 offset:13264
	ds_load_u16_d16 v215, v179 offset:13296
	ds_load_u16_d16 v220, v179 offset:13728
	ds_load_u16_d16 v224, v179 offset:13760
	ds_load_u16_d16 v228, v179 offset:13792
	ds_load_u16_d16 v216, v179 offset:13824
	ds_load_u16_d16 v221, v179 offset:14256
	ds_load_u16_d16 v225, v179 offset:14288
	ds_load_u16_d16 v229, v179 offset:14320
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:14616
	s_wait_dscnt 0xe
	ds_load_u16_d16_hi v218, v179 offset:12936
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v226, v179 offset:13000
	ds_load_u16_d16_hi v222, v179 offset:12968
	ds_load_u16_d16_hi v214, v179 offset:13032
	ds_load_u16_d16_hi v219, v179 offset:13464
	ds_load_u16_d16_hi v223, v179 offset:13496
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:13528
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v215, v179 offset:13560
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:13992
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:14024
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:14056
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:14088
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:14520
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:14552
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:14584
	ds_load_2addr_b64 v[230:233], v185 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[218:221], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[222:225], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[226:229], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[214:217], v[97:104]
	ds_load_2addr_b64 v[230:233], v184 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[218:221], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[222:225], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[226:229], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[214:217], v[65:72]
	ds_load_2addr_b64 v[230:233], v182 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[218:221], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[222:225], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[226:229], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[214:217], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[218:221], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[222:225], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[226:229], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[214:217], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_lt_u32 s16, 0xf8
	s_mov_b32 s16, s12
	s_wait_loadcnt 0x8
	ds_store_2addr_b64 v202, v[173:174], v[175:176] offset1:1
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v210, v[169:170], v[171:172] offset1:1
	ds_store_2addr_b64 v203, v[157:158], v[159:160] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v211, v[165:166], v[167:168] offset1:1
	ds_store_2addr_b64 v204, v[137:138], v[139:140] offset1:1
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v212, v[161:162], v[163:164] offset1:1
	ds_store_2addr_b64 v205, v[133:134], v[135:136] offset1:1
	ds_store_2addr_b64 v206, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v207, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v208, v[149:150], v[151:152] offset1:1
	ds_store_2addr_b64 v209, v[153:154], v[155:156] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v213, v[129:130], v[131:132] offset1:1
	s_cbranch_scc1 .LBB1_1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v0, 0xc0, v0
	v_or_b32_e32 v178, s3, v178
	s_wait_kmcnt 0x0
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x31027000
	v_or3_b32 v0, s2, v0, v177
	v_lshlrev_b32_e32 v177, 2, v178
	s_mov_b32 s2, 0x800000
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v0, v0, 12, v177
	s_barrier_wait -1
	ds_load_u16_d16 v137, v179
	ds_load_u16_d16 v129, v179 offset:32
	ds_load_u16_d16 v141, v179 offset:64
	ds_load_u16_d16 v133, v179 offset:96
	ds_load_u16_d16 v138, v179 offset:528
	ds_load_u16_d16 v142, v179 offset:592
	ds_load_u16_d16 v146, v179 offset:8976
	ds_load_u16_d16 v150, v179 offset:9008
	ds_load_u16_d16 v154, v179 offset:9040
	ds_load_u16_d16 v158, v179 offset:9072
	ds_load_u16_d16 v147, v179 offset:9504
	ds_load_u16_d16 v155, v179 offset:9568
	ds_load_u16_d16 v148, v179 offset:10032
	ds_load_u16_d16 v156, v179 offset:10096
	ds_load_u16_d16 v151, v179 offset:9536
	ds_load_u16_d16 v159, v179 offset:9600
	ds_load_u16_d16 v145, v201
	ds_load_u16_d16 v149, v201 offset:32
	ds_load_u16_d16 v153, v201 offset:64
	ds_load_u16_d16 v157, v201 offset:96
	ds_load_u16_d16 v161, v200
	ds_load_u16_d16 v165, v200 offset:32
	ds_load_u16_d16 v169, v200 offset:64
	ds_load_u16_d16 v173, v200 offset:96
	ds_load_u16_d16 v152, v179 offset:10064
	ds_load_u16_d16 v160, v179 offset:10128
	ds_load_u16_d16 v130, v179 offset:560
	ds_load_u16_d16 v134, v179 offset:624
	ds_load_u16_d16 v162, v179 offset:13200
	ds_load_u16_d16 v166, v179 offset:13232
	ds_load_u16_d16 v170, v179 offset:13264
	ds_load_u16_d16 v174, v179 offset:13296
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v146, v179 offset:9240
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v150, v179 offset:9272
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v154, v179 offset:9304
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v158, v179 offset:9336
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v147, v179 offset:9768
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v155, v179 offset:9832
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v151, v179 offset:9800
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v159, v179 offset:9864
	ds_load_u16_d16_hi v148, v179 offset:10296
	ds_load_u16_d16_hi v156, v179 offset:10360
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v161, v179 offset:12936
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v169, v179 offset:13000
	s_wait_dscnt 0x13
	ds_load_u16_d16_hi v152, v179 offset:10328
	s_wait_dscnt 0x13
	ds_load_u16_d16_hi v160, v179 offset:10392
	ds_load_u16_d16_hi v165, v179 offset:12968
	ds_load_u16_d16_hi v173, v179 offset:13032
	ds_load_u16_d16 v139, v179 offset:1056
	ds_load_u16_d16 v143, v179 offset:1120
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v162, v179 offset:13464
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v166, v179 offset:13496
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v170, v179 offset:13528
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v174, v179 offset:13560
	ds_load_u16_d16 v163, v179 offset:13728
	ds_load_u16_d16 v167, v179 offset:13760
	ds_load_u16_d16 v171, v179 offset:13792
	ds_load_u16_d16 v175, v179 offset:13824
	ds_load_u16_d16_hi v137, v179 offset:264
	ds_load_u16_d16_hi v129, v179 offset:296
	ds_load_u16_d16_hi v141, v179 offset:328
	ds_load_u16_d16_hi v133, v179 offset:360
	ds_load_u16_d16_hi v138, v179 offset:792
	ds_load_u16_d16_hi v142, v179 offset:856
	ds_load_u16_d16 v176, v179 offset:14352
	ds_load_u16_d16_hi v130, v179 offset:824
	ds_load_u16_d16_hi v134, v179 offset:888
	ds_load_u16_d16 v131, v179 offset:1088
	ds_load_u16_d16 v135, v179 offset:1152
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v139, v179 offset:1320
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v143, v179 offset:1384
	ds_load_u16_d16 v140, v179 offset:1584
	ds_load_u16_d16 v144, v179 offset:1648
	ds_load_u16_d16 v191, v190
	ds_load_u16_d16 v195, v190 offset:32
	ds_load_u16_d16 v199, v190 offset:64
	ds_load_u16_d16 v203, v190 offset:96
	ds_load_u16_d16 v132, v179 offset:1616
	ds_load_u16_d16 v136, v179 offset:1680
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v163, v179 offset:13992
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v167, v179 offset:14024
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v171, v179 offset:14056
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v175, v179 offset:14088
	ds_load_u16_d16 v164, v179 offset:14256
	ds_load_u16_d16 v168, v179 offset:14288
	ds_load_u16_d16 v172, v179 offset:14320
	ds_load_u16_d16 v192, v179 offset:4752
	ds_load_u16_d16 v196, v179 offset:4784
	ds_load_u16_d16 v200, v179 offset:4816
	ds_load_u16_d16 v204, v179 offset:4848
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v176, v179 offset:14616
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v131, v179 offset:1352
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v135, v179 offset:1416
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v140, v179 offset:1848
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v144, v179 offset:1912
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v191, v179 offset:4488
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v199, v179 offset:4552
	ds_load_u16_d16 v193, v179 offset:5280
	ds_load_u16_d16 v197, v179 offset:5312
	ds_load_u16_d16 v201, v179 offset:5344
	ds_load_u16_d16 v205, v179 offset:5376
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v132, v179 offset:1880
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v136, v179 offset:1944
	ds_load_u16_d16_hi v195, v179 offset:4520
	ds_load_u16_d16_hi v203, v179 offset:4584
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v164, v179 offset:14520
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v168, v179 offset:14552
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v172, v179 offset:14584
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v192, v179 offset:5016
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v196, v179 offset:5048
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v200, v179 offset:5080
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v204, v179 offset:5112
	ds_load_2addr_b64 v[207:210], v185 offset1:1
	ds_load_2addr_b64 v[211:214], v188 offset1:1
	ds_load_2addr_b64 v[215:218], v189 offset1:1
	ds_load_2addr_b64 v[219:222], v180 offset1:1
	ds_load_2addr_b64 v[223:226], v185 offset0:4 offset1:5
	ds_load_2addr_b64 v[227:230], v185 offset0:8 offset1:9
	ds_load_2addr_b64 v[231:234], v185 offset0:12 offset1:13
	ds_load_2addr_b64 v[187:190], v187 offset1:1
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v193, v179 offset:5544
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v197, v179 offset:5576
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v201, v179 offset:5608
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v205, v179 offset:5640
	ds_load_u16_d16 v194, v179 offset:5808
	ds_load_u16_d16 v198, v179 offset:5840
	ds_load_u16_d16 v202, v179 offset:5872
	ds_load_u16_d16 v206, v179 offset:5904
	ds_load_2addr_b64 v[235:238], v183 offset1:1
	ds_load_2addr_b64 v[239:242], v184 offset1:1
	ds_load_2addr_b64 v[183:186], v186 offset1:1
	ds_load_2addr_b64 v[243:246], v180 offset0:4 offset1:5
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x16_f16 v[121:128], v[207:210], v[137:140], v[121:128]
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x16_f16 v[89:96], v[211:214], v[137:140], v[89:96]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x16_f16 v[57:64], v[215:218], v[137:140], v[57:64]
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x16_f16 v[25:32], v[219:222], v[137:140], v[25:32]
	ds_load_2addr_b64 v[137:140], v181 offset1:1
	ds_load_2addr_b64 v[247:250], v182 offset1:1
	v_wmma_f32_16x16x16_f16 v[105:112], v[207:210], v[141:144], v[105:112]
	v_wmma_f32_16x16x16_f16 v[73:80], v[211:214], v[141:144], v[73:80]
	v_wmma_f32_16x16x16_f16 v[41:48], v[215:218], v[141:144], v[41:48]
	v_wmma_f32_16x16x16_f16 v[9:16], v[219:222], v[141:144], v[9:16]
	ds_load_2addr_b64 v[141:144], v180 offset0:8 offset1:9
	ds_load_2addr_b64 v[251:254], v180 offset0:12 offset1:13
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v194, v179 offset:6072
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v198, v179 offset:6104
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v202, v179 offset:6136
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v206, v179 offset:6168
	ds_load_u16_d16_hi v145, v179 offset:8712
	ds_load_u16_d16_hi v149, v179 offset:8744
	ds_load_u16_d16_hi v153, v179 offset:8776
	ds_load_u16_d16_hi v157, v179 offset:8808
	v_wmma_f32_16x16x16_f16 v[113:120], v[207:210], v[129:132], v[113:120]
	v_wmma_f32_16x16x16_f16 v[97:104], v[207:210], v[133:136], v[97:104]
	v_wmma_f32_16x16x16_f16 v[81:88], v[211:214], v[129:132], v[81:88]
	v_wmma_f32_16x16x16_f16 v[65:72], v[211:214], v[133:136], v[65:72]
	v_wmma_f32_16x16x16_f16 v[49:56], v[215:218], v[129:132], v[49:56]
	v_wmma_f32_16x16x16_f16 v[33:40], v[215:218], v[133:136], v[33:40]
	v_wmma_f32_16x16x16_f16 v[17:24], v[219:222], v[129:132], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[219:222], v[133:136], v[1:8]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_f16 v[121:128], v[223:226], v[191:194], v[121:128]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_f16 v[113:120], v[223:226], v[195:198], v[113:120]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_f16 v[105:112], v[223:226], v[199:202], v[105:112]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_f16 v[97:104], v[223:226], v[203:206], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[187:190], v[191:194], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[187:190], v[195:198], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[187:190], v[199:202], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[187:190], v[203:206], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[183:186], v[191:194], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[183:186], v[195:198], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[183:186], v[199:202], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[183:186], v[203:206], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[243:246], v[191:194], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[243:246], v[195:198], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[243:246], v[199:202], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[243:246], v[203:206], v[1:8]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[121:128], v[227:230], v[145:148], v[121:128]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_f16 v[113:120], v[227:230], v[149:152], v[113:120]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[105:112], v[227:230], v[153:156], v[105:112]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[227:230], v[157:160], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[235:238], v[145:148], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[235:238], v[149:152], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[235:238], v[153:156], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[235:238], v[157:160], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[137:140], v[145:148], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[137:140], v[149:152], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[137:140], v[153:156], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[137:140], v[157:160], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[141:144], v[145:148], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[141:144], v[149:152], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[141:144], v[153:156], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[141:144], v[157:160], v[1:8]
	v_wmma_f32_16x16x16_f16 v[121:128], v[231:234], v[161:164], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[231:234], v[165:168], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[231:234], v[169:172], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[231:234], v[173:176], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[239:242], v[161:164], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[239:242], v[165:168], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[239:242], v[169:172], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[239:242], v[173:176], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[247:250], v[161:164], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[247:250], v[165:168], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[247:250], v[169:172], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[247:250], v[173:176], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[251:254], v[161:164], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[251:254], v[165:168], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[251:254], v[169:172], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[251:254], v[173:176], v[1:8]
	s_clause 0x1f
	buffer_store_b32 v121, v0, s[0:3], null offen
	buffer_store_b32 v113, v0, s[0:3], null offen offset:64
	buffer_store_b32 v105, v0, s[0:3], null offen offset:128
	buffer_store_b32 v97, v0, s[0:3], null offen offset:192
	buffer_store_b32 v122, v0, s[0:3], null offen offset:4096
	buffer_store_b32 v114, v0, s[0:3], null offen offset:4160
	buffer_store_b32 v106, v0, s[0:3], null offen offset:4224
	buffer_store_b32 v98, v0, s[0:3], null offen offset:4288
	buffer_store_b32 v123, v0, s[0:3], null offen offset:8192
	buffer_store_b32 v115, v0, s[0:3], null offen offset:8256
	buffer_store_b32 v107, v0, s[0:3], null offen offset:8320
	buffer_store_b32 v99, v0, s[0:3], null offen offset:8384
	buffer_store_b32 v124, v0, s[0:3], null offen offset:12288
	buffer_store_b32 v116, v0, s[0:3], null offen offset:12352
	buffer_store_b32 v108, v0, s[0:3], null offen offset:12416
	buffer_store_b32 v100, v0, s[0:3], null offen offset:12480
	buffer_store_b32 v125, v0, s[0:3], null offen offset:16384
	buffer_store_b32 v117, v0, s[0:3], null offen offset:16448
	buffer_store_b32 v109, v0, s[0:3], null offen offset:16512
	buffer_store_b32 v101, v0, s[0:3], null offen offset:16576
	buffer_store_b32 v126, v0, s[0:3], null offen offset:20480
	buffer_store_b32 v118, v0, s[0:3], null offen offset:20544
	buffer_store_b32 v110, v0, s[0:3], null offen offset:20608
	buffer_store_b32 v102, v0, s[0:3], null offen offset:20672
	buffer_store_b32 v127, v0, s[0:3], null offen offset:24576
	buffer_store_b32 v119, v0, s[0:3], null offen offset:24640
	buffer_store_b32 v111, v0, s[0:3], null offen offset:24704
	buffer_store_b32 v103, v0, s[0:3], null offen offset:24768
	buffer_store_b32 v128, v0, s[0:3], null offen offset:28672
	buffer_store_b32 v120, v0, s[0:3], null offen offset:28736
	buffer_store_b32 v112, v0, s[0:3], null offen offset:28800
	buffer_store_b32 v104, v0, s[0:3], null offen offset:28864
	s_clause 0x1f
	buffer_store_b32 v89, v0, s[0:3], null offen offset:65536
	buffer_store_b32 v81, v0, s[0:3], null offen offset:65600
	buffer_store_b32 v73, v0, s[0:3], null offen offset:65664
	buffer_store_b32 v65, v0, s[0:3], null offen offset:65728
	buffer_store_b32 v90, v0, s[0:3], null offen offset:69632
	buffer_store_b32 v82, v0, s[0:3], null offen offset:69696
	buffer_store_b32 v74, v0, s[0:3], null offen offset:69760
	buffer_store_b32 v66, v0, s[0:3], null offen offset:69824
	buffer_store_b32 v91, v0, s[0:3], null offen offset:73728
	buffer_store_b32 v83, v0, s[0:3], null offen offset:73792
	buffer_store_b32 v75, v0, s[0:3], null offen offset:73856
	buffer_store_b32 v67, v0, s[0:3], null offen offset:73920
	buffer_store_b32 v92, v0, s[0:3], null offen offset:77824
	buffer_store_b32 v84, v0, s[0:3], null offen offset:77888
	buffer_store_b32 v76, v0, s[0:3], null offen offset:77952
	buffer_store_b32 v68, v0, s[0:3], null offen offset:78016
	buffer_store_b32 v93, v0, s[0:3], null offen offset:81920
	buffer_store_b32 v85, v0, s[0:3], null offen offset:81984
	buffer_store_b32 v77, v0, s[0:3], null offen offset:82048
	buffer_store_b32 v69, v0, s[0:3], null offen offset:82112
	buffer_store_b32 v94, v0, s[0:3], null offen offset:86016
	buffer_store_b32 v86, v0, s[0:3], null offen offset:86080
	buffer_store_b32 v78, v0, s[0:3], null offen offset:86144
	buffer_store_b32 v70, v0, s[0:3], null offen offset:86208
	buffer_store_b32 v95, v0, s[0:3], null offen offset:90112
	buffer_store_b32 v87, v0, s[0:3], null offen offset:90176
	buffer_store_b32 v79, v0, s[0:3], null offen offset:90240
	buffer_store_b32 v71, v0, s[0:3], null offen offset:90304
	buffer_store_b32 v96, v0, s[0:3], null offen offset:94208
	buffer_store_b32 v88, v0, s[0:3], null offen offset:94272
	buffer_store_b32 v80, v0, s[0:3], null offen offset:94336
	buffer_store_b32 v72, v0, s[0:3], null offen offset:94400
	s_clause 0x1f
	buffer_store_b32 v57, v0, s[0:3], null offen offset:131072
	buffer_store_b32 v49, v0, s[0:3], null offen offset:131136
	buffer_store_b32 v41, v0, s[0:3], null offen offset:131200
	buffer_store_b32 v33, v0, s[0:3], null offen offset:131264
	buffer_store_b32 v58, v0, s[0:3], null offen offset:135168
	buffer_store_b32 v50, v0, s[0:3], null offen offset:135232
	buffer_store_b32 v42, v0, s[0:3], null offen offset:135296
	buffer_store_b32 v34, v0, s[0:3], null offen offset:135360
	buffer_store_b32 v59, v0, s[0:3], null offen offset:139264
	buffer_store_b32 v51, v0, s[0:3], null offen offset:139328
	buffer_store_b32 v43, v0, s[0:3], null offen offset:139392
	buffer_store_b32 v35, v0, s[0:3], null offen offset:139456
	buffer_store_b32 v60, v0, s[0:3], null offen offset:143360
	buffer_store_b32 v52, v0, s[0:3], null offen offset:143424
	buffer_store_b32 v44, v0, s[0:3], null offen offset:143488
	buffer_store_b32 v36, v0, s[0:3], null offen offset:143552
	buffer_store_b32 v61, v0, s[0:3], null offen offset:147456
	buffer_store_b32 v53, v0, s[0:3], null offen offset:147520
	buffer_store_b32 v45, v0, s[0:3], null offen offset:147584
	buffer_store_b32 v37, v0, s[0:3], null offen offset:147648
	buffer_store_b32 v62, v0, s[0:3], null offen offset:151552
	buffer_store_b32 v54, v0, s[0:3], null offen offset:151616
	buffer_store_b32 v46, v0, s[0:3], null offen offset:151680
	buffer_store_b32 v38, v0, s[0:3], null offen offset:151744
	buffer_store_b32 v63, v0, s[0:3], null offen offset:155648
	buffer_store_b32 v55, v0, s[0:3], null offen offset:155712
	buffer_store_b32 v47, v0, s[0:3], null offen offset:155776
	buffer_store_b32 v39, v0, s[0:3], null offen offset:155840
	buffer_store_b32 v64, v0, s[0:3], null offen offset:159744
	buffer_store_b32 v56, v0, s[0:3], null offen offset:159808
	buffer_store_b32 v48, v0, s[0:3], null offen offset:159872
	buffer_store_b32 v40, v0, s[0:3], null offen offset:159936
	s_clause 0x1f
	buffer_store_b32 v25, v0, s[0:3], null offen offset:196608
	buffer_store_b32 v17, v0, s[0:3], null offen offset:196672
	buffer_store_b32 v9, v0, s[0:3], null offen offset:196736
	buffer_store_b32 v1, v0, s[0:3], null offen offset:196800
	buffer_store_b32 v26, v0, s[0:3], null offen offset:200704
	buffer_store_b32 v18, v0, s[0:3], null offen offset:200768
	buffer_store_b32 v10, v0, s[0:3], null offen offset:200832
	buffer_store_b32 v2, v0, s[0:3], null offen offset:200896
	buffer_store_b32 v27, v0, s[0:3], null offen offset:204800
	buffer_store_b32 v19, v0, s[0:3], null offen offset:204864
	buffer_store_b32 v11, v0, s[0:3], null offen offset:204928
	buffer_store_b32 v3, v0, s[0:3], null offen offset:204992
	buffer_store_b32 v28, v0, s[0:3], null offen offset:208896
	buffer_store_b32 v20, v0, s[0:3], null offen offset:208960
	buffer_store_b32 v12, v0, s[0:3], null offen offset:209024
	buffer_store_b32 v4, v0, s[0:3], null offen offset:209088
	buffer_store_b32 v29, v0, s[0:3], null offen offset:212992
	buffer_store_b32 v21, v0, s[0:3], null offen offset:213056
	buffer_store_b32 v13, v0, s[0:3], null offen offset:213120
	buffer_store_b32 v5, v0, s[0:3], null offen offset:213184
	buffer_store_b32 v30, v0, s[0:3], null offen offset:217088
	buffer_store_b32 v22, v0, s[0:3], null offen offset:217152
	buffer_store_b32 v14, v0, s[0:3], null offen offset:217216
	buffer_store_b32 v6, v0, s[0:3], null offen offset:217280
	buffer_store_b32 v31, v0, s[0:3], null offen offset:221184
	buffer_store_b32 v23, v0, s[0:3], null offen offset:221248
	buffer_store_b32 v15, v0, s[0:3], null offen offset:221312
	buffer_store_b32 v7, v0, s[0:3], null offen offset:221376
	buffer_store_b32 v32, v0, s[0:3], null offen offset:225280
	buffer_store_b32 v24, v0, s[0:3], null offen offset:225344
	buffer_store_b32 v16, v0, s[0:3], null offen offset:225408
	buffer_store_b32 v8, v0, s[0:3], null offen offset:225472
	s_barrier_signal -1
	s_barrier_wait -1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end1:
	.size	matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32, .Lfunc_end1-matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32
		.amdhsa_group_segment_fixed_size 51712
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 255
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 58
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.num_vgpr, 255
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.num_agpr, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.numbered_sgpr, 20
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.num_named_barrier, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.private_seg_size, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.uses_vcc, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.uses_flat_scratch, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.has_dyn_sized_stack, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.has_recursion, 0
	.set .Lmatmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.has_indirect_call, 0
	.globl	matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32
	.p2align	8
	.type	matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32,@function
matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32:
	v_lshlrev_b32_e32 v4, 3, v0
	s_load_b128 s[12:15], s[0:1], 0x0
	v_or_b32_e32 v1, 0x100, v0
	v_lshlrev_b32_e32 v5, 4, v0
	s_lshl_b32 s3, ttmp9, 7
	v_and_b32_e32 v183, 0x78, v4
	v_or_b32_e32 v2, 0x200, v0
	v_lshrrev_b32_e32 v180, 3, v1
	s_lshl_b32 s2, ttmp9, 3
	v_or_b32_e32 v3, 0x300, v0
	s_and_b32 s3, s3, 0xf80
	s_and_b32 s2, s2, 0xf00
	v_and_b32_e32 v184, 0x70, v5
	v_or_b32_e32 v5, s3, v183
	v_lshrrev_b32_e32 v179, 3, v0
	v_lshrrev_b32_e32 v181, 3, v2
	v_lshrrev_b32_e32 v192, 4, v1
	v_or_b32_e32 v1, s2, v180
	v_lshrrev_b32_e32 v182, 3, v3
	v_lshrrev_b32_e32 v191, 4, v0
	v_lshlrev_b32_e32 v197, 1, v5
	v_or_b32_e32 v4, s2, v179
	v_lshl_or_b32 v196, v1, 13, v184
	v_or_b32_e32 v1, s2, v181
	v_lshrrev_b32_e32 v194, 4, v2
	v_or_b32_e32 v2, s2, v182
	v_lshrrev_b32_e32 v195, 4, v3
	v_lshl_or_b32 v3, v191, 13, v197
	s_mov_b32 s7, 0x31027000
	s_brev_b32 s6, 64
	s_wait_kmcnt 0x0
	s_and_b32 s15, s15, 0xffff
	v_lshl_or_b32 v193, v4, 13, v184
	v_lshl_or_b32 v4, v192, 13, v197
	s_and_b32 s13, s13, 0xffff
	v_lshl_or_b32 v198, v1, 13, v184
	s_mov_b32 s10, s6
	s_mov_b32 s11, s7
	s_mov_b32 s8, s14
	s_mov_b32 s9, s15
	v_lshl_or_b32 v199, v2, 13, v184
	s_mov_b32 s4, s12
	s_mov_b32 s5, s13
	v_lshl_or_b32 v1, v194, 13, v197
	s_clause 0x1
	buffer_load_b128 v[137:140], v3, s[8:11], null offen
	buffer_load_b128 v[141:144], v4, s[8:11], null offen
	s_clause 0x1
	buffer_load_b128 v[145:148], v198, s[4:7], null offen
	buffer_load_b128 v[149:152], v199, s[4:7], null offen
	buffer_load_b128 v[153:156], v1, s[8:11], null offen
	v_lshl_or_b32 v1, v195, 13, v197
	s_clause 0x5
	buffer_load_b128 v[129:132], v193, s[4:7], null offen
	buffer_load_b128 v[133:136], v196, s[4:7], null offen
	buffer_load_b128 v[157:160], v193, s[4:7], null offen offset:1048576
	buffer_load_b128 v[161:164], v193, s[4:7], null offen offset:1310720
	buffer_load_b128 v[165:168], v193, s[4:7], null offen offset:1572864
	buffer_load_b128 v[169:172], v193, s[4:7], null offen offset:1835008
	buffer_load_b128 v[173:176], v1, s[8:11], null offen
	v_dual_mov_b32 v1, 0 :: v_dual_and_b32 v178, 15, v0
	v_lshrrev_b32_e32 v177, 1, v0
	v_lshlrev_b32_e32 v187, 1, v0
	s_load_b64 s[0:1], s[0:1], 0x10
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v8, v1 :: v_dual_and_b32 v185, 0xcf, v0
	v_dual_mov_b32 v5, v1 :: v_dual_mov_b32 v10, v1
	v_and_b32_e32 v177, 8, v177
	v_and_or_b32 v178, v187, 64, v178
	v_or_b32_e32 v186, 48, v0
	v_dual_mov_b32 v3, v1 :: v_dual_mov_b32 v4, v1
	v_mul_u32_u24_e32 v185, 0x44, v185
	v_mad_u32_u24 v202, 0x88, v179, v184
	v_mad_u32_u24 v203, 0x88, v180, v184
	v_mad_u32_u24 v204, 0x88, v181, v184
	v_mad_u32_u24 v205, 0x88, v182, v184
	v_lshlrev_b32_e32 v179, 1, v183
	v_mul_u32_u24_e32 v181, 0x108, v194
	v_lshlrev_b32_e32 v183, 1, v177
	v_mul_u32_u24_e32 v184, 0x84, v177
	v_lshlrev_b32_e32 v187, 1, v178
	v_mul_u32_u24_e32 v188, 0x108, v191
	v_mul_u32_u24_e32 v186, 0x44, v186
	v_mul_u32_u24_e32 v180, 0x108, v192
	v_mul_u32_u24_e32 v182, 0x108, v195
	v_add3_u32 v212, v181, v179, 0x8800
	v_lshl_add_u32 v181, v184, 1, v187
	v_lshl_add_u32 v185, v185, 1, v183
	v_mov_b32_e32 v2, v1
	v_dual_mov_b32 v6, v1 :: v_dual_mov_b32 v7, v1
	v_dual_mov_b32 v12, v1 :: v_dual_mov_b32 v9, v1
	v_dual_mov_b32 v14, v1 :: v_dual_mov_b32 v11, v1
	v_dual_mov_b32 v16, v1 :: v_dual_mov_b32 v13, v1
	v_dual_mov_b32 v18, v1 :: v_dual_mov_b32 v15, v1
	v_dual_mov_b32 v20, v1 :: v_dual_mov_b32 v17, v1
	v_dual_mov_b32 v22, v1 :: v_dual_mov_b32 v19, v1
	v_dual_mov_b32 v24, v1 :: v_dual_mov_b32 v21, v1
	v_dual_mov_b32 v26, v1 :: v_dual_mov_b32 v23, v1
	v_dual_mov_b32 v28, v1 :: v_dual_mov_b32 v25, v1
	v_dual_mov_b32 v30, v1 :: v_dual_mov_b32 v27, v1
	v_dual_mov_b32 v32, v1 :: v_dual_mov_b32 v29, v1
	v_dual_mov_b32 v34, v1 :: v_dual_mov_b32 v31, v1
	v_dual_mov_b32 v36, v1 :: v_dual_mov_b32 v33, v1
	v_dual_mov_b32 v38, v1 :: v_dual_mov_b32 v35, v1
	v_dual_mov_b32 v40, v1 :: v_dual_mov_b32 v37, v1
	v_dual_mov_b32 v42, v1 :: v_dual_mov_b32 v39, v1
	v_dual_mov_b32 v44, v1 :: v_dual_mov_b32 v41, v1
	v_dual_mov_b32 v46, v1 :: v_dual_mov_b32 v43, v1
	v_dual_mov_b32 v48, v1 :: v_dual_mov_b32 v45, v1
	v_dual_mov_b32 v50, v1 :: v_dual_mov_b32 v47, v1
	v_dual_mov_b32 v52, v1 :: v_dual_mov_b32 v49, v1
	v_dual_mov_b32 v54, v1 :: v_dual_mov_b32 v51, v1
	v_dual_mov_b32 v56, v1 :: v_dual_mov_b32 v53, v1
	v_dual_mov_b32 v58, v1 :: v_dual_mov_b32 v55, v1
	v_dual_mov_b32 v60, v1 :: v_dual_mov_b32 v57, v1
	v_dual_mov_b32 v62, v1 :: v_dual_mov_b32 v59, v1
	v_dual_mov_b32 v64, v1 :: v_dual_mov_b32 v61, v1
	v_dual_mov_b32 v66, v1 :: v_dual_mov_b32 v63, v1
	v_dual_mov_b32 v68, v1 :: v_dual_mov_b32 v65, v1
	v_dual_mov_b32 v70, v1 :: v_dual_mov_b32 v67, v1
	v_dual_mov_b32 v72, v1 :: v_dual_mov_b32 v69, v1
	v_dual_mov_b32 v74, v1 :: v_dual_mov_b32 v71, v1
	v_dual_mov_b32 v76, v1 :: v_dual_mov_b32 v73, v1
	v_dual_mov_b32 v78, v1 :: v_dual_mov_b32 v75, v1
	v_dual_mov_b32 v80, v1 :: v_dual_mov_b32 v77, v1
	v_dual_mov_b32 v82, v1 :: v_dual_mov_b32 v79, v1
	v_dual_mov_b32 v84, v1 :: v_dual_mov_b32 v81, v1
	v_dual_mov_b32 v86, v1 :: v_dual_mov_b32 v83, v1
	v_dual_mov_b32 v88, v1 :: v_dual_mov_b32 v85, v1
	v_dual_mov_b32 v90, v1 :: v_dual_mov_b32 v87, v1
	v_dual_mov_b32 v92, v1 :: v_dual_mov_b32 v89, v1
	v_dual_mov_b32 v94, v1 :: v_dual_mov_b32 v91, v1
	v_dual_mov_b32 v96, v1 :: v_dual_mov_b32 v93, v1
	v_dual_mov_b32 v98, v1 :: v_dual_mov_b32 v95, v1
	v_dual_mov_b32 v100, v1 :: v_dual_mov_b32 v97, v1
	v_dual_mov_b32 v102, v1 :: v_dual_mov_b32 v99, v1
	v_dual_mov_b32 v104, v1 :: v_dual_mov_b32 v101, v1
	v_dual_mov_b32 v106, v1 :: v_dual_mov_b32 v103, v1
	v_dual_mov_b32 v108, v1 :: v_dual_mov_b32 v105, v1
	v_dual_mov_b32 v110, v1 :: v_dual_mov_b32 v107, v1
	v_dual_mov_b32 v112, v1 :: v_dual_mov_b32 v109, v1
	v_dual_mov_b32 v114, v1 :: v_dual_mov_b32 v111, v1
	v_dual_mov_b32 v116, v1 :: v_dual_mov_b32 v113, v1
	v_dual_mov_b32 v118, v1 :: v_dual_mov_b32 v115, v1
	v_dual_mov_b32 v120, v1 :: v_dual_mov_b32 v117, v1
	v_dual_mov_b32 v122, v1 :: v_dual_mov_b32 v119, v1
	v_dual_mov_b32 v124, v1 :: v_dual_mov_b32 v121, v1
	v_dual_mov_b32 v126, v1 :: v_dual_mov_b32 v123, v1
	v_dual_mov_b32 v128, v1 :: v_dual_mov_b32 v125, v1
	v_dual_mov_b32 v127, v1 :: v_dual_add_nc_u32 v206, 0x4400, v202
	v_add_nc_u32_e32 v208, 0x6600, v202
	v_add_nc_u32_e32 v207, 0x5500, v202
	v_add_nc_u32_e32 v209, 0x7700, v202
	v_add3_u32 v210, v188, v179, 0x8800
	v_add3_u32 v211, v180, v179, 0x8800
	v_add3_u32 v213, v182, v179, 0x8800
	v_lshl_add_u32 v180, v186, 1, v183
	v_add_nc_u32_e32 v179, 0x8800, v181
	v_add_nc_u32_e32 v190, 0x9880, v181
	v_add_nc_u32_e32 v201, 0xa900, v181
	v_add_nc_u32_e32 v200, 0xb980, v181
	v_add_nc_u32_e32 v188, 0x880, v185
	v_add_nc_u32_e32 v187, 0x8a0, v185
	v_add_nc_u32_e32 v183, 0x8c0, v185
	v_add_nc_u32_e32 v184, 0x8e0, v185
	v_add_nc_u32_e32 v189, 0x1100, v185
	v_add_nc_u32_e32 v186, 0x1120, v185
	v_add_nc_u32_e32 v181, 0x1140, v185
	v_add_nc_u32_e32 v182, 0x1160, v185
	s_mov_b32 s16, 0
	s_wait_loadcnt 0x6
	ds_store_2addr_b64 v202, v[129:130], v[131:132] offset1:1
	ds_store_2addr_b64 v210, v[137:138], v[139:140] offset1:1
	s_wait_loadcnt 0x5
	ds_store_2addr_b64 v203, v[133:134], v[135:136] offset1:1
	ds_store_2addr_b64 v211, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v204, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v212, v[153:154], v[155:156] offset1:1
	ds_store_2addr_b64 v205, v[149:150], v[151:152] offset1:1
	s_wait_loadcnt 0x4
	ds_store_2addr_b64 v206, v[157:158], v[159:160] offset1:1
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v207, v[161:162], v[163:164] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v208, v[165:166], v[167:168] offset1:1
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v209, v[169:170], v[171:172] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v213, v[173:174], v[175:176] offset1:1
.LBB2_1:
	s_add_co_i32 s12, s16, 4
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s13, s12, 4
	s_lshl_b32 s14, s12, 5
	s_wait_alu depctr_sa_sdst(0)
	v_or_b32_e32 v129, s13, v191
	v_or_b32_e32 v130, s13, v192
	v_or_b32_e32 v131, s13, v194
	v_or_b32_e32 v132, s13, v195
	v_add_nc_u32_e32 v133, s14, v198
	v_add_nc_u32_e32 v134, s14, v199
	v_add_nc_u32_e32 v141, s14, v196
	v_add_nc_u32_e32 v153, s14, v193
	v_lshl_or_b32 v129, v129, 13, v197
	v_lshl_or_b32 v130, v130, 13, v197
	v_lshl_or_b32 v131, v131, 13, v197
	v_lshl_or_b32 v132, v132, 13, v197
	s_clause 0x7
	buffer_load_b128 v[137:140], v133, s[4:7], null offen
	buffer_load_b128 v[133:136], v134, s[4:7], null offen
	buffer_load_b128 v[157:160], v141, s[4:7], null offen
	buffer_load_b128 v[173:176], v153, s[4:7], null offen
	buffer_load_b128 v[141:144], v153, s[4:7], null offen offset:1048576
	buffer_load_b128 v[145:148], v153, s[4:7], null offen offset:1310720
	buffer_load_b128 v[149:152], v153, s[4:7], null offen offset:1572864
	buffer_load_b128 v[153:156], v153, s[4:7], null offen offset:1835008
	s_clause 0x3
	buffer_load_b128 v[169:172], v129, s[8:11], null offen
	buffer_load_b128 v[165:168], v130, s[8:11], null offen
	buffer_load_b128 v[161:164], v131, s[8:11], null offen
	buffer_load_b128 v[129:132], v132, s[8:11], null offen
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_u16_d16 v214, v179
	ds_load_u16_d16 v215, v179 offset:528
	ds_load_u16_d16 v216, v179 offset:1056
	ds_load_u16_d16 v217, v179 offset:1584
	ds_load_2addr_b64 v[222:225], v188 offset1:1
	ds_load_2addr_b64 v[230:233], v180 offset1:1
	ds_load_2addr_b64 v[226:229], v189 offset1:1
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v214, v179 offset:264
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v215, v179 offset:792
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v216, v179 offset:1320
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v217, v179 offset:1848
	ds_load_2addr_b64 v[218:221], v185 offset1:1
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[89:96], v[222:225], v[214:217], v[89:96]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[218:221], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[57:64], v[226:229], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	ds_load_u16_d16 v216, v179 offset:1120
	ds_load_u16_d16 v214, v179 offset:64
	ds_load_u16_d16 v215, v179 offset:592
	ds_load_u16_d16 v217, v179 offset:1648
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v216, v179 offset:1384
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v214, v179 offset:328
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v215, v179 offset:856
	s_wait_dscnt 0x3
	ds_load_u16_d16_hi v217, v179 offset:1912
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[105:112], v[218:221], v[214:217], v[105:112]
	v_wmma_f32_16x16x16_f16 v[73:80], v[222:225], v[214:217], v[73:80]
	v_wmma_f32_16x16x16_f16 v[41:48], v[226:229], v[214:217], v[41:48]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[214:217], v[9:16]
	ds_load_u16_d16 v214, v179 offset:32
	ds_load_u16_d16 v215, v179 offset:560
	ds_load_u16_d16 v234, v179 offset:96
	ds_load_u16_d16 v235, v179 offset:624
	ds_load_u16_d16 v216, v179 offset:1088
	ds_load_u16_d16 v236, v179 offset:1152
	ds_load_u16_d16 v217, v179 offset:1616
	ds_load_u16_d16 v237, v179 offset:1680
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v214, v179 offset:296
	s_wait_dscnt 0x6
	ds_load_u16_d16_hi v234, v179 offset:360
	ds_load_u16_d16_hi v215, v179 offset:824
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v235, v179 offset:888
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v216, v179 offset:1352
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v236, v179 offset:1416
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v217, v179 offset:1880
	s_wait_dscnt 0x7
	ds_load_u16_d16_hi v237, v179 offset:1944
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[113:120], v[218:221], v[214:217], v[113:120]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[218:221], v[234:237], v[97:104]
	v_wmma_f32_16x16x16_f16 v[81:88], v[222:225], v[214:217], v[81:88]
	v_wmma_f32_16x16x16_f16 v[65:72], v[222:225], v[234:237], v[65:72]
	v_wmma_f32_16x16x16_f16 v[49:56], v[226:229], v[214:217], v[49:56]
	v_wmma_f32_16x16x16_f16 v[33:40], v[226:229], v[234:237], v[33:40]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[214:217], v[17:24]
	ds_load_u16_d16 v214, v190
	ds_load_u16_d16 v215, v179 offset:4752
	ds_load_u16_d16 v218, v190 offset:32
	ds_load_u16_d16 v219, v179 offset:4784
	ds_load_u16_d16 v222, v190 offset:64
	ds_load_u16_d16 v223, v179 offset:4816
	ds_load_u16_d16 v226, v190 offset:96
	ds_load_u16_d16 v227, v179 offset:4848
	ds_load_u16_d16 v216, v179 offset:5280
	ds_load_u16_d16 v220, v179 offset:5312
	ds_load_u16_d16 v224, v179 offset:5344
	ds_load_u16_d16 v228, v179 offset:5376
	ds_load_u16_d16 v217, v179 offset:5808
	ds_load_u16_d16 v221, v179 offset:5840
	ds_load_u16_d16 v225, v179 offset:5872
	ds_load_u16_d16 v229, v179 offset:5904
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[234:237], v[1:8]
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v214, v179 offset:4488
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v222, v179 offset:4552
	ds_load_u16_d16_hi v218, v179 offset:4520
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v226, v179 offset:4584
	ds_load_u16_d16_hi v215, v179 offset:5016
	ds_load_u16_d16_hi v219, v179 offset:5048
	ds_load_u16_d16_hi v223, v179 offset:5080
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:5112
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:5544
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:5576
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:5608
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:5640
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:6072
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:6104
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:6136
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:6168
	ds_load_2addr_b64 v[230:233], v185 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[218:221], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[222:225], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[226:229], v[97:104]
	ds_load_2addr_b64 v[230:233], v187 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[214:217], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[218:221], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[222:225], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[226:229], v[65:72]
	ds_load_2addr_b64 v[230:233], v186 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[218:221], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[222:225], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[226:229], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:4 offset1:5
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[218:221], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[222:225], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[226:229], v[1:8]
	ds_load_u16_d16 v214, v201
	ds_load_u16_d16 v215, v179 offset:8976
	ds_load_u16_d16 v218, v201 offset:32
	ds_load_u16_d16 v219, v179 offset:9008
	ds_load_u16_d16 v222, v201 offset:64
	ds_load_u16_d16 v223, v179 offset:9040
	ds_load_u16_d16 v226, v201 offset:96
	ds_load_u16_d16 v227, v179 offset:9072
	ds_load_u16_d16 v216, v179 offset:9504
	ds_load_u16_d16 v224, v179 offset:9568
	ds_load_u16_d16 v220, v179 offset:9536
	ds_load_u16_d16 v228, v179 offset:9600
	ds_load_u16_d16 v217, v179 offset:10032
	ds_load_u16_d16 v225, v179 offset:10096
	ds_load_u16_d16 v221, v179 offset:10064
	ds_load_u16_d16 v229, v179 offset:10128
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v214, v179 offset:8712
	s_wait_dscnt 0xe
	ds_load_u16_d16_hi v218, v179 offset:8744
	s_wait_dscnt 0xd
	ds_load_u16_d16_hi v222, v179 offset:8776
	s_wait_dscnt 0xc
	ds_load_u16_d16_hi v226, v179 offset:8808
	ds_load_u16_d16_hi v215, v179 offset:9240
	ds_load_u16_d16_hi v219, v179 offset:9272
	ds_load_u16_d16_hi v223, v179 offset:9304
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:9336
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:9768
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:9832
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:9800
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:9864
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:10296
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:10360
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:10328
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:10392
	ds_load_2addr_b64 v[230:233], v185 offset0:8 offset1:9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[214:217], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[218:221], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[222:225], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[226:229], v[97:104]
	ds_load_2addr_b64 v[230:233], v183 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[214:217], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[218:221], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[222:225], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[226:229], v[65:72]
	ds_load_2addr_b64 v[230:233], v181 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[214:217], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[218:221], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[222:225], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[226:229], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:8 offset1:9
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[214:217], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[218:221], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[222:225], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[226:229], v[1:8]
	ds_load_u16_d16 v217, v179 offset:14352
	ds_load_u16_d16 v214, v200 offset:96
	ds_load_u16_d16 v218, v200
	ds_load_u16_d16 v219, v179 offset:13200
	ds_load_u16_d16 v222, v200 offset:32
	ds_load_u16_d16 v223, v179 offset:13232
	ds_load_u16_d16 v226, v200 offset:64
	ds_load_u16_d16 v227, v179 offset:13264
	ds_load_u16_d16 v215, v179 offset:13296
	ds_load_u16_d16 v220, v179 offset:13728
	ds_load_u16_d16 v224, v179 offset:13760
	ds_load_u16_d16 v228, v179 offset:13792
	ds_load_u16_d16 v216, v179 offset:13824
	ds_load_u16_d16 v221, v179 offset:14256
	ds_load_u16_d16 v225, v179 offset:14288
	ds_load_u16_d16 v229, v179 offset:14320
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v217, v179 offset:14616
	s_wait_dscnt 0xe
	ds_load_u16_d16_hi v218, v179 offset:12936
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v226, v179 offset:13000
	ds_load_u16_d16_hi v222, v179 offset:12968
	ds_load_u16_d16_hi v214, v179 offset:13032
	ds_load_u16_d16_hi v219, v179 offset:13464
	ds_load_u16_d16_hi v223, v179 offset:13496
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v227, v179 offset:13528
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v215, v179 offset:13560
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v220, v179 offset:13992
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v224, v179 offset:14024
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v228, v179 offset:14056
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v216, v179 offset:14088
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v221, v179 offset:14520
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v225, v179 offset:14552
	s_wait_dscnt 0xf
	ds_load_u16_d16_hi v229, v179 offset:14584
	ds_load_2addr_b64 v[230:233], v185 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[121:128], v[230:233], v[218:221], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[230:233], v[222:225], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[230:233], v[226:229], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[230:233], v[214:217], v[97:104]
	ds_load_2addr_b64 v[230:233], v184 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[89:96], v[230:233], v[218:221], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[230:233], v[222:225], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[230:233], v[226:229], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[230:233], v[214:217], v[65:72]
	ds_load_2addr_b64 v[230:233], v182 offset1:1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[57:64], v[230:233], v[218:221], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[230:233], v[222:225], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[230:233], v[226:229], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[230:233], v[214:217], v[33:40]
	ds_load_2addr_b64 v[230:233], v180 offset0:12 offset1:13
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[25:32], v[230:233], v[218:221], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[230:233], v[222:225], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[230:233], v[226:229], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[230:233], v[214:217], v[1:8]
	s_barrier_signal -1
	s_barrier_wait -1
	s_cmp_lt_u32 s16, 0xf8
	s_mov_b32 s16, s12
	s_wait_loadcnt 0x8
	ds_store_2addr_b64 v202, v[173:174], v[175:176] offset1:1
	s_wait_loadcnt 0x3
	ds_store_2addr_b64 v210, v[169:170], v[171:172] offset1:1
	ds_store_2addr_b64 v203, v[157:158], v[159:160] offset1:1
	s_wait_loadcnt 0x2
	ds_store_2addr_b64 v211, v[165:166], v[167:168] offset1:1
	ds_store_2addr_b64 v204, v[137:138], v[139:140] offset1:1
	s_wait_loadcnt 0x1
	ds_store_2addr_b64 v212, v[161:162], v[163:164] offset1:1
	ds_store_2addr_b64 v205, v[133:134], v[135:136] offset1:1
	ds_store_2addr_b64 v206, v[141:142], v[143:144] offset1:1
	ds_store_2addr_b64 v207, v[145:146], v[147:148] offset1:1
	ds_store_2addr_b64 v208, v[149:150], v[151:152] offset1:1
	ds_store_2addr_b64 v209, v[153:154], v[155:156] offset1:1
	s_wait_loadcnt 0x0
	ds_store_2addr_b64 v213, v[129:130], v[131:132] offset1:1
	s_cbranch_scc1 .LBB2_1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v0, 0xc0, v0
	v_or_b32_e32 v178, s3, v178
	s_wait_kmcnt 0x0
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x31027000
	v_or3_b32 v0, s2, v0, v177
	v_lshlrev_b32_e32 v177, 2, v178
	s_brev_b32 s2, 32
	s_delay_alu instid0(VALU_DEP_1)
	v_lshl_or_b32 v0, v0, 14, v177
	s_barrier_wait -1
	ds_load_u16_d16 v137, v179
	ds_load_u16_d16 v129, v179 offset:32
	ds_load_u16_d16 v141, v179 offset:64
	ds_load_u16_d16 v133, v179 offset:96
	ds_load_u16_d16 v138, v179 offset:528
	ds_load_u16_d16 v142, v179 offset:592
	ds_load_u16_d16 v146, v179 offset:8976
	ds_load_u16_d16 v150, v179 offset:9008
	ds_load_u16_d16 v154, v179 offset:9040
	ds_load_u16_d16 v158, v179 offset:9072
	ds_load_u16_d16 v147, v179 offset:9504
	ds_load_u16_d16 v155, v179 offset:9568
	ds_load_u16_d16 v148, v179 offset:10032
	ds_load_u16_d16 v156, v179 offset:10096
	ds_load_u16_d16 v151, v179 offset:9536
	ds_load_u16_d16 v159, v179 offset:9600
	ds_load_u16_d16 v145, v201
	ds_load_u16_d16 v149, v201 offset:32
	ds_load_u16_d16 v153, v201 offset:64
	ds_load_u16_d16 v157, v201 offset:96
	ds_load_u16_d16 v161, v200
	ds_load_u16_d16 v165, v200 offset:32
	ds_load_u16_d16 v169, v200 offset:64
	ds_load_u16_d16 v173, v200 offset:96
	ds_load_u16_d16 v152, v179 offset:10064
	ds_load_u16_d16 v160, v179 offset:10128
	ds_load_u16_d16 v130, v179 offset:560
	ds_load_u16_d16 v134, v179 offset:624
	ds_load_u16_d16 v162, v179 offset:13200
	ds_load_u16_d16 v166, v179 offset:13232
	ds_load_u16_d16 v170, v179 offset:13264
	ds_load_u16_d16 v174, v179 offset:13296
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v146, v179 offset:9240
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v150, v179 offset:9272
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v154, v179 offset:9304
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v158, v179 offset:9336
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v147, v179 offset:9768
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v155, v179 offset:9832
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v151, v179 offset:9800
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v159, v179 offset:9864
	ds_load_u16_d16_hi v148, v179 offset:10296
	ds_load_u16_d16_hi v156, v179 offset:10360
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v161, v179 offset:12936
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v169, v179 offset:13000
	s_wait_dscnt 0x13
	ds_load_u16_d16_hi v152, v179 offset:10328
	s_wait_dscnt 0x13
	ds_load_u16_d16_hi v160, v179 offset:10392
	ds_load_u16_d16_hi v165, v179 offset:12968
	ds_load_u16_d16_hi v173, v179 offset:13032
	ds_load_u16_d16 v139, v179 offset:1056
	ds_load_u16_d16 v143, v179 offset:1120
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v162, v179 offset:13464
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v166, v179 offset:13496
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v170, v179 offset:13528
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v174, v179 offset:13560
	ds_load_u16_d16 v163, v179 offset:13728
	ds_load_u16_d16 v167, v179 offset:13760
	ds_load_u16_d16 v171, v179 offset:13792
	ds_load_u16_d16 v175, v179 offset:13824
	ds_load_u16_d16_hi v137, v179 offset:264
	ds_load_u16_d16_hi v129, v179 offset:296
	ds_load_u16_d16_hi v141, v179 offset:328
	ds_load_u16_d16_hi v133, v179 offset:360
	ds_load_u16_d16_hi v138, v179 offset:792
	ds_load_u16_d16_hi v142, v179 offset:856
	ds_load_u16_d16 v176, v179 offset:14352
	ds_load_u16_d16_hi v130, v179 offset:824
	ds_load_u16_d16_hi v134, v179 offset:888
	ds_load_u16_d16 v131, v179 offset:1088
	ds_load_u16_d16 v135, v179 offset:1152
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v139, v179 offset:1320
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v143, v179 offset:1384
	ds_load_u16_d16 v140, v179 offset:1584
	ds_load_u16_d16 v144, v179 offset:1648
	ds_load_u16_d16 v191, v190
	ds_load_u16_d16 v195, v190 offset:32
	ds_load_u16_d16 v199, v190 offset:64
	ds_load_u16_d16 v203, v190 offset:96
	ds_load_u16_d16 v132, v179 offset:1616
	ds_load_u16_d16 v136, v179 offset:1680
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v163, v179 offset:13992
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v167, v179 offset:14024
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v171, v179 offset:14056
	s_wait_dscnt 0x18
	ds_load_u16_d16_hi v175, v179 offset:14088
	ds_load_u16_d16 v164, v179 offset:14256
	ds_load_u16_d16 v168, v179 offset:14288
	ds_load_u16_d16 v172, v179 offset:14320
	ds_load_u16_d16 v192, v179 offset:4752
	ds_load_u16_d16 v196, v179 offset:4784
	ds_load_u16_d16 v200, v179 offset:4816
	ds_load_u16_d16 v204, v179 offset:4848
	s_wait_dscnt 0x19
	ds_load_u16_d16_hi v176, v179 offset:14616
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v131, v179 offset:1352
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v135, v179 offset:1416
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v140, v179 offset:1848
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v144, v179 offset:1912
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v191, v179 offset:4488
	s_wait_dscnt 0x14
	ds_load_u16_d16_hi v199, v179 offset:4552
	ds_load_u16_d16 v193, v179 offset:5280
	ds_load_u16_d16 v197, v179 offset:5312
	ds_load_u16_d16 v201, v179 offset:5344
	ds_load_u16_d16 v205, v179 offset:5376
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v132, v179 offset:1880
	s_wait_dscnt 0x17
	ds_load_u16_d16_hi v136, v179 offset:1944
	ds_load_u16_d16_hi v195, v179 offset:4520
	ds_load_u16_d16_hi v203, v179 offset:4584
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v164, v179 offset:14520
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v168, v179 offset:14552
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v172, v179 offset:14584
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v192, v179 offset:5016
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v196, v179 offset:5048
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v200, v179 offset:5080
	s_wait_dscnt 0x15
	ds_load_u16_d16_hi v204, v179 offset:5112
	ds_load_2addr_b64 v[207:210], v185 offset1:1
	ds_load_2addr_b64 v[211:214], v188 offset1:1
	ds_load_2addr_b64 v[215:218], v189 offset1:1
	ds_load_2addr_b64 v[219:222], v180 offset1:1
	ds_load_2addr_b64 v[223:226], v185 offset0:4 offset1:5
	ds_load_2addr_b64 v[227:230], v185 offset0:8 offset1:9
	ds_load_2addr_b64 v[231:234], v185 offset0:12 offset1:13
	ds_load_2addr_b64 v[187:190], v187 offset1:1
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v193, v179 offset:5544
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v197, v179 offset:5576
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v201, v179 offset:5608
	s_wait_dscnt 0x16
	ds_load_u16_d16_hi v205, v179 offset:5640
	ds_load_u16_d16 v194, v179 offset:5808
	ds_load_u16_d16 v198, v179 offset:5840
	ds_load_u16_d16 v202, v179 offset:5872
	ds_load_u16_d16 v206, v179 offset:5904
	ds_load_2addr_b64 v[235:238], v183 offset1:1
	ds_load_2addr_b64 v[239:242], v184 offset1:1
	ds_load_2addr_b64 v[183:186], v186 offset1:1
	ds_load_2addr_b64 v[243:246], v180 offset0:4 offset1:5
	s_wait_dscnt 0x13
	v_wmma_f32_16x16x16_f16 v[121:128], v[207:210], v[137:140], v[121:128]
	s_wait_dscnt 0x12
	v_wmma_f32_16x16x16_f16 v[89:96], v[211:214], v[137:140], v[89:96]
	s_wait_dscnt 0x11
	v_wmma_f32_16x16x16_f16 v[57:64], v[215:218], v[137:140], v[57:64]
	s_wait_dscnt 0x10
	v_wmma_f32_16x16x16_f16 v[25:32], v[219:222], v[137:140], v[25:32]
	ds_load_2addr_b64 v[137:140], v181 offset1:1
	ds_load_2addr_b64 v[247:250], v182 offset1:1
	v_wmma_f32_16x16x16_f16 v[105:112], v[207:210], v[141:144], v[105:112]
	v_wmma_f32_16x16x16_f16 v[73:80], v[211:214], v[141:144], v[73:80]
	v_wmma_f32_16x16x16_f16 v[41:48], v[215:218], v[141:144], v[41:48]
	v_wmma_f32_16x16x16_f16 v[9:16], v[219:222], v[141:144], v[9:16]
	ds_load_2addr_b64 v[141:144], v180 offset0:8 offset1:9
	ds_load_2addr_b64 v[251:254], v180 offset0:12 offset1:13
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v194, v179 offset:6072
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v198, v179 offset:6104
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v202, v179 offset:6136
	s_wait_dscnt 0xb
	ds_load_u16_d16_hi v206, v179 offset:6168
	ds_load_u16_d16_hi v145, v179 offset:8712
	ds_load_u16_d16_hi v149, v179 offset:8744
	ds_load_u16_d16_hi v153, v179 offset:8776
	ds_load_u16_d16_hi v157, v179 offset:8808
	v_wmma_f32_16x16x16_f16 v[113:120], v[207:210], v[129:132], v[113:120]
	v_wmma_f32_16x16x16_f16 v[97:104], v[207:210], v[133:136], v[97:104]
	v_wmma_f32_16x16x16_f16 v[81:88], v[211:214], v[129:132], v[81:88]
	v_wmma_f32_16x16x16_f16 v[65:72], v[211:214], v[133:136], v[65:72]
	v_wmma_f32_16x16x16_f16 v[49:56], v[215:218], v[129:132], v[49:56]
	v_wmma_f32_16x16x16_f16 v[33:40], v[215:218], v[133:136], v[33:40]
	v_wmma_f32_16x16x16_f16 v[17:24], v[219:222], v[129:132], v[17:24]
	v_wmma_f32_16x16x16_f16 v[1:8], v[219:222], v[133:136], v[1:8]
	s_wait_dscnt 0x7
	v_wmma_f32_16x16x16_f16 v[121:128], v[223:226], v[191:194], v[121:128]
	s_wait_dscnt 0x6
	v_wmma_f32_16x16x16_f16 v[113:120], v[223:226], v[195:198], v[113:120]
	s_wait_dscnt 0x5
	v_wmma_f32_16x16x16_f16 v[105:112], v[223:226], v[199:202], v[105:112]
	s_wait_dscnt 0x4
	v_wmma_f32_16x16x16_f16 v[97:104], v[223:226], v[203:206], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[187:190], v[191:194], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[187:190], v[195:198], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[187:190], v[199:202], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[187:190], v[203:206], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[183:186], v[191:194], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[183:186], v[195:198], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[183:186], v[199:202], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[183:186], v[203:206], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[243:246], v[191:194], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[243:246], v[195:198], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[243:246], v[199:202], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[243:246], v[203:206], v[1:8]
	s_wait_dscnt 0x3
	v_wmma_f32_16x16x16_f16 v[121:128], v[227:230], v[145:148], v[121:128]
	s_wait_dscnt 0x2
	v_wmma_f32_16x16x16_f16 v[113:120], v[227:230], v[149:152], v[113:120]
	s_wait_dscnt 0x1
	v_wmma_f32_16x16x16_f16 v[105:112], v[227:230], v[153:156], v[105:112]
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[97:104], v[227:230], v[157:160], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[235:238], v[145:148], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[235:238], v[149:152], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[235:238], v[153:156], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[235:238], v[157:160], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[137:140], v[145:148], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[137:140], v[149:152], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[137:140], v[153:156], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[137:140], v[157:160], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[141:144], v[145:148], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[141:144], v[149:152], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[141:144], v[153:156], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[141:144], v[157:160], v[1:8]
	v_wmma_f32_16x16x16_f16 v[121:128], v[231:234], v[161:164], v[121:128]
	v_wmma_f32_16x16x16_f16 v[113:120], v[231:234], v[165:168], v[113:120]
	v_wmma_f32_16x16x16_f16 v[105:112], v[231:234], v[169:172], v[105:112]
	v_wmma_f32_16x16x16_f16 v[97:104], v[231:234], v[173:176], v[97:104]
	v_wmma_f32_16x16x16_f16 v[89:96], v[239:242], v[161:164], v[89:96]
	v_wmma_f32_16x16x16_f16 v[81:88], v[239:242], v[165:168], v[81:88]
	v_wmma_f32_16x16x16_f16 v[73:80], v[239:242], v[169:172], v[73:80]
	v_wmma_f32_16x16x16_f16 v[65:72], v[239:242], v[173:176], v[65:72]
	v_wmma_f32_16x16x16_f16 v[57:64], v[247:250], v[161:164], v[57:64]
	v_wmma_f32_16x16x16_f16 v[49:56], v[247:250], v[165:168], v[49:56]
	v_wmma_f32_16x16x16_f16 v[41:48], v[247:250], v[169:172], v[41:48]
	v_wmma_f32_16x16x16_f16 v[33:40], v[247:250], v[173:176], v[33:40]
	v_wmma_f32_16x16x16_f16 v[25:32], v[251:254], v[161:164], v[25:32]
	v_wmma_f32_16x16x16_f16 v[17:24], v[251:254], v[165:168], v[17:24]
	v_wmma_f32_16x16x16_f16 v[9:16], v[251:254], v[169:172], v[9:16]
	v_wmma_f32_16x16x16_f16 v[1:8], v[251:254], v[173:176], v[1:8]
	s_clause 0x1f
	buffer_store_b32 v121, v0, s[0:3], null offen
	buffer_store_b32 v113, v0, s[0:3], null offen offset:64
	buffer_store_b32 v105, v0, s[0:3], null offen offset:128
	buffer_store_b32 v97, v0, s[0:3], null offen offset:192
	buffer_store_b32 v122, v0, s[0:3], null offen offset:16384
	buffer_store_b32 v114, v0, s[0:3], null offen offset:16448
	buffer_store_b32 v106, v0, s[0:3], null offen offset:16512
	buffer_store_b32 v98, v0, s[0:3], null offen offset:16576
	buffer_store_b32 v123, v0, s[0:3], null offen offset:32768
	buffer_store_b32 v115, v0, s[0:3], null offen offset:32832
	buffer_store_b32 v107, v0, s[0:3], null offen offset:32896
	buffer_store_b32 v99, v0, s[0:3], null offen offset:32960
	buffer_store_b32 v124, v0, s[0:3], null offen offset:49152
	buffer_store_b32 v116, v0, s[0:3], null offen offset:49216
	buffer_store_b32 v108, v0, s[0:3], null offen offset:49280
	buffer_store_b32 v100, v0, s[0:3], null offen offset:49344
	buffer_store_b32 v125, v0, s[0:3], null offen offset:65536
	buffer_store_b32 v117, v0, s[0:3], null offen offset:65600
	buffer_store_b32 v109, v0, s[0:3], null offen offset:65664
	buffer_store_b32 v101, v0, s[0:3], null offen offset:65728
	buffer_store_b32 v126, v0, s[0:3], null offen offset:81920
	buffer_store_b32 v118, v0, s[0:3], null offen offset:81984
	buffer_store_b32 v110, v0, s[0:3], null offen offset:82048
	buffer_store_b32 v102, v0, s[0:3], null offen offset:82112
	buffer_store_b32 v127, v0, s[0:3], null offen offset:98304
	buffer_store_b32 v119, v0, s[0:3], null offen offset:98368
	buffer_store_b32 v111, v0, s[0:3], null offen offset:98432
	buffer_store_b32 v103, v0, s[0:3], null offen offset:98496
	buffer_store_b32 v128, v0, s[0:3], null offen offset:114688
	buffer_store_b32 v120, v0, s[0:3], null offen offset:114752
	buffer_store_b32 v112, v0, s[0:3], null offen offset:114816
	buffer_store_b32 v104, v0, s[0:3], null offen offset:114880
	s_clause 0x1f
	buffer_store_b32 v89, v0, s[0:3], null offen offset:262144
	buffer_store_b32 v81, v0, s[0:3], null offen offset:262208
	buffer_store_b32 v73, v0, s[0:3], null offen offset:262272
	buffer_store_b32 v65, v0, s[0:3], null offen offset:262336
	buffer_store_b32 v90, v0, s[0:3], null offen offset:278528
	buffer_store_b32 v82, v0, s[0:3], null offen offset:278592
	buffer_store_b32 v74, v0, s[0:3], null offen offset:278656
	buffer_store_b32 v66, v0, s[0:3], null offen offset:278720
	buffer_store_b32 v91, v0, s[0:3], null offen offset:294912
	buffer_store_b32 v83, v0, s[0:3], null offen offset:294976
	buffer_store_b32 v75, v0, s[0:3], null offen offset:295040
	buffer_store_b32 v67, v0, s[0:3], null offen offset:295104
	buffer_store_b32 v92, v0, s[0:3], null offen offset:311296
	buffer_store_b32 v84, v0, s[0:3], null offen offset:311360
	buffer_store_b32 v76, v0, s[0:3], null offen offset:311424
	buffer_store_b32 v68, v0, s[0:3], null offen offset:311488
	buffer_store_b32 v93, v0, s[0:3], null offen offset:327680
	buffer_store_b32 v85, v0, s[0:3], null offen offset:327744
	buffer_store_b32 v77, v0, s[0:3], null offen offset:327808
	buffer_store_b32 v69, v0, s[0:3], null offen offset:327872
	buffer_store_b32 v94, v0, s[0:3], null offen offset:344064
	buffer_store_b32 v86, v0, s[0:3], null offen offset:344128
	buffer_store_b32 v78, v0, s[0:3], null offen offset:344192
	buffer_store_b32 v70, v0, s[0:3], null offen offset:344256
	buffer_store_b32 v95, v0, s[0:3], null offen offset:360448
	buffer_store_b32 v87, v0, s[0:3], null offen offset:360512
	buffer_store_b32 v79, v0, s[0:3], null offen offset:360576
	buffer_store_b32 v71, v0, s[0:3], null offen offset:360640
	buffer_store_b32 v96, v0, s[0:3], null offen offset:376832
	buffer_store_b32 v88, v0, s[0:3], null offen offset:376896
	buffer_store_b32 v80, v0, s[0:3], null offen offset:376960
	buffer_store_b32 v72, v0, s[0:3], null offen offset:377024
	s_clause 0x1f
	buffer_store_b32 v57, v0, s[0:3], null offen offset:524288
	buffer_store_b32 v49, v0, s[0:3], null offen offset:524352
	buffer_store_b32 v41, v0, s[0:3], null offen offset:524416
	buffer_store_b32 v33, v0, s[0:3], null offen offset:524480
	buffer_store_b32 v58, v0, s[0:3], null offen offset:540672
	buffer_store_b32 v50, v0, s[0:3], null offen offset:540736
	buffer_store_b32 v42, v0, s[0:3], null offen offset:540800
	buffer_store_b32 v34, v0, s[0:3], null offen offset:540864
	buffer_store_b32 v59, v0, s[0:3], null offen offset:557056
	buffer_store_b32 v51, v0, s[0:3], null offen offset:557120
	buffer_store_b32 v43, v0, s[0:3], null offen offset:557184
	buffer_store_b32 v35, v0, s[0:3], null offen offset:557248
	buffer_store_b32 v60, v0, s[0:3], null offen offset:573440
	buffer_store_b32 v52, v0, s[0:3], null offen offset:573504
	buffer_store_b32 v44, v0, s[0:3], null offen offset:573568
	buffer_store_b32 v36, v0, s[0:3], null offen offset:573632
	buffer_store_b32 v61, v0, s[0:3], null offen offset:589824
	buffer_store_b32 v53, v0, s[0:3], null offen offset:589888
	buffer_store_b32 v45, v0, s[0:3], null offen offset:589952
	buffer_store_b32 v37, v0, s[0:3], null offen offset:590016
	buffer_store_b32 v62, v0, s[0:3], null offen offset:606208
	buffer_store_b32 v54, v0, s[0:3], null offen offset:606272
	buffer_store_b32 v46, v0, s[0:3], null offen offset:606336
	buffer_store_b32 v38, v0, s[0:3], null offen offset:606400
	buffer_store_b32 v63, v0, s[0:3], null offen offset:622592
	buffer_store_b32 v55, v0, s[0:3], null offen offset:622656
	buffer_store_b32 v47, v0, s[0:3], null offen offset:622720
	buffer_store_b32 v39, v0, s[0:3], null offen offset:622784
	buffer_store_b32 v64, v0, s[0:3], null offen offset:638976
	buffer_store_b32 v56, v0, s[0:3], null offen offset:639040
	buffer_store_b32 v48, v0, s[0:3], null offen offset:639104
	buffer_store_b32 v40, v0, s[0:3], null offen offset:639168
	s_clause 0x1f
	buffer_store_b32 v25, v0, s[0:3], null offen offset:786432
	buffer_store_b32 v17, v0, s[0:3], null offen offset:786496
	buffer_store_b32 v9, v0, s[0:3], null offen offset:786560
	buffer_store_b32 v1, v0, s[0:3], null offen offset:786624
	buffer_store_b32 v26, v0, s[0:3], null offen offset:802816
	buffer_store_b32 v18, v0, s[0:3], null offen offset:802880
	buffer_store_b32 v10, v0, s[0:3], null offen offset:802944
	buffer_store_b32 v2, v0, s[0:3], null offen offset:803008
	buffer_store_b32 v27, v0, s[0:3], null offen offset:819200
	buffer_store_b32 v19, v0, s[0:3], null offen offset:819264
	buffer_store_b32 v11, v0, s[0:3], null offen offset:819328
	buffer_store_b32 v3, v0, s[0:3], null offen offset:819392
	buffer_store_b32 v28, v0, s[0:3], null offen offset:835584
	buffer_store_b32 v20, v0, s[0:3], null offen offset:835648
	buffer_store_b32 v12, v0, s[0:3], null offen offset:835712
	buffer_store_b32 v4, v0, s[0:3], null offen offset:835776
	buffer_store_b32 v29, v0, s[0:3], null offen offset:851968
	buffer_store_b32 v21, v0, s[0:3], null offen offset:852032
	buffer_store_b32 v13, v0, s[0:3], null offen offset:852096
	buffer_store_b32 v5, v0, s[0:3], null offen offset:852160
	buffer_store_b32 v30, v0, s[0:3], null offen offset:868352
	buffer_store_b32 v22, v0, s[0:3], null offen offset:868416
	buffer_store_b32 v14, v0, s[0:3], null offen offset:868480
	buffer_store_b32 v6, v0, s[0:3], null offen offset:868544
	buffer_store_b32 v31, v0, s[0:3], null offen offset:884736
	buffer_store_b32 v23, v0, s[0:3], null offen offset:884800
	buffer_store_b32 v15, v0, s[0:3], null offen offset:884864
	buffer_store_b32 v7, v0, s[0:3], null offen offset:884928
	buffer_store_b32 v32, v0, s[0:3], null offen offset:901120
	buffer_store_b32 v24, v0, s[0:3], null offen offset:901184
	buffer_store_b32 v16, v0, s[0:3], null offen offset:901248
	buffer_store_b32 v8, v0, s[0:3], null offen offset:901312
	s_barrier_signal -1
	s_barrier_wait -1
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end2:
	.size	matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32, .Lfunc_end2-matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32
		.amdhsa_group_segment_fixed_size 51712
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 255
		.amdhsa_next_free_sgpr 17
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 57
		.amdhsa_round_robin_scheduling 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text

	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.num_vgpr, 255
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.num_agpr, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.numbered_sgpr, 17
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.num_named_barrier, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.private_seg_size, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.uses_vcc, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.uses_flat_scratch, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.has_dyn_sized_stack, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.has_recursion, 0
	.set .Lmatmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.has_indirect_call, 0
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 51712
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         matmul_2048x2048x2048_dispatch_0_matmul_2048x2048x2048_f16xf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     255
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 51712
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     20
    .sgpr_spill_count: 0
    .symbol:         matmul_2048x1024x4096_dispatch_0_matmul_2048x1024x4096_f16xf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     255
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 51712
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 256
    .name:           matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     17
    .sgpr_spill_count: 0
    .symbol:         matmul_4096x4096x4096_dispatch_0_matmul_4096x4096x4096_f16xf16xf32.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     255
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
