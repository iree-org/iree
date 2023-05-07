// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/partitioner/GSPMDPipeline.h"

#include "mhlo/transforms/passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/transforms/hlo_constant_splitter.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/call_inliner.h"
#include "xla/service/conditional_canonicalizer.h"
#include "xla/service/conditional_simplifier.h"
#include "xla/service/gather_expander.h"
#include "xla/service/gather_simplifier.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/hlo_constant_folding.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/hlo_pass_fix.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/reshape_mover.h"
#include "xla/service/scatter_expander.h"
#include "xla/service/scatter_simplifier.h"
#include "xla/service/sharding_propagation.h"
#include "xla/service/sharding_remover.h"
#include "xla/service/sort_simplifier.h"
#include "xla/service/spmd/collective_permute_motion.h"
#include "xla/service/spmd/spmd_partitioner.h"
#include "xla/service/spmd/stateful_rng_spmd_partitioner.h"
#include "xla/service/topk_rewriter.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/service/while_loop_simplifier.h"
#include "xla/service/zero_sized_hlo_elimination.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

using namespace mlir;

namespace xla {

bool ConvIsLowerable(HloInstruction *conv) {
  return gpu::GpuConvRewriter::ConvIsLowerable(conv);
}

Status RunSPMDOptimizer(HloModule *hlo_module, int64_t num_partitions) {
  AlgebraicSimplifierOptions layout_insensitive_algsimp_opts({},
                                                             ConvIsLowerable);

  // GPU only supports canonical convolutions.
  layout_insensitive_algsimp_opts.set_supports_non_canonical_dots(false);

  // "slow" minmax means we propagate nan.
  layout_insensitive_algsimp_opts.set_minmax_propagate_nan(
      /*!debug_options.xla_gpu_enable_fast_min_max()*/ false);

  // Always simplify reduce(transpose(x)) and reduce(reshape(x)), even when
  // the transpose/reshape has multiple users.  This helps int8 models, which
  // tend to have lots of transpose+reshape's (converting between NCHW and
  // NCHW_VECT_C).  Without this, those reshape+transposes can get materialized
  // out, which is really bad for perf.
  layout_insensitive_algsimp_opts
      .set_unconditionally_simplify_reduce_of_transpose_or_reshape(true);

  HloPassPipeline pre_spmd_pipeline("pre-spmd-partitioner");
  // Run some IR cleanup passes before running the SPMD partitioning
  // passes.
  pre_spmd_pipeline.AddPass<CallInliner>();
  pre_spmd_pipeline.AddPass<ZeroSizedHloElimination>();
  pre_spmd_pipeline.AddPass<ConditionalCanonicalizer>();
  // The SPMD partitioner would mess up the sort+slice structure, so we need
  // to rewrite Topk before that happens.
  pre_spmd_pipeline.AddPass<TopkRewriter>(
      [](const HloSortInstruction *, int64_t) { return true; });
  TF_RETURN_IF_ERROR(pre_spmd_pipeline.Run(hlo_module).status());

  if (num_partitions > 1) {
    // TODO: Populate the config and enable check to match gpu_compiler?
    // if (!hlo_module->config().use_spmd_partitioning()) {
    //   return InvalidArgument(
    //       "num_partitions=%d but SPMD partitioning not enabled.",
    //       num_partitions);
    // }
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    HloPassPipeline &spmd_simplify =
        spmd_pipeline.AddPass<HloPassFix<HloPassPipeline>>("spmd-simplify");

    spmd_simplify.AddPass<AlgebraicSimplifier>(layout_insensitive_algsimp_opts);

    spmd_simplify.AddPass<SortSimplifier>();
    spmd_simplify.AddPass<TupleSimplifier>();
    spmd_simplify.AddPass<ScatterSimplifier>();
    spmd_simplify.AddPass<ScatterExpander>(
        ScatterExpander::kEliminateSimpleScatters);
    spmd_simplify.AddPass<GatherSimplifier>();
    spmd_simplify.AddPass<GatherExpander>(
        GatherExpander::kEliminateSimpleGathers);
    spmd_simplify.AddPass<WhileLoopConstantSinking>();
    spmd_simplify.AddPass<WhileLoopSimplifier>();

    spmd_simplify.AddPass<ReshapeMover>();
    spmd_simplify.AddPass<HloConstantFolding>();
    spmd_simplify.AddPass<ConditionalSimplifier>();
    spmd_simplify.AddPass<HloDCE>();

    spmd_pipeline.AddPass<HloConstantSplitter>();
    spmd_pipeline.AddPass<ShardingPropagation>(
        /*is_spmd=*/true, /*propagate_metadata=*/false,
        hlo_module->config().allow_spmd_sharding_propagation_to_output());
    spmd_pipeline.AddPass<spmd::StatefulRngSpmdPartitioner>(
        num_partitions, hlo_module->config().replica_count());
    spmd_pipeline.AddPass<CollectivePermuteMotion>();

    TF_RETURN_IF_ERROR(spmd_pipeline.Run(hlo_module).status());
  } else {
    HloPassPipeline sharding_removal_pipeline("sharding-removal");
    // Remove redundant sharding ops when partition_count == 1.
    sharding_removal_pipeline.AddPass<ShardingRemover>();
    sharding_removal_pipeline.AddPass<HloDCE>();
    TF_RETURN_IF_ERROR(sharding_removal_pipeline.Run(hlo_module).status());
  }
  return OkStatus();
}

}  // namespace xla

namespace openxla::partitioner {

namespace {

class RunGSPMDPartitionerPass
    : public PassWrapper<RunGSPMDPartitionerPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "openxla-partitioner-gspmd"; }
  void runOnOperation() override {
    // Convert to HLO.
    xla::HloProto hlo_proto;
    auto status = ConvertMlirHloToHlo(getOperation(), &hlo_proto,
                                      /*use_tuple_args=*/false,
                                      /*return_tuple=*/false);
    if (!status.ok()) {
      getOperation()->emitError()
          << "failed to convert mhlo to hlo: " << status.ToString();
      return signalPassFailure();
    }

    // Run the optimizer.
    auto moduleOr = runHLOOptimizer(hlo_proto);
    if (!moduleOr.ok()) {
      getOperation()->emitError()
          << "failed to run hlo optimizer: " << status.ToString();
      return signalPassFailure();
    }
    auto hloModule = std::move(moduleOr).value();

    // When converting back, the HLO is inlined into the MLIR module. So
    // first, we remove all children from the MLIR module first.
    for (auto &childOp : llvm::make_early_inc_range(
             getOperation().getBody()->without_terminator())) {
      childOp.erase();
    }
    status = xla::ConvertHloToMlirHlo(getOperation(), hloModule.get());
    if (!status.ok()) {
      getOperation()->emitError()
          << "failed to convert hlo to mhlo: " << status.ToString();
      return signalPassFailure();
    }
  }

  xla::StatusOr<std::unique_ptr<xla::HloModule>> runHLOOptimizer(
      xla::HloProto hlo_proto) {
    xla::DebugOptions debugOptions;
    TF_ASSIGN_OR_RETURN(xla::HloModuleConfig hlo_module_config,
                        xla::HloModule::CreateModuleConfigFromProto(
                            hlo_proto.hlo_module(), debugOptions));
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::HloModule> hlo_module,
                        xla::HloModule::CreateFromProto(hlo_proto.hlo_module(),
                                                        hlo_module_config));
    // TODO: Get the number of partitions from somewhere intelligible.
    TF_RETURN_IF_ERROR(
        RunSPMDOptimizer(hlo_module.get(), /*num_partitions=*/2));

    return hlo_module;
  }
};

}  // namespace

// Builds a pipeline which runs the GSPMD partitioner.
void buildGSPMDPipeline(mlir::PassManager &passManager) {
  // To MHLO.
  passManager.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());

  passManager.addPass(std::make_unique<RunGSPMDPartitionerPass>());

  // And back to stablehlo.
  passManager.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

}  // namespace openxla::partitioner
