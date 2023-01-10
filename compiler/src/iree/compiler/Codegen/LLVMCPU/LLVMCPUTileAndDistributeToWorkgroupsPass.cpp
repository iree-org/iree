namespace mlir {
namespace iree_compiler {

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUTileAndDistributeToWorkgroupsPass() {
  return createTileAndDistributeToWorkgroupsPass(
      populateLLVMCPUDispatchWorkgroupCountPatterns);
}

}  // namespace iree_compiler
}  // namespace mlir
