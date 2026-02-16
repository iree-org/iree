// Simple test to verify Gemmini dialect loads
module {
  func.func @test_gemmini_dialect(%arg0: memref<16x16xi8>, %arg1: memref<16x16xi8>, %arg2: memref<16x16xi8>) {
    // Test basic Gemmini operations
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    
    // Test ConfigLd operation
    %scale = arith.constant 1.0 : f32
    gemmini.config_ld %c1 scale(%scale)
    
    // Test Mvin operation  
    gemmini.mvin %arg0 -> %c0
    
    // Test Flush operation
    gemmini.flush %c0
    
    return
  }
}
