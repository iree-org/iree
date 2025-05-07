// RUN: iree-opt --split-input-file --iree-codegen-test-flatten-vector-extract-insert %s | FileCheck %s


// before : extract scalar from rank 3
// after  : extract scalar from rank 1
// CHECK: func.func @extract_scalar_from_rank3(%[[ARG0:.*]]: vector<6x5x4xf32>) -> f32
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract %[[SHAPE_CAST]][50] : f32 from vector<120xf32>
// CHECK: return %[[EXTRACT]] : f32
func.func @extract_scalar_from_rank3(%arg0: vector<6x5x4xf32>) -> f32 {
  %0 = vector.extract %arg0[2, 2, 2] : f32 from vector<6x5x4xf32>
  return %0 : f32
}

// -----

// before  : extract scalar from rank 1
// after   : extract scalar from rank 1
// i.e. unchanged.
// CHECK: func.func @extract_scalar_from_rank1(%[[ARG0:.*]]: vector<1xf32>) -> f32
// CHECK: %[[EXTRACT:.*]] = vector.extract %[[ARG0]][] : f32 from vector<1xf32>
// CHECK: return %[[EXTRACT]] : f32
func.func @extract_scalar_from_rank1(%arg0: vector<1xf32>) -> f32 {
  %0 = vector.extract %arg0[] : f32 from vector<1xf32>
  return %0 : f32
}

// -----

// before  : extract rank 1 from rank 3
// after   : extract_strided_slice rank 1 from rank 1
// CHECK: func.func @extract_rank1_from_rank3(%[[ARG0:.*]]: vector<6x5x4xf32>) -> vector<4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[SHAPE_CAST]] {offsets = [48], sizes = [4], strides = [1]} : vector<120xf32> to vector<4xf32>
// CHECK: return %[[EXTRACT]] : vector<4xf32>
func.func @extract_rank1_from_rank3(%arg0: vector<6x5x4xf32>) -> vector<4xf32> {
  %0 = vector.extract %arg0[2, 2] : vector<4xf32> from vector<6x5x4xf32>
  return %0 : vector<4xf32>
}

// -----

// before  : extract rank 2 from rank 3
// after   : extract_strided_slice rank 1 from rank 1
// CHECK: func.func @extract_rank2_from_rank3(%[[ARG0:.*]]: vector<6x5x4xf32>) -> vector<5x4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[SHAPE_CAST]] {offsets = [40], sizes = [20], strides = [1]} : vector<120xf32> to vector<20xf32>
// CHECK: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[EXTRACT]] : vector<20xf32> to vector<5x4xf32>
func.func @extract_rank2_from_rank3(%arg0: vector<6x5x4xf32>) -> vector<5x4xf32> {
  %0 = vector.extract %arg0[2] : vector<5x4xf32> from vector<6x5x4xf32>
  return %0 : vector<5x4xf32>
}

// -----

// vector.extract which preserves the number of elements is converted to a shape_cast. Note
// that this pass does no folding, so the 'identity' shape_cast is retained.
// CHECK: func.func @extract_identity(%[[ARG0:.*]]: vector<6x5x4xf32>) -> vector<6x5x4xf32>
// CHECK-NOT: vector.extract
// CHECK: vector.shape_cast
func.func @extract_identity(%arg0: vector<6x5x4xf32>) -> vector<6x5x4xf32> {
  %0 = vector.extract %arg0[] : vector<6x5x4xf32> from vector<6x5x4xf32>
  return %0 : vector<6x5x4xf32>
}


// -----

// before  :  insert rank 2 into rank 3
// after   :  insert_strided_slice 1 into 1
// CHECK: func.func @insert_to_rank_one_r2(%[[ARG0:.*]]: vector<6x5x4xf32>, %[[ARG1:.*]]: vector<5x4xf32>) -> vector<6x5x4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[ARG1]] : vector<5x4xf32> to vector<20xf32>
// CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[SHAPE_CAST1]], %[[SHAPE_CAST]] {offsets = [40], strides = [1]} : vector<20xf32> into vector<120xf32>
// CHECK: %[[SHAPE_CAST2:.*]] = vector.shape_cast %[[INSERT]] : vector<120xf32> to vector<6x5x4xf32>
// CHECK: return %[[SHAPE_CAST2]] : vector<6x5x4xf32>
func.func @insert_to_rank_one_r2(%arg0: vector<6x5x4xf32>, %arg1: vector<5x4xf32>) -> vector<6x5x4xf32> {
  %0 = vector.insert %arg1, %arg0 [2] : vector<5x4xf32> into vector<6x5x4xf32>
  return %0 : vector<6x5x4xf32>
}

// -----

// vector.insert %183, %182 [0, 1, 0, 0] : vector<4xf32> into vector<8x2x1x1x4xf32>

// CHECK: vector.insert_strided_slice {{.*}}offsets = [4]
func.func @insert_rank1_into_rank4(%arg0: vector<8x2x1x1x4xf32>, %arg1: vector<4xf32>) -> vector<8x2x1x1x4xf32> {
  %0 = vector.insert %arg1, %arg0 [0, 1, 0, 0] : vector<4xf32> into vector<8x2x1x1x4xf32>
  return %0 : vector<8x2x1x1x4xf32>
}

// -----

// before  : insert 1 into 3
// after   : insert_strided_slice 1 into 1
// CHECK: func.func @insert_to_rank_one_r1(%[[ARG0:.*]]: vector<6x5x4xf32>, %[[ARG1:.*]]: vector<4xf32>) -> vector<6x5x4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[INSERT:.*]] = vector.insert_strided_slice %[[ARG1]], %[[SHAPE_CAST]] {offsets = [48], strides = [1]} : vector<4xf32> into vector<120xf32>
// CHECK: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[INSERT]] : vector<120xf32> to vector<6x5x4xf32>
// CHECK: return %[[SHAPE_CAST1]] : vector<6x5x4xf32>
func.func @insert_to_rank_one_r1(%arg0: vector<6x5x4xf32>, %arg1: vector<4xf32>) -> vector<6x5x4xf32> {
  %0 = vector.insert %arg1, %arg0 [2, 2] : vector<4xf32> into vector<6x5x4xf32>
  return %0 : vector<6x5x4xf32>
}

// -----

// before  : insert 0 into 3
// after   : insert 0 into 1
// CHECK: func.func @insert_to_rank_one_r0(%[[ARG0:.*]]: vector<6x5x4xf32>, %[[ARG1:.*]]: f32) -> vector<6x5x4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[INSERT:.*]] = vector.insert %[[ARG1]], %[[SHAPE_CAST]] [50] : f32 into vector<120xf32>
// CHECK: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[INSERT]] : vector<120xf32> to vector<6x5x4xf32>
// CHECK: return %[[SHAPE_CAST1]] : vector<6x5x4xf32>
func.func @insert_to_rank_one_r0(%arg0: vector<6x5x4xf32>, %arg1: f32) -> vector<6x5x4xf32> {
  %0 = vector.insert %arg1, %arg0 [2, 2, 2] : f32 into vector<6x5x4xf32>
  return %0 : vector<6x5x4xf32>
}

// -----

// before  : insert 0 into 1
// after   : insert 0 into 1
// CHECK: func.func @insert_to_rank_one_r0_r1(%[[ARG0:.*]]: vector<4xf32>, %[[ARG1:.*]]: f32) -> vector<4xf32>
// CHECK: %[[INSERT:.*]] = vector.insert %[[ARG1]], %[[ARG0]] [2] : f32 into vector<4xf32>
// CHECK: return %[[INSERT]] : vector<4xf32>
func.func @insert_to_rank_one_r0_r1(%arg0: vector<4xf32>, %arg1: f32) -> vector<4xf32> {
  %0 = vector.insert %arg1, %arg0 [2] : f32 into vector<4xf32>
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func.func @insert_strided_slice_r2_to_r2_v0
// CHECK: vector.insert_strided_slice {{.*}}offsets = [5], strides = [1]} : vector<10xf32> into vector<20xf32>
func.func @insert_strided_slice_r2_to_r2_v0(%arg0 : vector<2x5xf32>, %arg1 : vector<4x5xf32>) -> vector<4x5xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1, 0], strides = [1, 1]} : vector<2x5xf32> into vector<4x5xf32>
  return %0 : vector<4x5xf32>
}

// -----

// CHECK-LABEL: func.func @insert_strided_slice_r2_to_r2_v1
// CHECK: vector.shuffle{{.*}}[0, 1, 2, 3, 4, 20, 21, 22, 23, 9, 24, 25, 26, 27, 14, 15, 16, 17, 18, 19] : vector<20xf32>, vector<8xf32>
func.func @insert_strided_slice_r2_to_r2_v1(%arg0 : vector<2x4xf32>, %arg1 : vector<4x5xf32>) -> vector<4x5xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1, 0], strides = [1, 1]} : vector<2x4xf32> into vector<4x5xf32>
  return %0 : vector<4x5xf32>
}

// -----

// CHECK-LABEL: func.func @insert_strided_slice_r2_to_r3
// CHECK: vector.insert_strided_slice {{.*}}offsets = [45]{{.*}}vector<10xf32> into vector<60xf32>
func.func @insert_strided_slice_r2_to_r3(%arg0 : vector<2x5xf32>, %arg1 : vector<3x4x5xf32>) -> vector<3x4x5xf32> {
  %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [2, 1, 0], strides = [1, 1]} : vector<2x5xf32> into vector<3x4x5xf32>
  return %0 : vector<3x4x5xf32>
}

// -----

// CHECK-LABEL: func.func @extract_strided_slice_r2_to_r2_v0
// CHECK: vector.extract_strided_slice {{.*}}offsets = [12], sizes = [12], strides = [1]} : vector<36xf32> to vector<12xf32>
func.func @extract_strided_slice_r2_to_r2_v0(%arg0 : vector<6x6xf32>) -> vector<2x6xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2, 0], sizes = [2, 6], strides = [1, 1]} : vector<6x6xf32> to vector<2x6xf32>
  return %0 : vector<2x6xf32>
}

// -----


// CHECK-LABEL: func.func @extract_strided_slice_r2_to_r2_v1
// CHECK: vector.shuffle{{.*}} [12, 13, 14, 15, 18, 19, 20, 21] : vector<36xf32>, vector<36xf32>
func.func @extract_strided_slice_r2_to_r2_v1(%arg0 : vector<6x6xf32>) -> vector<2x4xf32> {
  %0 = vector.extract_strided_slice %arg0 {offsets = [2, 0], sizes = [2, 4], strides = [1, 1]} : vector<6x6xf32> to vector<2x4xf32>
  return %0 : vector<2x4xf32>
}

// -----

func.func @foo(%arg0 : vector<8xf16>)  -> vector<64xf16> {
  %poison8x8 = ub.poison : vector<8x8xf16>
  %inserted = vector.insert %arg0, %poison8x8 [0] : vector<8xf16> into vector<8x8xf16>
  %downed = vector.shape_cast %inserted : vector<8x8xf16> to vector<64xf16>
  return %downed : vector<64xf16>
}

// ----- 

func.func @bar(%615 : vector<8x8xf16>) -> vector<4x1xf16> {
  %616 = vector.extract_strided_slice %615 {offsets = [0, 0], sizes = [4, 1], strides = [1, 1]} : vector<8x8xf16> to vector<4x1xf16>
  return %616 : vector<4x1xf16>
}


// // func.func @extract_strided_slice_r2_to_r2_v0(%arg0 : vector<6x6xf32>) -> vector<2x6xf32> {
// //   %0 = vector.extract_strided_slice %arg0 {offsets = [2, 0], sizes = [2, 6], strides = [1, 1]} : vector<6x6xf32> to vector<2x6xf32>
// //   return %0 : vector<2x6xf32>
// // }
//
// // func.func @extract_strided_slice_r2_to_r2_v0(%arg0 : vector<6x6x6x6xf32>) -> vector<2x6x2x6xf32> {
// //   %0 = vector.extract_strided_slice %arg0 {offsets = [1, 0, 1, 0], sizes = [2, 6, 2, 6], strides = [1, 1, 1, 1]} : vector<6x6x6x6xf32> to vector<2x6x2x6xf32>
// //   return %0 : vector<2x6x2x6xf32>
// // }
//
// func.func @extract_strided_slice_r2_to_r2_v0(%arg0 : vector<6x6xf32>) -> vector<2x6xf32> {
//   %0 = vector.extract_strided_slice %arg0 {offsets = [1, 0], sizes = [2, 6], strides = [1, 1]} : vector<6x6xf32> to vector<2x6xf32>
//   return %0 : vector<2x6xf32>
// }
//
// func.func @extract_strided_slice_r2_to_r2_v1(%arg0 : vector<3x4x5xf32>) -> vector<2x3x4xf32> {
//   %0 = vector.extract_strided_slice %arg0 {offsets = [1, 1, 1], sizes = [2, 3, 4], strides = [1, 1, 1]} : vector<3x4x5xf32> to vector<2x3x4xf32>
//   return %0 : vector<2x3x4xf32>
// }
//
// func.func @insert_strided_slice(%arg0 : vector<1x1x5xi8>, %arg2 : vector<3x4x5xi8>) -> vector<3x4x5xi8> {
//   %0 = vector.insert_strided_slice %arg0, %arg2 {offsets = [1, 1, 0], sizes = [1, 1, 5], strides = [1, 1, 1]} : vector<1x1x5xi8> into vector<3x4x5xi8>
//   return %0 : vector<3x4x5xi8>
// }
//
// func.func @insert_strided_slice_1(%arg0 : vector<3x1x2xi8>, %arg2 : vector<3x4x2xi8>) -> vector<3x4x2xi8> {
//   %0 = vector.insert_strided_slice %arg0, %arg2 {offsets = [0, 0, 0], sizes = [1, 1, 2], strides = [1, 1, 1]} : vector<3x1x2xi8> into vector<3x4x2xi8>
//   return %0 : vector<3x4x2xi8>
// }
//
// func.func @insert_to_rank_one_r0(%arg0: vector<6x5x4xf32>, %arg1: f32) -> vector<6x5x4xf32> {
//   %0 = vector.insert %arg1, %arg0 [2, 2, 2] : f32 into vector<6x5x4xf32>
//   return %0 : vector<6x5x4xf32>
// }
//
// func.func @insert_strided_slice_r2_to_r2_v1(%arg0 : vector<2x4xf32>, %arg1 : vector<4x5xf32>) -> vector<4x5xf32> {
//   %0 = vector.insert_strided_slice %arg0, %arg1 {offsets = [1, 0], strides = [1, 1]} : vector<2x4xf32> into vector<4x5xf32>
//   return %0 : vector<4x5xf32>
// }
