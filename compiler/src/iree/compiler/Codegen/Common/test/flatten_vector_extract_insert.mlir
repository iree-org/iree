// RUN: iree-opt --split-input-file --iree-codegen-test-flatten-vector-extract-insert %s | FileCheck %s


// before : extract 3 -> 0
// after  : extract 1 -> 0
// CHECK: func.func @extract_to_rank_one_r0(%[[ARG0:.*]]: vector<6x5x4xf32>) -> f32
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract %[[SHAPE_CAST]][50] : f32 from vector<120xf32>
// CHECK: return %[[EXTRACT]] : f32
func.func @extract_to_rank_one_r0(%arg0: vector<6x5x4xf32>) -> f32 {
  %0 = vector.extract %arg0[2, 2, 2] : f32 from vector<6x5x4xf32>
  return %0 : f32
}

// -----

// before  : extract 1 -> 0
// after   : extract 1 -> 0
// CHECK: func.func @extract_to_rank_one_r1_r0(%[[ARG0:.*]]: vector<1xf32>) -> f32
// CHECK: %[[EXTRACT:.*]] = vector.extract %[[ARG0]][] : f32 from vector<1xf32>
// CHECK: return %[[EXTRACT]] : f32
func.func @extract_to_rank_one_r1_r0(%arg0: vector<1xf32>) -> f32 {
  %0 = vector.extract %arg0[] : f32 from vector<1xf32>
  return %0 : f32
}

// -----

// before  : extract 3 -> 1
// after   : extract_strided_slice 1 -> 1
// CHECK: func.func @extract_to_rank_one_r1(%[[ARG0:.*]]: vector<6x5x4xf32>) -> vector<4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[SHAPE_CAST]] {offsets = [48], sizes = [4], strides = [1]} : vector<120xf32> to vector<4xf32>
// CHECK: return %[[EXTRACT]] : vector<4xf32>
func.func @extract_to_rank_one_r1(%arg0: vector<6x5x4xf32>) -> vector<4xf32> {
  %0 = vector.extract %arg0[2, 2] : vector<4xf32> from vector<6x5x4xf32>
  return %0 : vector<4xf32>
}

// -----

// before  : extract 3 -> 2
// after   : extract_strided_slice 1 -> 1
// CHECK: func.func @extract_to_rank_one_r2(%[[ARG0:.*]]: vector<6x5x4xf32>) -> vector<5x4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[SHAPE_CAST]] {offsets = [40], sizes = [20], strides = [1]} : vector<120xf32> to vector<20xf32>
// CHECK: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[EXTRACT]] : vector<20xf32> to vector<5x4xf32>
func.func @extract_to_rank_one_r2(%arg0: vector<6x5x4xf32>) -> vector<5x4xf32> {
  %0 = vector.extract %arg0[2] : vector<5x4xf32> from vector<6x5x4xf32>
  return %0 : vector<5x4xf32>
}

// -----

// before  : extract 3 -> 3
// after   : extract_strided_slice 1 -> 1
// CHECK: func.func @extract_to_rank_one_r3(%[[ARG0:.*]]: vector<6x5x4xf32>) -> vector<6x5x4xf32>
// CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG0]] : vector<6x5x4xf32> to vector<120xf32>
// CHECK: %[[EXTRACT:.*]] = vector.extract_strided_slice %[[SHAPE_CAST]] {offsets = [0], sizes = [120], strides = [1]} : vector<120xf32> to vector<120xf32>
// CHECK: %[[SHAPE_CAST1:.*]] = vector.shape_cast %[[EXTRACT]] : vector<120xf32> to vector<6x5x4xf32>
func.func @extract_to_rank_one_r3(%arg0: vector<6x5x4xf32>) -> vector<6x5x4xf32> {
  %0 = vector.extract %arg0[] : vector<6x5x4xf32> from vector<6x5x4xf32>
  return %0 : vector<6x5x4xf32>
}



// -----

// before  :  insert 2 into 3
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
