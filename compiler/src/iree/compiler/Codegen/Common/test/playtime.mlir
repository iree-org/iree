func.func @bar(%615 : vector<8x8xf16>) -> vector<4x1xf16> {
  %616 = vector.extract_strided_slice %615 {offsets = [0, 0], sizes = [4, 1], strides = [1, 1]} : vector<8x8xf16> to vector<4x1xf16>
  return %616 : vector<4x1xf16>
}
