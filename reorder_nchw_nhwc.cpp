#include <array>

void reorder_nchw_nhwc(const float* src,
                       float* dst,
                       const std::array<size_t, 4>& shape) {
    const size_t batchStride = shape[1] * shape[2] * shape[3];

    const size_t N = shape[0]; const size_t C = shape[1];
    const size_t H = shape[2]; const size_t W = shape[3];

    const size_t srcChStride = H * W;

    for (size_t b = 0; b < N; b++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t h = 0; h < H; h++) {
                size_t src_off = b * batchStride + c * srcChStride + h * W;
                size_t dst_off = b * batchStride + h * W * C + c;

                for (size_t w = 0; w < W; w++) {
                    dst[dst_off] = src[src_off];
                    src_off++;
                    dst_off += C;
                }
            }
        }
    }
}
