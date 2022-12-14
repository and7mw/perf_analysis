#include <cstddef>
#include <array>

#include <immintrin.h>

void conv_nhwc_simd(const float *input,
                    const float *weight,
                    float *output,
                    const std::array<size_t, 4>& inShape,
                    const std::array<size_t, 4>& wShape,
                    const std::array<size_t, 4>& outShape) {
    const size_t vlen = 8;

    auto kernel = [&](const float* inDataB,
                      float* outDataB,
                      const size_t offset,
                      const size_t oc,
                      const size_t oh,
                      const size_t ow) {
        __m256 in_vec, w_vec;
        __m256 res_vec = _mm256_set1_ps(0.0f);

        for (size_t kh = 0; kh < wShape[2]; kh++) {
            for (size_t kw = 0; kw < wShape[3]; kw++) {
                for (size_t k_ic = 0; k_ic < wShape[1]; k_ic+=vlen) {
                    in_vec = _mm256_loadu_ps(inDataB + offset + kh * inShape[1] * inShape[3] + kw * inShape[1] + k_ic);
                    w_vec = _mm256_loadu_ps(weight + kh * wShape[1] * wShape[3] + kw * wShape[1] + k_ic);
                    
                    res_vec = _mm256_fmadd_ps(in_vec, w_vec, res_vec);
                }
            }
        }
        float vec_dump[vlen];
        _mm256_storeu_ps(vec_dump, res_vec);
        float result = 0.0f;
        for (size_t i = 0; i < vlen; i++) {
            result += vec_dump[i];
        }

        outDataB[oc * outShape[2] * outShape[3] + oh * outShape[3] + ow] = result;
    };
    
    const size_t inBatchStride = inShape[1] * inShape[2] * inShape[3];
    const size_t outBatchStride = outShape[1] * outShape[2] * outShape[3];
    for (size_t on = 0; on < outShape[0]; on++) {
        for (size_t oc = 0; oc < outShape[1]; oc++) {
            for (size_t oh = 0; oh < outShape[2]; oh++) {
                for (size_t ow = 0; ow < outShape[3]; ow++) {
                    const size_t offset = oh * inShape[1] * inShape[3] + ow * inShape[1];

                    kernel(input + on * inBatchStride, output + on * outBatchStride,
                           offset, oc, oh, ow);
                }
            }
        }
    }
}
