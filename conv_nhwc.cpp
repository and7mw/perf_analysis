#include <cstddef>
#include <array>

// input: nhwc
// weight: nhwc
// output: nchw
void conv_nhwc(const float *input,
               const float *weight,
               float *output,
               const std::array<size_t, 4>& inShape,
               const std::array<size_t, 4>& wShape,
               const std::array<size_t, 4>& outShape) {
    auto kernel = [&](const float* inDataB,
                      float* outDataB,
                      const size_t offset,
                      const size_t oc,
                      const size_t oh,
                      const size_t ow) {
        float result = 0.0f;
        for (size_t kh = 0; kh < wShape[2]; kh++) {
            for (size_t kw = 0; kw < wShape[3]; kw++) {
                for (size_t k_ic = 0; k_ic < wShape[1]; k_ic++) {
                    const float inVal = inDataB[offset + kh * inShape[1] * inShape[3] + kw * inShape[1] + k_ic];
                    const float wVal = weight[kh * wShape[1] * wShape[3] + kw * wShape[1] + k_ic];
                    
                    result += inVal * wVal;
                }
            }
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
