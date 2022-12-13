#include <cstddef>
#include <array>

void conv_nchw(const float *input,
               const float *weight,
               float *output,
               const std::array<size_t, 4>& inShape,
               const std::array<size_t, 4>& wShape,
               const std::array<size_t, 4>& outShape) {
    auto kernel = [&](const float* inDataB,
                      float* outDataB,
                      const size_t offset_in_each_ch,
                      const size_t oc,
                      const size_t oh,
                      const size_t ow) {
        const size_t inChStride = inShape[2] * inShape[3];
        const size_t wInChStride = wShape[2] * wShape[3];
        const size_t wOutChStride = wShape[1] * wShape[2] * wShape[3];

        float result = 0.0f;
        for (size_t k_ic = 0; k_ic < wShape[1]; k_ic++) {
            for (size_t kh = 0; kh < wShape[2]; kh++) {
                for (size_t kw = 0; kw < wShape[3]; kw++) {
                    // std::cout << "k_ic: " << k_ic << ", "
                    //           << "kh: " << kh << ", "
                    //           << "kw: " << kw << std::endl;

                    const float inVal = inDataB[k_ic * inChStride + offset_in_each_ch + kh * inShape[3] + kw];
                    const float wVal = weight[oc * wOutChStride + k_ic * wInChStride + kh * wShape[3] + kw];

                    result += inVal * wVal;
                }
            }
        }
        // std::cout << "result: " << result << std::endl;
        outDataB[oc * outShape[2] * outShape[3] + oh * outShape[3] + ow] = result;
    };
    
    const size_t inBatchStride = inShape[1] * inShape[2] * inShape[3];
    const size_t outBatchStride = outShape[1] * outShape[2] * outShape[3];
    for (size_t on = 0; on < outShape[0]; on++) {
        for (size_t oc = 0; oc < outShape[1]; oc++) {
            for (size_t oh = 0; oh < outShape[2]; oh++) {
                for (size_t ow = 0; ow < outShape[3]; ow++) {
                    // std::cout << "on: " << on << ", "
                    //           << "oc: " << oc << ", "
                    //           << "oh: " << oh << ", "
                    //           << "ow: " << ow << std::endl;

                    const size_t offset_in_each_ch = oh * inShape[3] + ow;

                    kernel(input + on * inBatchStride, output + on * outBatchStride,
                           offset_in_each_ch, oc, oh, ow);
                }
            }
        }
    }
}
