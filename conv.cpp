#include <iostream>
#include <array>
#include <vector>

using shape = std::array<size_t, 4>;

// input shape: [N, C_IN, H, W]
// weight shape: [C_OUT, C_IN, H, W]
// output shape: [N, C_OUT, H, W]

void conv_nchw(const float *input,
               const float *weight,
               float *output,
               const shape& inShape,
               const shape& wShape,
               const shape& outShape) {
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

void accuracy_test();

int main() {
    accuracy_test();

    // const size_t N = 1;
    // const size_t C_IN = 32;
    // const size_t C_OUT = 32;

    // const size_t IH = 64; const size_t IW = 64;

    // const size_t KH = 3;
    // const size_t KW = 3;

    // const size_t OH = IH - KH + 1;
    // const size_t OW = IW - KW + 1;

    // shape inShape{N, C_IN, IH, IW};
    // shape wShape{C_OUT, C_IN, KH, KW};
    // shape outShape{N, C_OUT, OH, OW};

    // std::vector<float> input(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    // std::vector<float> weight(wShape[0] * wShape[1] * wShape[2] * wShape[3]);
    // std::vector<float> output(outShape[0] * outShape[1] * outShape[2] * outShape[3]);

    // conv_nchw(input.data(), weight.data(), output.data(), inShape, wShape, outShape);

    return 0;
}

void accuracy_test() {
    const size_t N = 1;
    const size_t C_IN = 8;
    const size_t C_OUT = 1;

    const size_t IH = 4; const size_t IW = 4;

    const size_t KH = 2;
    const size_t KW = 2;

    const size_t OH = IH - KH + 1;
    const size_t OW = IW - KW + 1;

    shape inShape{N, C_IN, IH, IW};
    shape wShape{C_OUT, C_IN, KH, KW};
    shape outShape{N, C_OUT, OH, OW};

    // per each channel
    //
    // [ 1, 2, 3, 1
    //   2, 1, 1, 4
    //   3, 2, 3, 1
    //   1, 1, 1, 1 ]
    const std::vector<float> inPerCh{1, 2, 3, 1, 2, 1, 1, 4, 3, 2, 3, 1, 1, 1, 1, 1};
    std::vector<float> input(inShape[0] * inShape[1] * inShape[2] * inShape[3]);

    // per each channel
    //
    // [ 2, 3
    //   1, 2 ]
    const std::vector<float> wPerCh{2, 3, 1, 2};
    std::vector<float> weight(wShape[0] * wShape[1] * wShape[2] * wShape[3]);

    std::vector<float> output(outShape[0] * outShape[1] * outShape[2] * outShape[3]);
    std::vector<float> outRef{96.0f, 128.0f, 144.0f, 112.0f, 104.0f, 152.0f, 120.0f, 128.0f,  96.0f};

    for (size_t i = 0; i < (inShape[0] * inShape[1]); i++) {
        std::copy(inPerCh.begin(), inPerCh.end(), input.begin() + i * (inShape[2] * inShape[3]));
        std::copy(wPerCh.begin(), wPerCh.end(), weight.begin() + i * (wShape[2] * wShape[3]));
    }

    conv_nchw(input.data(), weight.data(), output.data(), inShape, wShape, outShape);

    for (size_t i = 0; i < outRef.size(); i++) {
        if (std::abs(outRef[i] - output[i]) > 0.0001f) {
            throw std::runtime_error("Accuracy error!");
        }
    }
}
