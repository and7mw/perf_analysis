#include <iostream>
#include <array>
#include <vector>

#include <chrono>

using shape = std::array<size_t, 4>;

// input shape: [N, C_IN, H, W]
// weight shape: [C_OUT, C_IN, H, W]
// output shape: [N, C_OUT, H, W]

void conv_nchw(const float *input,
               const float *weight,
               float *output,
               const shape& inShape,
               const shape& wShape,
               const shape& outShape);

void conv_nhwc(const float *input,
               const float *weight,
               float *output,
               const shape& inShape,
               const shape& wShape,
               const shape& outShape);

void conv_nhwc_simd(const float *input,
                    const float *weight,
                    float *output,
                    const std::array<size_t, 4>& inShape,
                    const std::array<size_t, 4>& wShape,
                    const std::array<size_t, 4>& outShape);

void reorder_nchw_nhwc(const float* src,
                       float* dst,
                       const std::array<size_t, 4>& shape);

struct params {
    size_t N;
    size_t C_IN;
    size_t C_OUT;
    size_t IH;
    size_t IW;
    size_t KH;
    size_t KW;
    size_t ITER_NUM;

    bool accuracy = false;
    std::vector<float> input;
    std::vector<float> weight;
    std::vector<float> output;
};

params accuracy_params = {
    .N = 1,
    .C_IN = 8,
    .C_OUT = 1,
    .IH = 4,
    .IW = 4,
    .KH = 2,
    .KW = 2,
    .ITER_NUM = 1,

    .accuracy = true,
    // per each channel
    //
    // [ 1, 2, 3, 1
    //   2, 1, 1, 4
    //   3, 2, 3, 1
    //   1, 1, 1, 1 ]
    .input = std::vector<float>{1, 2, 3, 1, 2, 1, 1, 4, 3, 2, 3, 1, 1, 1, 1, 1},
    // per each channel
    //
    // [ 2, 3
    //   1, 2 ]
    .weight = std::vector<float>{2, 3, 1, 2},
    .output = std::vector<float>{96.0f, 128.0f, 144.0f, 112.0f, 104.0f, 152.0f, 120.0f, 128.0f,  96.0f}
};

params perf_params = {
    .N = 1,
    .C_IN = 64,
    .C_OUT = 32,
    .IH = 128,
    .IW = 128,
    .KH = 3,
    .KW = 3,
    .ITER_NUM = 10,
};

int main() {
    const bool accuracy = false;

    params runParams;
    if (accuracy) {
        runParams = accuracy_params;
    } else {
        runParams = perf_params;
    }

    const size_t ITER_NUM = runParams.ITER_NUM;

    const size_t N = runParams.N;
    const size_t C_IN = runParams.C_IN;
    const size_t C_OUT = runParams.C_OUT;

    const size_t IH = runParams.IH; const size_t IW = runParams.IW;

    const size_t KH = runParams.KH;
    const size_t KW = runParams.KW;

    const size_t OH = IH - KH + 1;
    const size_t OW = IW - KW + 1;

    const shape inShape{N, C_IN, IH, IW};
    const shape wShape{C_OUT, C_IN, KH, KW};
    const shape outShape{N, C_OUT, OH, OW};

    std::vector<float> inputNCHW(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    std::vector<float> inputNHWC(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    std::vector<float> weightNCHW(wShape[0] * wShape[1] * wShape[2] * wShape[3]);
    std::vector<float> output(outShape[0] * outShape[1] * outShape[2] * outShape[3]);

    std::vector<float> outRef = runParams.output;

    if (accuracy) {
        for (size_t i = 0; i < (inShape[0] * inShape[1]); i++) {
            std::copy(runParams.input.begin(), runParams.input.end(), inputNCHW.begin() + i * (inShape[2] * inShape[3]));
            std::copy(runParams.weight.begin(), runParams.weight.end(), weightNCHW.begin() + i * (wShape[2] * wShape[3]));
        }
    }

    std::vector<float> weight;
#if defined(SIMD_IMPL) || defined(NHWC_IMPL)
    weight.resize(wShape[0] * wShape[1] * wShape[2] * wShape[3]);
    reorder_nchw_nhwc(weightNCHW.data(), weight.data(), wShape);
#else
    weight = std::move(weightNCHW);
#endif

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < ITER_NUM; i++) {
        if (!accuracy) {
            std::fill(inputNCHW.begin(), inputNCHW.end(), i + 1);
        }

    #if defined(SIMD_IMPL)
        reorder_nchw_nhwc(inputNCHW.data(), inputNHWC.data(), inShape);
        conv_nhwc_simd(inputNHWC.data(), weight.data(), output.data(), inShape, wShape, outShape);
    #elif defined(NHWC_IMPL)
        reorder_nchw_nhwc(inputNCHW.data(), inputNHWC.data(), inShape);
        conv_nhwc(inputNHWC.data(), weight.data(), output.data(), inShape, wShape, outShape);
    #else
        conv_nchw(inputNCHW.data(), weight.data(), output.data(), inShape, wShape, outShape);
    #endif

        if (accuracy) {
            for (size_t i = 0; i < outRef.size(); i++) {
                if (std::abs(outRef[i] - output[i]) > 0.0001f) {
                    throw std::runtime_error("Accuracy error!");
                }
            }
            std::cout << "Accuracy OK!" << std::endl;
        }
    }

    end = std::chrono::high_resolution_clock::now();
    const size_t time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << ITER_NUM << " iterations, took " << time << " ms" << std::endl;

    return 0;
}
