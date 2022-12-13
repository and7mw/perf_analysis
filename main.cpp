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

void reorder_nchw_nhwc(const float* src,
                       float* dst,
                       const std::array<size_t, 4>& shape);

void accuracy_test();

int main() {
    // accuracy_test();

    const size_t ITER_NUM = 2;

    const size_t N = 1;
    const size_t C_IN = 64;
    const size_t C_OUT = 32;

    const size_t IH = 512; const size_t IW = 512;

    const size_t KH = 3;
    const size_t KW = 3;

    const size_t OH = IH - KH + 1;
    const size_t OW = IW - KW + 1;

    const shape inShape{N, C_IN, IH, IW};
    const shape wShape{C_OUT, C_IN, KH, KW};
    const shape outShape{N, C_OUT, OH, OW};

    std::vector<float> input(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    std::vector<float> weight(wShape[0] * wShape[1] * wShape[2] * wShape[3]);
    std::vector<float> output(outShape[0] * outShape[1] * outShape[2] * outShape[3]);

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < ITER_NUM; i++) {
        std::fill(input.begin(), input.end(), i + 1);
        conv_nchw(input.data(), weight.data(), output.data(), inShape, wShape, outShape);
    }
    end = std::chrono::high_resolution_clock::now();
    const size_t time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << ITER_NUM << " iterations, took " << time << " ms" << std::endl;

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
    std::cout << "Accuracy OK!" << std::endl;
}
