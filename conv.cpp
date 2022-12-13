#include <iostream>

// [N, C_IN, Z, Y, X]
// [C_OUT, C_IN, Z, Y, X]
// [N, C_OUT, Z, Y, X]

// input: [N, C, H, W]
// output: [N, KN, H - KH + 1, W - KW + 1]
void conv_nchw(const float *input,
               const float *weight,
               float *output,
               const size_t N,
               const size_t C,
               const size_t H,
               const size_t W,
               const size_t KN,
               const size_t KC,
               const size_t KH,
               const size_t KW) {
    assert(C == KC)

    const size_t ON = N; const size_t OC = KN;
    const size_t OH = H - KH + 1;
    const size_t OW = W - KW + 1;

    for (size_t on = 0; on < ON; on++) {
        for (size_t oc = 0; oc < OC; oc++) {
            for (size_t oh = 0; oh < OH; oh++) {
                for (size_t ow = 0; ow < OW; ow++) {
        
                }
            }
        }
    }
}

int main() {
    return 0;
}