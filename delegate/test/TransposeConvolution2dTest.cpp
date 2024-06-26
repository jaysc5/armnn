//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvolutionTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

void TransposeConvInt8Test()
{
    // Set input data
    std::vector<int32_t> transposeTensorShape { 4 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    std::vector<int32_t> transposeData = { 1, 3, 3, 1 };
    static std::vector<int8_t> inputValues = { 1, 2, 3, 4 };
    std::vector<int8_t> filterValues = { 0, 1, 2, 4 };
    std::vector<int8_t> expectedOutputValues =
        {
            0, 1,  2,
            2, 11, 12,
            6, 20, 16
        };

    tflite::Padding padding = tflite::Padding_VALID;
    TransposeConvTest<int8_t>(::tflite::TensorType_INT8,
                              1, // strideX
                              1, // strideY
                              padding,
                              transposeTensorShape,
                              filterShape,
                              inputShape,
                              outputShape,
                              transposeData,
                              filterValues,
                              inputValues,
                              expectedOutputValues);
}

void TransposeConvFp32Test()
{
    std::vector<int32_t> transposeTensorShape { 4 };
    std::vector<int32_t> filterShape { 1, 2, 2, 1 };
    std::vector<int32_t> inputShape { 1, 2, 2, 1 };
    std::vector<int32_t> outputShape { 1, 3, 3, 1 };

    std::vector<int32_t> transposeData = { 1, 3, 3, 1 };
    static std::vector<float> inputValues = { 1, 2, 3, 4 };
    std::vector<float> filterValues = { 0, 1, 2, 4 };
    std::vector<float> expectedOutputValues =
        {
            0, 1,  2,
            2, 11, 12,
            6, 20, 16
        };

    tflite::Padding padding = tflite::Padding_VALID;
    TransposeConvTest<float>(::tflite::TensorType_FLOAT32,
                             1, // strideX
                             1, // strideY
                             padding,
                             transposeTensorShape,
                             filterShape,
                             inputShape,
                             outputShape,
                             transposeData,
                             filterValues,
                             inputValues,
                             expectedOutputValues);
}

TEST_SUITE("TransposeConv_Test")
{

TEST_CASE ("TransposeConv_Fp32_Test")
{
    TransposeConvFp32Test();
}

TEST_CASE ("TransposeConv_Int8_Test")
{
    TransposeConvInt8Test();
}

} // End of  TEST_SUITE(TransposeConv_Test)

} // namespace armnnDelegate