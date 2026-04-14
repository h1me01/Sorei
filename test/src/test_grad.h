#pragma once

#include "sorei/nn.h"

#include "framework.h"
#include "grad_check.h"

using namespace sorei;
using namespace sorei::cuda;
using namespace sorei::test;

// ElemwiseUnary gradient checks

TEST(Grad, ElemwiseUnary_ReLU) {
    auto p = make_param({6, 4}, 0.2f, 1.5f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), ReLU{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_ClampedReLU) {
    auto p = make_param({6, 4}, 0.1f, 0.9f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), ClampedReLU{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_SquaredClampedReLU) {
    auto p = make_param({6, 4}, 0.2f, 0.8f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), SquaredClampedReLU{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_Sigmoid) {
    auto p = make_param({6, 4}, -1.5f, 1.5f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), Sigmoid{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.02f);
}

TEST(Grad, ElemwiseUnary_Abs) {
    auto p = make_param({6, 4}, 0.3f, 1.5f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), Abs{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_PowInt2) {
    auto p = make_param({4, 3}, 0.5f, 2.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), PowInt{2});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.02f);
}

TEST(Grad, ElemwiseUnary_PowInt3) {
    auto p = make_param({4, 3}, 0.5f, 2.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), PowInt{3});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_PowFloat_Half) {
    auto p = make_param({4, 3}, 0.5f, 4.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), PowFloat{0.5f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_Clamp) {
    auto p = make_param({4, 3}, -0.5f, 0.5f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), Clamp{-1.0f, 1.0f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.02f);
}

TEST(Grad, ElemwiseUnary_AddScale) {
    auto p = make_param({4, 3}, -1.0f, 1.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), AddScaleUnary{3.0f, -1.0f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.01f);
}

TEST(Grad, ElemwiseUnary_DivLeft) {
    auto p = make_param({4, 3}, 0.5f, 2.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseUnary>(p.get(), DivLeftUnary{6.0f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

// ElemwiseBinary gradient checks

TEST(Grad, ElemwiseBinary_Add) {
    auto a = make_param({4, 3});
    auto b = make_param({4, 3});
    auto layer = std::make_unique<nn::layer::ElemwiseBinary>(a.get(), b.get(), AddBinary{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.01f);
}

TEST(Grad, ElemwiseBinary_Sub) {
    auto a = make_param({4, 3});
    auto b = make_param({4, 3});
    auto layer = std::make_unique<nn::layer::ElemwiseBinary>(a.get(), b.get(), SubBinary{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.01f);
}

TEST(Grad, ElemwiseBinary_Mul) {
    auto a = make_param({4, 3}, 0.2f, 1.0f);
    auto b = make_param({4, 3}, 0.2f, 1.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseBinary>(a.get(), b.get(), MulBinary{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.05f);
}

TEST(Grad, ElemwiseBinary_Div) {
    auto a = make_param({4, 3}, 0.5f, 2.0f);
    auto b = make_param({4, 3}, 1.0f, 3.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseBinary>(a.get(), b.get(), DivBinary{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.05f);
}

// ElemwiseBinary broadcast gradient checks

TEST(Grad, ElemwiseBinary_Broadcast_Add) {
    auto bias = make_param({4, 1});
    auto data = make_param({4, 5});
    auto layer = std::make_unique<nn::layer::ElemwiseBinary>(bias.get(), data.get(), AddBinary{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {bias.get(), data.get()}), 0.01f);
}

TEST(Grad, ElemwiseBinary_Broadcast_Mul) {
    auto scale = make_param({3, 1}, 0.5f, 2.0f);
    auto data = make_param({3, 4}, 0.5f, 2.0f);
    auto layer = std::make_unique<nn::layer::ElemwiseBinary>(scale.get(), data.get(), MulBinary{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {scale.get(), data.get()}), 0.05f);
}

// MatMul gradient check

TEST(Grad, MatMul) {
    auto w = make_param({3, 4});
    auto x = make_param({4, 5});
    auto layer = std::make_unique<nn::layer::MatMul>(w.get(), x.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {w.get(), x.get()}), 0.01f);
}

// Affine gradient check

TEST(Grad, Affine) {
    auto w = make_param({4, 6});
    auto x = make_param({6, 5});
    auto bias = make_param({4, 1}, -0.1f, 0.1f);
    auto layer = std::make_unique<nn::layer::Affine>(x.get(), w.get(), bias.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {w.get(), x.get(), bias.get()}), 0.01f);
}

// Mean gradient check

TEST(Grad, Mean) {
    auto p = make_param({5, 4});
    auto layer = std::make_unique<nn::layer::Mean>(p.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.01f);
}

// PairwiseMul gradient check

TEST(Grad, PairwiseMul) {
    auto p = make_param({6, 4}, 0.2f, 1.0f);
    auto layer = std::make_unique<nn::layer::PairwiseMul>(p.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

// Concat gradient checks

TEST(Grad, Concat_Rows) {
    auto a = make_param({3, 4});
    auto b = make_param({3, 4});
    auto c = make_param({3, 4});
    auto layer = std::make_unique<nn::layer::Concat>(
        std::vector<nn::layer::Layer*>{a.get(), b.get(), c.get()}, nn::layer::ConcatAxis::Rows
    );
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get(), c.get()}), 0.01f);
}

TEST(Grad, Concat_Cols) {
    auto a = make_param({4, 2});
    auto b = make_param({4, 3});
    auto layer = std::make_unique<nn::layer::Concat>(
        std::vector<nn::layer::Layer*>{a.get(), b.get()}, nn::layer::ConcatAxis::Cols
    );
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.01f);
}

// SoftmaxCrossEntropy gradient check

TEST(Grad, SoftmaxCrossEntropy) {
    const int num_classes = 4;
    const int batch = 3;

    auto logits = make_param({num_classes, batch}, -1.0f, 1.0f);
    auto labels = make_input_int(batch, {0, 2, 1});

    auto layer = std::make_unique<nn::layer::SoftmaxCrossEntropy>(logits.get(), labels.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {logits.get()}), 0.05f);
}

// Select gradient check

TEST(Grad, Select) {
    const int count = 2;
    const int output_dim = 4;
    const int batch = 3;

    auto input = make_param({count * output_dim, batch}, -1.0f, 1.0f);

    nn::layer::BucketIndex bidx(count, batch);
    {
        tensor::HostMatrix<int> bdata({1, batch});
        bdata(0, 0) = 0;
        bdata(0, 1) = 1;
        bdata(0, 2) = 0;
        bidx.data().upload(bdata);
    }

    auto layer = std::make_unique<nn::layer::Select>(input.get(), &bidx);
    EXPECT_GRAD_OK(grad_check(layer.get(), {input.get()}), 0.05f);
}
