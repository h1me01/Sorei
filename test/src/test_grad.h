#pragma once

#include "sorei/nn.h"

#include "framework.h"
#include "grad_check.h"

using namespace sorei;
using namespace sorei::cuda;
using namespace sorei::test;

namespace unary = sorei::nn::unary;
namespace binary = sorei::nn::binary;

TEST(Grad, ElemwiseUnary_ReLU) {
    auto p = make_param({6, 4}, 0.2f, 1.5f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::ReLU{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_ClampedReLU) {
    auto p = make_param({6, 4}, 0.1f, 0.9f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::ClampedReLU{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_SquaredClampedReLU) {
    auto p = make_param({6, 4}, 0.2f, 0.8f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::SquaredClampedReLU{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_Sigmoid) {
    auto p = make_param({6, 4}, -1.5f, 1.5f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::Sigmoid{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.02f);
}

TEST(Grad, ElemwiseUnary_Abs) {
    auto p = make_param({6, 4}, 0.3f, 1.5f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::Abs{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseUnary_Clamp) {
    auto p = make_param({4, 3}, -0.5f, 0.5f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::Clamp{-1.0f, 1.0f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.02f);
}

TEST(Grad, ElemwiseUnary_AddScale) {
    auto p = make_param({4, 3}, -1.0f, 1.0f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::AddScale{3.0f, -1.0f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.01f);
}

TEST(Grad, ElemwiseUnary_DivLeft) {
    auto p = make_param({4, 3}, 0.5f, 2.0f);
    auto layer = std::make_unique<nn::ElemwiseUnary>(p.get(), unary::DivLeft{6.0f});
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, ElemwiseBinary_Add) {
    auto a = make_param({4, 3});
    auto b = make_param({4, 3});
    auto layer = std::make_unique<nn::ElemwiseBinary>(a.get(), b.get(), binary::Add{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.01f);
}

TEST(Grad, ElemwiseBinary_Sub) {
    auto a = make_param({4, 3});
    auto b = make_param({4, 3});
    auto layer = std::make_unique<nn::ElemwiseBinary>(a.get(), b.get(), binary::Sub{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.01f);
}

TEST(Grad, ElemwiseBinary_Mul) {
    auto a = make_param({4, 3}, 0.2f, 1.0f);
    auto b = make_param({4, 3}, 0.2f, 1.0f);
    auto layer = std::make_unique<nn::ElemwiseBinary>(a.get(), b.get(), binary::Mul{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.05f);
}

TEST(Grad, ElemwiseBinary_Div) {
    auto a = make_param({4, 3}, 0.5f, 2.0f);
    auto b = make_param({4, 3}, 1.0f, 3.0f);
    auto layer = std::make_unique<nn::ElemwiseBinary>(a.get(), b.get(), binary::Div{});
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.05f);
}

TEST(Grad, MatMul) {
    auto w = make_param({3, 4});
    auto x = make_param({4, 5});
    auto layer = std::make_unique<nn::MatMul>(w.get(), x.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {w.get(), x.get()}), 0.01f);
}

TEST(Grad, Affine) {
    auto w = make_param({4, 6});
    auto x = make_param({6, 5});
    auto bias = make_param({4, 1}, -0.1f, 0.1f);
    auto layer = std::make_unique<nn::Affine>(x.get(), w.get(), bias.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {w.get(), x.get(), bias.get()}), 0.01f);
}

TEST(Grad, Mean) {
    auto p = make_param({5, 4});
    auto layer = std::make_unique<nn::Mean>(p.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.01f);
}

TEST(Grad, PairwiseMul) {
    auto p = make_param({6, 4}, 0.2f, 1.0f);
    auto layer = std::make_unique<nn::PairwiseMul>(p.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {p.get()}), 0.05f);
}

TEST(Grad, Concat_Rows) {
    auto a = make_param({3, 4});
    auto b = make_param({3, 4});
    auto c = make_param({3, 4});
    auto layer = std::make_unique<nn::Concat>(
        std::vector<nn::Layer*>{a.get(), b.get(), c.get()}, nn::ConcatAxis::Rows
    );
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get(), c.get()}), 0.01f);
}

TEST(Grad, Concat_Cols) {
    auto a = make_param({4, 2});
    auto b = make_param({4, 3});
    auto layer = std::make_unique<nn::Concat>(
        std::vector<nn::Layer*>{a.get(), b.get()}, nn::ConcatAxis::Cols
    );
    EXPECT_GRAD_OK(grad_check(layer.get(), {a.get(), b.get()}), 0.01f);
}

TEST(Grad, SoftmaxCrossEntropy) {
    const int num_classes = 4;
    const int batch = 3;

    auto logits = make_param({num_classes, batch}, -1.0f, 1.0f);
    auto labels = make_input_int(batch, {0, 2, 1});

    auto layer = std::make_unique<nn::SoftmaxCrossEntropy>(logits.get(), labels.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {logits.get()}), 0.05f);
}

TEST(Grad, Select) {
    const int count = 2;
    const int output_dim = 4;
    const int batch = 3;

    auto input = make_param({count * output_dim, batch}, -1.0f, 1.0f);

    nn::BucketIndex bidx("bidx", count, batch);
    {
        matrix::HostMatrix<int> bdata({1, batch});
        bdata(0, 0) = 0;
        bdata(0, 1) = 1;
        bdata(0, 2) = 0;
        bidx.data().upload(bdata);
    }

    auto layer = std::make_unique<nn::Select>(input.get(), &bidx);
    EXPECT_GRAD_OK(grad_check(layer.get(), {input.get()}), 0.05f);
}

TEST(Grad, SparseAffine_ScalarPath) {
    const int out_dim = 3, n_features = 5, max_entries = 3, batch = 4;

    auto weight = make_param({out_dim, n_features}, -0.5f, 0.5f);
    auto bias = make_param({out_dim, 1}, -0.1f, 0.1f);

    nn::Input<int> indices("indices", {max_entries, batch});
    {
        matrix::HostMatrix<int> idx({max_entries, batch});
        idx(0, 0) = 0;
        idx(1, 0) = 2;
        idx(2, 0) = -1;
        idx(0, 1) = 1;
        idx(1, 1) = -1;
        idx(2, 1) = -1;
        idx(0, 2) = 3;
        idx(1, 2) = 4;
        idx(2, 2) = -1;
        idx(0, 3) = 0;
        idx(1, 3) = -1;
        idx(2, 3) = -1;
        indices.data().upload(idx);
    }

    auto layer = std::make_unique<nn::SparseAffine>(&indices, weight.get(), bias.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {weight.get(), bias.get()}), 0.05f);
}

TEST(Grad, SparseAffine_VecPath) {
    const int out_dim = 4, n_features = 4, max_entries = 3, batch = 3;

    auto weight = make_param({out_dim, n_features}, -0.5f, 0.5f);
    auto bias = make_param({out_dim, 1}, -0.1f, 0.1f);

    nn::Input<int> indices("indices", {max_entries, batch});
    {
        matrix::HostMatrix<int> idx({max_entries, batch});
        idx(0, 0) = 0;
        idx(1, 0) = 3;
        idx(2, 0) = -1;
        idx(0, 1) = 1;
        idx(1, 1) = 2;
        idx(2, 1) = -1;
        idx(0, 2) = 2;
        idx(1, 2) = -1;
        idx(2, 2) = -1;
        indices.data().upload(idx);
    }

    auto layer = std::make_unique<nn::SparseAffine>(&indices, weight.get(), bias.get());
    EXPECT_GRAD_OK(grad_check(layer.get(), {weight.get(), bias.get()}), 0.05f);
}
