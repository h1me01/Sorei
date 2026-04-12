#pragma once

#include <cmath>
#include <string>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei::nn::lr_sched;

// CosineAnnealing

TEST(LRSched, CosineAnnealing_InitialValue) {
    CosineAnnealing sched(1e-3f, 1e-5f, 100);
    EXPECT_NEAR(sched.get(), 1e-3f, 1e-8f);
}

TEST(LRSched, CosineAnnealing_FinalValue) {
    const int max_steps = 10;
    CosineAnnealing sched(1e-2f, 1e-4f, max_steps);
    for (int i = 0; i < max_steps - 1; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.get(), 1e-4, 1e-6);
}

TEST(LRSched, CosineAnnealing_Midpoint) {
    const int max_steps = 11;
    const float start = 1.0f, end = 0.0f;
    CosineAnnealing sched(start, end, max_steps);
    for (int i = 0; i < 5; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.get(), 0.5, 1e-5);
}

TEST(LRSched, CosineAnnealing_Monotone_Decreasing) {
    const int max_steps = 20;
    CosineAnnealing sched(1e-2f, 1e-5f, max_steps);
    float prev = sched.get();
    for (int i = 0; i < max_steps - 1; ++i) {
        sched.step();
        float cur = sched.get();
        EXPECT_LE((double)cur, (double)prev + 1e-8);
        prev = cur;
    }
}

TEST(LRSched, CosineAnnealing_Monotone_Increasing) {
    const int max_steps = 20;
    CosineAnnealing sched(1e-5f, 1e-2f, max_steps);
    float prev = sched.get();
    for (int i = 0; i < max_steps - 1; ++i) {
        sched.step();
        float cur = sched.get();
        EXPECT_GE((double)cur, (double)prev - 1e-8);
        prev = cur;
    }
}

TEST(LRSched, CosineAnnealing_ClampsBeyondMaxSteps) {
    CosineAnnealing sched(1.0f, 0.0f, 5);
    for (int i = 0; i < 10; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.get(), 0.0, 1e-5);
}

TEST(LRSched, CosineAnnealing_InfoString) {
    CosineAnnealing sched(1e-3f, 1e-5f, 100);
    std::string info = sched.info();
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(
        info.find("Cosine") != std::string::npos || info.find("cosine") != std::string::npos
    );
}

// StepDecay

TEST(LRSched, StepDecay_InitialValue) {
    StepDecay sched(1e-2f, 0.1f, 10);
    EXPECT_NEAR(sched.get(), 1e-2f, 1e-8f);
}

TEST(LRSched, StepDecay_NoDecayBeforeStepSize) {
    StepDecay sched(1e-2f, 0.5f, 5);
    for (int i = 0; i < 4; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.get(), 1e-2, 1e-7);
}

TEST(LRSched, StepDecay_AfterOneDecayEpoch) {
    const float base = 0.1f, gamma = 0.5f;
    const int step_size = 4;
    StepDecay sched(base, gamma, step_size);
    for (int i = 0; i < step_size; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.get(), (double)base * gamma, 1e-6);
}

TEST(LRSched, StepDecay_AfterTwoDecayEpochs) {
    const float base = 1.0f, gamma = 0.5f;
    const int step_size = 3;
    StepDecay sched(base, gamma, step_size);
    for (int i = 0; i < 2 * step_size; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.get(), (double)(base * gamma * gamma), 1e-5);
}

TEST(LRSched, StepDecay_InfoString) {
    StepDecay sched(1e-3f, 0.5f, 10);
    std::string info = sched.info();
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(
        info.find("Step") != std::string::npos || info.find("step") != std::string::npos ||
        info.find("Decay") != std::string::npos
    );
}

TEST(LRSched, StepDecay_GammaZero_DropsToZero) {
    StepDecay sched(1.0f, 0.0f, 1);
    sched.step();
    EXPECT_NEAR((double)sched.get(), 0.0, 1e-7);
}
