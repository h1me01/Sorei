#pragma once

#include <cmath>
#include <string>

#include "sorei/nn.h"

#include "framework.h"

using namespace sorei::nn;

TEST(LRSched, CosineAnnealingLR_InitialValue) {
    CosineAnnealingLR sched(1e-3f, 1e-5f, 100);
    EXPECT_NEAR(sched.lr(), 1e-3f, 1e-8f);
}

TEST(LRSched, CosineAnnealingLR_FinalValue) {
    const int max_steps = 10;
    CosineAnnealingLR sched(1e-2f, 1e-4f, max_steps);
    for (int i = 0; i < max_steps - 1; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.lr(), 1e-4, 1e-6);
}

TEST(LRSched, StepLR_InitialValue) {
    StepLR sched(1e-2f, 0.1f, 10);
    EXPECT_NEAR(sched.lr(), 1e-2f, 1e-8f);
}

TEST(LRSched, StepLR_NoDecayBeforeStepSize) {
    StepLR sched(1e-2f, 0.5f, 5);
    for (int i = 0; i < 4; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.lr(), 1e-2, 1e-7);
}

TEST(LRSched, StepLR_AfterEpochs) {
    const float base = 1.0f, gamma = 0.5f;
    const int step_size = 3;
    StepLR sched(base, gamma, step_size);
    for (int i = 0; i < 2 * step_size; ++i)
        sched.step();
    EXPECT_NEAR((double)sched.lr(), (double)(base * gamma * gamma), 1e-5);
}
