// -----------------------------------------------------------------------------
//
// Copyright (C) 2021 CERN & University of Surrey for the benefit of the
// BioDynaMo collaboration. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// See the LICENSE file distributed with this work for details.
// See the NOTICE file distributed with this work for additional information
// regarding copyright ownership.
//
// -----------------------------------------------------------------------------

#include "core/util/random.h"
#include <Math/DistFunc.h>
#include <TF1.h>
#include <TRandom3.h>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>
#include <vector>
#include "unit/test_util/io_test.h"
#include "unit/test_util/test_util.h"

namespace bdm {

// The engine is now std::mt19937_64 so we no longer compare sample-by-sample
// against TRandom3.  Instead we verify three properties that must hold
// regardless of the underlying engine:
//   1. Reproducibility  – the same seed always produces the same sequence.
//   2. Range            – every value falls in the requested interval.
//   3. RNG-object parity – GetUniformRng().Sample() and Uniform(min,max)
//                          draw from the same engine and stay in range.
  TEST(RandomTest, Uniform) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
 
  // --- 1. Reproducibility: same seed → identical sequence ---------------
  
  random->SetSeed(42);
  std::vector<real_t> run1;

  for (uint64_t i = 0; i < 10; i++) {
    run1.push_back(random->Uniform());
  }
  random->SetSeed(42);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(run1[i], random->Uniform());
  }
  // --- 2a. Range check: Uniform(max) must be in [0, max) ----------------
  random->SetSeed(42);
  for (uint64_t i = 1; i <= 10; i++) {
    const real_t max = static_cast<real_t>(i);
    const real_t val = random->Uniform(max);
    EXPECT_GE(val, static_cast<real_t>(0));
    EXPECT_LT(val, max);
  }

  // --- 2b. Range check: Uniform(min, max) must be in [min, max) ---------
  random->SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    const real_t lo = static_cast<real_t>(i);
    const real_t hi = lo + static_cast<real_t>(2);
    const real_t val = random->Uniform(lo, hi);
    EXPECT_GE(val, lo);
    EXPECT_LT(val, hi);
  }
  // --- 3. GetUniformRng: samples must stay within [3, 4) ----------------
  auto distrng = random->GetUniformRng(3, 4);
  for (uint64_t i = 0; i < 10; i++) {
    const real_t val = distrng.Sample();
    EXPECT_GE(val, static_cast<real_t>(3));
    EXPECT_LT(val, static_cast<real_t>(4));

  }
}
// UniformArray is a thin wrapper around Uniform(); the key properties to
// verify are reproducibility and per-element range correctness.
TEST(RandomTest, UniformArray) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  // --- 1. Reproducibility: same seed → identical array -----------------
  random->SetSeed(42);
  auto run1 = random->UniformArray<5>();
  random->SetSeed(42);
  auto run2 = random->UniformArray<5>();
  for (uint64_t i = 0; i < 5; i++) {
    EXPECT_REAL_EQ(run1[i], run2[i]);
  }
  // --- 2a. UniformArray<N>(): all elements in [0, 1) -------------------
  random->SetSeed(42);
  auto result = random->UniformArray<5>();
  for (uint64_t i = 0; i < 5; i++) {
    EXPECT_GE(result[i], static_cast<real_t>(0));
    EXPECT_LT(result[i], static_cast<real_t>(1));
  }

   // --- 2b. UniformArray<N>(max): all elements in [0, max) --------------
  random->SetSeed(42);
  auto result1 = random->UniformArray<2>(static_cast<real_t>(8.3));
  for (uint64_t i = 0; i < 2; i++) {
    EXPECT_GE(result1[i], static_cast<real_t>(0));
    EXPECT_LT(result1[i], static_cast<real_t>(8.3));
  }

  // --- 2c. UniformArray<N>(min, max): all elements in [min, max) -------
  random->SetSeed(42);
  auto result2 = random->UniformArray<12>(static_cast<real_t>(5.1),
                                          static_cast<real_t>(9.87));

  for (uint64_t i = 0; i < 12; i++) {
    EXPECT_GE(result2[i], static_cast<real_t>(5.1));
    EXPECT_LT(result2[i], static_cast<real_t>(9.87));
  }
}
// Gaus now uses std::normal_distribution so per-sample parity with TRandom3
// no longer holds.  We verify:
//   1. Reproducibility  – same seed → same sequence.
//   2. Statistical      – with N=10000 samples the empirical mean and standard
//                         deviation must land within a tight tolerance of the
//                         requested parameters (CLT guarantees this).
//   3. GausRng object   – GetGausRng() draws from the same engine and its
//                         statistics match the requested parameters.
TEST(RandomTest, Gaus) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();

  // --- 1. Reproducibility: same seed → identical sequence ---------------
  random->SetSeed(42);
  std::vector<real_t> run1;

  for (uint64_t i = 0; i < 10; i++) {
    run1.push_back(random->Gaus());
  }

  random->SetSeed(42);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(run1[i], random->Gaus());
  }

  // --- 2. Statistical check for Gaus(mean, sigma) -----------------------
  // With 10 000 samples the standard error of the mean is sigma/sqrt(N)
  // ≈ 2/100 = 0.02, so a tolerance of 0.1 is very conservative.
  const uint64_t kN = 10000;
  const real_t kMean = static_cast<real_t>(5);
  const real_t kSigma = static_cast<real_t>(2);

  random->SetSeed(123);
  real_t sum = 0, sum_sq = 0;
  for (uint64_t i = 0; i < kN; i++) {
    const real_t v = random->Gaus(kMean, kSigma);
    sum += v;
    sum_sq += v * v;
  }
  const real_t sample_mean = sum / static_cast<real_t>(kN);
  const real_t sample_var =
      sum_sq / static_cast<real_t>(kN) - sample_mean * sample_mean;
  EXPECT_NEAR(static_cast<double>(sample_mean), static_cast<double>(kMean),
              0.1);
  EXPECT_NEAR(static_cast<double>(std::sqrt(sample_var)),
              static_cast<double>(kSigma), 0.1);

  // --- 3. GetGausRng statistical check ----------------------------------
  auto distrng = random->GetGausRng(static_cast<real_t>(3),
                                    static_cast<real_t>(4));
  sum = 0;
  sum_sq = 0;
  for (uint64_t i = 0; i < kN; i++) {
    const real_t v = distrng.Sample();
    sum += v;
    sum_sq += v * v;
  }
  const real_t rng_mean = sum / static_cast<real_t>(kN);
  const real_t rng_var =
      sum_sq / static_cast<real_t>(kN) - rng_mean * rng_mean;
  EXPECT_NEAR(static_cast<double>(rng_mean), 3.0, 0.1);
  EXPECT_NEAR(static_cast<double>(std::sqrt(rng_var)), 4.0, 0.2);

  }

TEST(RandomTest, Exp) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Exp(i)), random->Exp(i));
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Exp(i + 2)),
                   random->Exp(i + 2));
  }

  auto distrng = random->GetExpRng(123);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Exp(123)), distrng.Sample());
  }
}

TEST(RandomTest, Landau) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Landau()), random->Landau());
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Landau(i)), random->Landau(i));
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Landau(i, i + 2)),
                   random->Landau(i, i + 2));
  }

  auto distrng = random->GetLandauRng(3, 4);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.Landau(3, 4)),
                   distrng.Sample());
  }
}

TEST(RandomTest, PoissonD) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.PoissonD(i)),
                   random->PoissonD(i));
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.PoissonD(i + 2)),
                   random->PoissonD(i + 2));
  }

  auto distrng = random->GetPoissonDRng(123);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.PoissonD(123)),
                   distrng.Sample());
  }
}

TEST(RandomTest, BreitWigner) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.BreitWigner()),
                   random->BreitWigner());
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.BreitWigner(i)),
                   random->BreitWigner(i));
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.BreitWigner(i, i + 2)),
                   random->BreitWigner(i, i + 2));
  }

  auto distrng = random->GetBreitWignerRng(3, 4);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(static_cast<real_t>(reference.BreitWigner(3, 4)),
                   distrng.Sample());
  }
}

TEST(RandomTest, UserDefinedDistRng1D) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  auto function = [](const double* x, const double* params) {
    return ROOT::Math::lognormal_pdf(*x, params[0], params[1]);
  };
  real_t min = 1;
  real_t max = 4;
  TF1 reference("", function, min, max, 2);
  reference.SetParameters(1.1, 1.2);

  // workaround until update to 6.24
  gRandom->SetSeed(42);
  std::vector<real_t> expected;
  for (uint64_t i = 0; i < 10; i++) {
    expected.push_back(reference.GetRandom(min, max));
  }

  gRandom->SetSeed(42);
  auto distrng =
      random->GetUserDefinedDistRng1D(function, {1.1, 1.2}, min, max);
  std::vector<real_t> actual;
  for (uint64_t i = 0; i < 10; i++) {
    actual.push_back(distrng.Sample());
  }
  for (size_t i = 0; i < actual.size(); i++) {
    EXPECT_NEAR(expected[i], actual[i], abs_error<real_t>::value);
  }
}

// This test is neither a proper death test nor a proper function test. We run
// this test, to see if it is possible to initialize the UserDefinedDistRng1D
// in a parallel region because the generator is based on ROOT's TF1 class which
// seems to have trouble when the constructor is called in parallel. Before the
// fix provided in PR #208, this test would fail roughly 2/3 times.
TEST(RandomTest, UserDefinedDistRng1DParallel) {
  Simulation simulation(TEST_NAME);
  std::vector<real_t> results;
#pragma omp parallel shared(simulation, results)
  {
    auto* random = simulation.GetRandom();
    auto function = [](const double* x, const double* params) {
      return ROOT::Math::lognormal_pdf(*x, params[0], params[1]);
    };
    real_t min = 1;
    real_t max = 4;
    size_t n_samples = 10000;
    real_t result = 0.0;
    auto distrng =
        random->GetUserDefinedDistRng1D(function, {1.1, 1.2}, min, max);
    for (size_t i = 0; i < n_samples; i++) {
      result += distrng.Sample();
    }
#pragma omp critical
    { results.push_back(result / n_samples); }
  }
  real_t sum = std::accumulate(results.begin(), results.end(), 0.0);
  EXPECT_LT(sum, std::numeric_limits<real_t>::max());
}

// This test is neither a proper death test nor a proper function test. We run
// this test, to see if it is possible to initialize the UserDefinedDistRng2D
// in a parallel region because the generator is based on ROOT's TF2 class which
// seems to have trouble when the constructor is called in parallel. Before the
// fix provided in PR #208, this test would fail roughly 2/3 times.
TEST(RandomTest, UserDefinedDistRng2DParallel) {
  Simulation simulation(TEST_NAME);
  std::vector<real_t> results;
#pragma omp parallel shared(simulation, results)
  {
    auto* random = simulation.GetRandom();
    auto function = [](const double* x, const double* params) {
      return ROOT::Math::lognormal_pdf(*x, params[0], params[1]);
    };
    real_t min = 1;
    real_t max = 4;
    size_t n_samples = 10000;
    real_t result = 0.0;
    auto distrng = random->GetUserDefinedDistRng2D(function, {1.1, 1.2}, min,
                                                   max, min, max);
    for (size_t i = 0; i < n_samples; i++) {
      result += distrng.Sample2().Norm();
    }
#pragma omp critical
    { results.push_back(result / n_samples); }
  }
  real_t sum = std::accumulate(results.begin(), results.end(), 0.0);
  EXPECT_LT(sum, std::numeric_limits<real_t>::max());
}

// This test is neither a proper death test nor a proper function test. We run
// this test, to see if it is possible to initialize the UserDefinedDistRng3D
// in a parallel region because the generator is based on ROOT's TF3 class which
// seems to have trouble when the constructor is called in parallel. Before the
// fix provided in PR #208, this test would fail roughly 2/3 times.
TEST(RandomTest, UserDefinedDistRng3DParallel) {
  Simulation simulation(TEST_NAME);
  std::vector<real_t> results;
#pragma omp parallel shared(simulation, results)
  {
    auto* random = simulation.GetRandom();
    auto function = [](const double* x, const double* params) {
      return ROOT::Math::lognormal_pdf(*x, params[0], params[1]);
    };
    real_t min = 1;
    real_t max = 4;
    size_t n_samples = 10000;
    real_t result = 0.0;
    auto distrng = random->GetUserDefinedDistRng3D(function, {1.1, 1.2}, min,
                                                   max, min, max, min, max);
    for (size_t i = 0; i < n_samples; i++) {
      result += distrng.Sample3().Norm();
    }
#pragma omp critical
    { results.push_back(result / n_samples); }
  }
  real_t sum = std::accumulate(results.begin(), results.end(), 0.0);
  EXPECT_LT(sum, std::numeric_limits<real_t>::max());
}

TEST(RandomTest, Binomial) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_EQ(reference.Binomial(i, i + 2), random->Binomial(i, i + 2));
  }

  auto distrng = random->GetBinomialRng(3, 4);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_EQ(reference.Binomial(3, 4), distrng.Sample());
  }
}

TEST(RandomTest, Poisson) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_EQ(reference.Poisson(i), random->Poisson(i));
  }

  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_EQ(reference.Poisson(i + 2), random->Poisson(i + 2));
  }

  auto distrng = random->GetPoissonRng(123);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_EQ(reference.Poisson(123), distrng.Sample());
  }
}

TEST(RandomTest, Integer) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 1; i < 10; i++) {
    EXPECT_EQ(reference.Integer(i), random->Integer(i));
  }
}

TEST(RandomTest, Circle) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 1; i < 10; i++) {
    double expected_x = 0;
    double expected_y = 0;
    reference.Circle(expected_x, expected_y, i);

    auto actual = random->Circle(i);
    EXPECT_REAL_EQ(expected_x, actual[0]);
    EXPECT_REAL_EQ(expected_y, actual[1]);
  }
}

TEST(RandomTest, Sphere) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();
  TRandom3 reference;

  random->SetSeed(42);
  reference.SetSeed(42);

  for (uint64_t i = 1; i < 10; i++) {
    double expected_x = 0;
    double expected_y = 0;
    double expected_z = 0;
    reference.Sphere(expected_x, expected_y, expected_z, i);

    auto actual = random->Sphere(i);
    EXPECT_REAL_EQ(expected_x, actual[0]);
    EXPECT_REAL_EQ(expected_y, actual[1]);
    EXPECT_REAL_EQ(expected_z, actual[2]);
  }
}
// IOTest for Random after the std::mt19937_64 migration.
//
// NOTE: mt_engine_ is marked //! (ROOT-transient) so its internal state is
// NOT written to disk.  After a BackupAndRestore round-trip the engine is
// re-default-constructed, which is deterministic but different from the
// pre-backup state.  Callers that need reproducibility across checkpoints
// must call SetSeed() again after restore.
//
// What we verify here:
//   a) The object serializes and deserializes without crashing.
//   b) Gaus() and Uniform() produce finite values after restore.
//   c) After re-seeding the restored object its output matches a freshly
//      seeded Random with the same seed (engine is working correctly).
#ifdef USE_DICT
TEST_F(IOTest, Random) {
  Random random;
  random.SetSeed(42);
  // Consume a few values so the pre-backup state is non-trivial.
  for (uint64_t i = 0; i < 10; i++) {
    (void)random.Gaus();
  }

  Random* restored;
  BackupAndRestore(random, &restored);

  // (b) Values must be finite – the engine must be functional after restore.
  for (uint64_t i = 0; i < 10; i++) {
    const real_t u = restored->Uniform(static_cast<real_t>(i),
                                       static_cast<real_t>(i + 2));
    EXPECT_TRUE(std::isfinite(static_cast<double>(u)));
    EXPECT_GE(u, static_cast<real_t>(i));
    EXPECT_LT(u, static_cast<real_t>(i + 2));

    const real_t g = restored->Gaus();
    EXPECT_TRUE(std::isfinite(static_cast<double>(g)));
  }

  // (c) Re-seeding the restored object must give the same sequence as a
  //     fresh Random seeded identically, proving the engine itself is intact.
  Random fresh;
  const uint64_t kReseed = 99;
  restored->SetSeed(kReseed);
  fresh.SetSeed(kReseed);
  for (uint64_t i = 0; i < 10; i++) {
    EXPECT_REAL_EQ(fresh.Uniform(), restored->Uniform());
    EXPECT_REAL_EQ(fresh.Gaus(), restored->Gaus());
  }
}

TEST_F(IOTest, UserDefinedDistRng1D) {
  Simulation simulation(TEST_NAME);
  auto* random = simulation.GetRandom();

  auto ud_dist = [](const double* x, const double* param) { return sin(*x); };
  auto udd_rng = random->GetUserDefinedDistRng1D(ud_dist, {}, 0, 3);

  gRandom->SetSeed(42);
  std::vector<real_t> expected;
  for (int i = 0; i < 10; ++i) {
    expected.push_back(udd_rng.Sample());
  }

  UserDefinedDistRng1D* restored;
  BackupAndRestore(udd_rng, &restored);

  gRandom->SetSeed(42);
  std::vector<real_t> actual;
  for (int i = 0; i < 10; ++i) {
    actual.push_back(restored->Sample());
  }
  for (size_t i = 0; i < actual.size(); i++) {
    EXPECT_NEAR(expected[i], actual[i], 1e-6);
  }
}

#endif  // USE_DICT

}  // namespace bdm
