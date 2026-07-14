#include "ripple.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

constexpr size_t kChunkSamples = 480;

bool check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAILED: " << message << '\n';
  }
  return condition;
}

}  // namespace

int main() {
  bool passed = true;
  const auto artifact =
      std::filesystem::temp_directory_path() / "ripple_runtime_mock_artifact";
  std::error_code error;
  std::filesystem::remove_all(artifact, error);
  std::filesystem::create_directories(artifact);
  {
    std::ofstream manifest(artifact / "manifest.json");
    manifest << R"({
      "family": "ripple-vc",
      "backend": "mock",
      "sample_rate": 24000,
      "chunk_samples": 480
    })";
  }

  RippleConfig config{24000, 480, 0, RIPPLE_BACKEND_MOCK};
  RippleSession *session = nullptr;
  passed &= check(
      ripple_create(artifact.string().c_str(), &config, &session) == RIPPLE_OK,
      "create accepts a valid unpacked mock artifact");
  passed &= check(session != nullptr, "create returns a session");
  if (session == nullptr) {
    return 1;
  }

  const std::array<unsigned char, 4> profile{1, 2, 3, 4};
  passed &= check(
      ripple_load_speaker_profile(session, profile.data(), profile.size()) ==
          RIPPLE_OK,
      "speaker profile loads");

  std::array<float, kChunkSamples> input_pcm{};
  std::array<float, kChunkSamples> output_pcm{};
  for (size_t index = 0; index < input_pcm.size(); ++index) {
    input_pcm[index] = static_cast<float>(index) / 480.0F;
  }
  RippleInput input{input_pcm.data(), 480, RIPPLE_PACKET_OK, 123};
  RippleOutput output{output_pcm.data(), 480, 0, 0, -1.0F};
  passed &= check(ripple_push(session, &input, &output) == RIPPLE_OK,
                  "push succeeds");
  passed &= check(output.produced == 480, "push produces one fixed chunk");
  passed &= check(output.compute_ms == 0.0F, "mock timing is deterministic");
  passed &= check(output.health_flags == RIPPLE_HEALTH_OK,
                  "loaded profile produces healthy status");
  for (size_t index = 0; index < input_pcm.size(); ++index) {
    passed &= check(input_pcm[index] == output_pcm[index],
                    "mock backend is passthrough");
  }

  input = RippleInput{nullptr, 0, RIPPLE_PACKET_MISSING, 124};
  passed &= check(ripple_push(session, &input, &output) == RIPPLE_OK,
                  "missing packet is concealed");
  passed &= check(
      (output.health_flags & RIPPLE_HEALTH_CONCEALED) != 0,
      "concealment health flag is set");
  passed &= check(std::fabs(output_pcm[479] - input_pcm[479] * 0.98F) < 1e-6F,
                  "concealment attenuates the previous frame");

  passed &= check(ripple_soft_reset(session) == RIPPLE_OK,
                  "soft reset succeeds");
  passed &= check(ripple_push(session, &input, &output) == RIPPLE_OK,
                  "push after soft reset succeeds");
  passed &= check(output_pcm[479] == 0.0F,
                  "soft reset clears source-derived frame state");
  passed &= check((output.health_flags & RIPPLE_HEALTH_PROFILE_MISSING) == 0,
                  "soft reset retains speaker profile");

  passed &= check(ripple_hard_reset(session) == RIPPLE_OK,
                  "hard reset succeeds");
  passed &= check(ripple_push(session, &input, &output) == RIPPLE_OK,
                  "push after hard reset succeeds");
  passed &= check(
      (output.health_flags & RIPPLE_HEALTH_PROFILE_MISSING) != 0,
      "hard reset clears speaker profile");

  passed &= check(ripple_flush(session, &output) == RIPPLE_OK,
                  "flush succeeds");
  passed &= check(output.produced == 0, "mock backend has no decoder tail");
  ripple_destroy(session);
  std::filesystem::remove_all(artifact, error);
  return passed ? 0 : 1;
}
