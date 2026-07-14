#include "ripple.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRippleSampleRate = 24000;
constexpr uint32_t kRippleChunkSamples = 480;
constexpr size_t kMaxProfileBytes = 64u * 1024u;

class Arena {
 public:
  explicit Arena(size_t capacity) : storage_(capacity) {}

  template <typename T>
  T *allocate(size_t count) noexcept {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      return nullptr;
    }
    const size_t alignment = alignof(T);
    const size_t aligned = (offset_ + alignment - 1u) & ~(alignment - 1u);
    const size_t bytes = count * sizeof(T);
    if (aligned > storage_.size() || bytes > storage_.size() - aligned) {
      return nullptr;
    }
    auto *result = reinterpret_cast<T *>(storage_.data() + aligned);
    offset_ = aligned + bytes;
    high_water_ = std::max(high_water_, offset_);
    return result;
  }

 private:
  std::vector<std::byte> storage_;
  size_t offset_{0};
  size_t high_water_{0};
};

struct ManifestInfo {
  std::string family;
  std::string backend;
  uint32_t sample_rate{0};
  uint32_t chunk_samples{0};
};

std::optional<std::string> read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    return std::nullopt;
  }
  std::ostringstream contents;
  contents << stream.rdbuf();
  if (!stream.good() && !stream.eof()) {
    return std::nullopt;
  }
  return contents.str();
}

size_t value_start(const std::string &json, const std::string &key) {
  const std::string quoted_key = "\"" + key + "\"";
  const size_t key_position = json.find(quoted_key);
  if (key_position == std::string::npos) {
    return std::string::npos;
  }
  size_t position = json.find(':', key_position + quoted_key.size());
  if (position == std::string::npos) {
    return position;
  }
  ++position;
  while (position < json.size() &&
         std::isspace(static_cast<unsigned char>(json[position])) != 0) {
    ++position;
  }
  return position;
}

std::optional<std::string> json_string(const std::string &json,
                                       const std::string &key) {
  size_t position = value_start(json, key);
  if (position == std::string::npos || position >= json.size() ||
      json[position] != '"') {
    return std::nullopt;
  }
  const size_t end = json.find('"', position + 1u);
  if (end == std::string::npos) {
    return std::nullopt;
  }
  return json.substr(position + 1u, end - position - 1u);
}

std::optional<uint32_t> json_uint32(const std::string &json,
                                    const std::string &key) {
  size_t position = value_start(json, key);
  if (position == std::string::npos || position >= json.size() ||
      !std::isdigit(static_cast<unsigned char>(json[position]))) {
    return std::nullopt;
  }
  uint64_t value = 0;
  while (position < json.size() &&
         std::isdigit(static_cast<unsigned char>(json[position])) != 0) {
    value = value * 10u + static_cast<unsigned>(json[position] - '0');
    if (value > std::numeric_limits<uint32_t>::max()) {
      return std::nullopt;
    }
    ++position;
  }
  return static_cast<uint32_t>(value);
}

std::optional<ManifestInfo> load_manifest(const char *artifact_path) {
  if (artifact_path == nullptr || artifact_path[0] == '\0') {
    return std::nullopt;
  }
  std::filesystem::path path(artifact_path);
  std::error_code error;
  if (std::filesystem::is_directory(path, error)) {
    path /= "manifest.json";
  } else if (path.filename() != "manifest.json") {
    // ZIP bundle loading and checksum/signature validation belong to the
    // production artifact loader. This scaffold accepts an unpacked manifest.
    return std::nullopt;
  }
  const auto contents = read_text_file(path);
  if (!contents) {
    return std::nullopt;
  }
  const auto family = json_string(*contents, "family");
  const auto backend = json_string(*contents, "backend");
  const auto sample_rate = json_uint32(*contents, "sample_rate");
  const auto chunk_samples = json_uint32(*contents, "chunk_samples");
  if (!family || !backend || !sample_rate || !chunk_samples ||
      *family != "ripple-vc") {
    return std::nullopt;
  }
  return ManifestInfo{*family, *backend, *sample_rate, *chunk_samples};
}

}  // namespace

struct RippleSession {
  static constexpr size_t arena_capacity(uint32_t chunk_samples) {
    return (2u * static_cast<size_t>(chunk_samples) * sizeof(float)) +
           kMaxProfileBytes + 64u;
  }

  explicit RippleSession(const RippleConfig &new_config)
      : config(new_config), arena(arena_capacity(new_config.chunk_samples)) {
    previous = arena.allocate<float>(config.chunk_samples);
    scratch = arena.allocate<float>(config.chunk_samples);
    profile = arena.allocate<std::byte>(kMaxProfileBytes);
    if (previous != nullptr) {
      std::fill_n(previous, config.chunk_samples, 0.0F);
    }
    if (scratch != nullptr) {
      std::fill_n(scratch, config.chunk_samples, 0.0F);
    }
  }

  bool valid() const noexcept {
    return previous != nullptr && scratch != nullptr && profile != nullptr;
  }

  RippleConfig config{};
  Arena arena;
  float *previous{nullptr};
  float *scratch{nullptr};
  std::byte *profile{nullptr};
  size_t profile_bytes{0};
  uint64_t pushed_frames{0};
};

extern "C" {

int ripple_create(const char *artifact_path, const RippleConfig *config,
                  RippleSession **session) {
  if (session == nullptr) {
    return RIPPLE_ERROR_INVALID_ARGUMENT;
  }
  *session = nullptr;
  if (config == nullptr) {
    return RIPPLE_ERROR_INVALID_ARGUMENT;
  }
  if (config->sample_rate != kRippleSampleRate ||
      config->chunk_samples != kRippleChunkSamples ||
      (config->backend_preference != RIPPLE_BACKEND_AUTO &&
       config->backend_preference != RIPPLE_BACKEND_MOCK)) {
    return RIPPLE_ERROR_UNSUPPORTED_CONFIG;
  }
  try {
    const auto manifest = load_manifest(artifact_path);
    if (!manifest || manifest->sample_rate != config->sample_rate ||
        manifest->chunk_samples != config->chunk_samples) {
      return RIPPLE_ERROR_ARTIFACT;
    }
    if (config->backend_preference == RIPPLE_BACKEND_AUTO &&
        manifest->backend != "mock" && manifest->backend != "passthrough") {
      return RIPPLE_ERROR_UNSUPPORTED_CONFIG;
    }
    auto *created = new (std::nothrow) RippleSession(*config);
    if (created == nullptr) {
      return RIPPLE_ERROR_INTERNAL;
    }
    if (!created->valid()) {
      delete created;
      return RIPPLE_ERROR_INTERNAL;
    }
    *session = created;
    return RIPPLE_OK;
  } catch (...) {
    return RIPPLE_ERROR_INTERNAL;
  }
}

int ripple_load_speaker_profile(RippleSession *session, const void *profile,
                                size_t profile_bytes) {
  if (session == nullptr || (profile == nullptr && profile_bytes != 0u)) {
    return RIPPLE_ERROR_INVALID_ARGUMENT;
  }
  if (profile_bytes > kMaxProfileBytes) {
    return RIPPLE_ERROR_PROFILE_TOO_LARGE;
  }
  if (profile_bytes != 0u) {
    const auto *source = static_cast<const std::byte *>(profile);
    std::copy_n(source, profile_bytes, session->profile);
  }
  if (profile_bytes < session->profile_bytes) {
    std::fill(session->profile + profile_bytes,
              session->profile + session->profile_bytes, std::byte{0});
  }
  session->profile_bytes = profile_bytes;
  return RIPPLE_OK;
}

int ripple_push(RippleSession *session, const RippleInput *input,
                RippleOutput *output) {
  if (session == nullptr || input == nullptr || output == nullptr ||
      output->pcm == nullptr) {
    return RIPPLE_ERROR_INVALID_ARGUMENT;
  }
  output->produced = 0;
  output->health_flags = RIPPLE_HEALTH_OK;
  output->compute_ms = 0.0F;
  if (output->capacity < session->config.chunk_samples) {
    return RIPPLE_ERROR_BUFFER_TOO_SMALL;
  }
  if (input->packet_status == RIPPLE_PACKET_LATE) {
    output->health_flags = RIPPLE_HEALTH_LATE_DROPPED;
    return RIPPLE_OK;
  }
  if (input->packet_status == RIPPLE_PACKET_MISSING) {
    for (uint32_t index = 0; index < session->config.chunk_samples; ++index) {
      session->scratch[index] = session->previous[index] * 0.98F;
    }
    output->health_flags = RIPPLE_HEALTH_CONCEALED;
  } else {
    if (input->packet_status != RIPPLE_PACKET_OK || input->pcm == nullptr ||
        input->samples != session->config.chunk_samples) {
      return RIPPLE_ERROR_INVALID_ARGUMENT;
    }
    std::copy_n(input->pcm, session->config.chunk_samples, session->scratch);
  }
  std::copy_n(session->scratch, session->config.chunk_samples, output->pcm);
  std::copy_n(session->scratch, session->config.chunk_samples,
              session->previous);
  output->produced = session->config.chunk_samples;
  if (session->profile_bytes == 0u) {
    output->health_flags |= RIPPLE_HEALTH_PROFILE_MISSING;
  }
  ++session->pushed_frames;
  return RIPPLE_OK;
}

int ripple_soft_reset(RippleSession *session) {
  if (session == nullptr) {
    return RIPPLE_ERROR_INVALID_ARGUMENT;
  }
  std::fill_n(session->previous, session->config.chunk_samples, 0.0F);
  std::fill_n(session->scratch, session->config.chunk_samples, 0.0F);
  session->pushed_frames = 0;
  return RIPPLE_OK;
}

int ripple_hard_reset(RippleSession *session) {
  const int status = ripple_soft_reset(session);
  if (status != RIPPLE_OK) {
    return status;
  }
  std::fill_n(session->profile, session->profile_bytes, std::byte{0});
  session->profile_bytes = 0;
  return RIPPLE_OK;
}

int ripple_flush(RippleSession *session, RippleOutput *output) {
  if (session == nullptr || output == nullptr) {
    return RIPPLE_ERROR_INVALID_ARGUMENT;
  }
  output->produced = 0;
  output->health_flags = RIPPLE_HEALTH_OK;
  output->compute_ms = 0.0F;
  return RIPPLE_OK;
}

void ripple_destroy(RippleSession *session) { delete session; }

}  // extern "C"
