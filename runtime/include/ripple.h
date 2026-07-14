#ifndef RIPPLE_H_
#define RIPPLE_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(RIPPLE_SHARED)
#if defined(RIPPLE_BUILDING_LIBRARY)
#define RIPPLE_API __declspec(dllexport)
#else
#define RIPPLE_API __declspec(dllimport)
#endif
#else
#define RIPPLE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RippleSession RippleSession;

typedef enum RippleStatus {
  RIPPLE_OK = 0,
  RIPPLE_ERROR_INVALID_ARGUMENT = -1,
  RIPPLE_ERROR_UNSUPPORTED_CONFIG = -2,
  RIPPLE_ERROR_ARTIFACT = -3,
  RIPPLE_ERROR_BUFFER_TOO_SMALL = -4,
  RIPPLE_ERROR_PROFILE_TOO_LARGE = -5,
  RIPPLE_ERROR_INTERNAL = -6
} RippleStatus;

typedef enum RippleBackendPreference {
  RIPPLE_BACKEND_AUTO = 0,
  RIPPLE_BACKEND_MOCK = 1,
  RIPPLE_BACKEND_ONNXRUNTIME = 2,
  RIPPLE_BACKEND_TENSORRT = 3,
  RIPPLE_BACKEND_COREML = 4,
  RIPPLE_BACKEND_LITERT = 5,
  RIPPLE_BACKEND_EXECUTORCH = 6
} RippleBackendPreference;

typedef enum RipplePacketStatus {
  RIPPLE_PACKET_OK = 0,
  RIPPLE_PACKET_MISSING = 1,
  RIPPLE_PACKET_LATE = 2
} RipplePacketStatus;

typedef enum RippleHealthFlags {
  RIPPLE_HEALTH_OK = 0,
  RIPPLE_HEALTH_CONCEALED = 1u << 0,
  RIPPLE_HEALTH_LATE_DROPPED = 1u << 1,
  RIPPLE_HEALTH_PROFILE_MISSING = 1u << 2
} RippleHealthFlags;

typedef struct RippleConfig {
  uint32_t sample_rate;
  uint32_t chunk_samples;
  uint32_t flags;
  uint32_t backend_preference;
} RippleConfig;

typedef struct RippleInput {
  const float *pcm;
  uint32_t samples;
  uint32_t packet_status;
  uint64_t timestamp_ns;
} RippleInput;

typedef struct RippleOutput {
  float *pcm;
  uint32_t capacity;
  uint32_t produced;
  uint32_t health_flags;
  float compute_ms;
} RippleOutput;

RIPPLE_API int ripple_create(const char *artifact_path,
                             const RippleConfig *config,
                             RippleSession **session);

RIPPLE_API int ripple_load_speaker_profile(RippleSession *session,
                                           const void *profile,
                                           size_t profile_bytes);

RIPPLE_API int ripple_push(RippleSession *session, const RippleInput *input,
                           RippleOutput *output);

RIPPLE_API int ripple_soft_reset(RippleSession *session);
RIPPLE_API int ripple_hard_reset(RippleSession *session);
RIPPLE_API int ripple_flush(RippleSession *session, RippleOutput *output);
RIPPLE_API void ripple_destroy(RippleSession *session);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // RIPPLE_H_
