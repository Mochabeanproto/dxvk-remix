#include "rtx/pass/ntsc/ntsc_vhs.h"

layout(binding = NTSC_VHS_INPUT)
Sampler2D InColorSampler;

layout(binding = NTSC_VHS_OUTPUT)
RWTexture2D<float4> OutColorTexture;

layout(push_constant)
ConstantBuffer<NtscVhsArgs> cb;

static const float PI = 3.14159265f;
static const float TAU = 6.28318531f;
static const float3 LUMA = float3(0.299f, 0.587f, 0.114f);
static const float SAMPLE_RATE = 14318180.24f;
static const float NYQUIST = SAMPLE_RATE * 0.5f;

float hash(float2 v) { return frac(sin(dot(v, float2(12.9898f, 78.233f))) * 43758.5453f); }
float hash3(float3 v) { return frac(sin(dot(v, float3(12.9898f, 78.233f, 37.719f))) * 43758.5453f); }
float3 toYIQ(float3 c) { return float3(dot(c, LUMA), dot(c, float3(0.595901f, -0.274557f, -0.321344f)), dot(c, float3(0.211537f, -0.522736f, 0.311200f))); }
float3 toRGB(float3 v) { return float3(dot(v, float3(1.0f, 0.956f, 0.621f)), dot(v, float3(1.0f, -0.272f, -0.647f)), dot(v, float3(1.0f, -1.106f, 1.703f))); }
float3 linearToSrgb(float3 c) { c = max(c, 0.0f); return lerp(1.055f * pow(c, 1.0f / 2.4f) - 0.055f, 12.92f * c, 1.0f - step(0.0031308f, c)); }
float3 srgbToLinear(float3 c) { c = max(c, 0.0f); return lerp(pow((c + 0.055f) / 1.055f, 2.4f), c / 12.92f, 1.0f - step(0.04045f, c)); }
float3 sampleColor(float2 uv) { return InColorSampler.SampleLevel(uv, 0.0f).rgb; }
float sampleLuma(float2 uv) { return dot(sampleColor(uv), LUMA); }
float3 sampleDisplayColor(float2 uv) { return linearToSrgb(sampleColor(uv)); }
float sampleDisplayLuma(float2 uv) { return dot(sampleDisplayColor(uv), LUMA); }
float frameNumber() { return floor(cb.time * 29.97f) + float(cb.frameIdx); }
float fieldNumber() { return floor(cb.time * 59.94f) + float(cb.frameIdx) * 2.0f; }
float activeSamples() { return max(float(cb.activeSamples), 1.0f); }
float recordedScanlines() { return max(float(cb.recordedScanlines), 1.0f); }

float cubicWeight(float x, float blur, float softness) {
  x = abs(x);
  const float b = saturate(blur);
  const float c = lerp(0.50f, 0.0f, saturate(softness));
  if (x < 1.0f) {
    return ((12.0f - 9.0f * b - 6.0f * c) * x * x * x
      + (-18.0f + 12.0f * b + 6.0f * c) * x * x
      + (6.0f - 2.0f * b)) / 6.0f;
  }
  if (x < 2.0f) {
    return ((-b - 6.0f * c) * x * x * x
      + (6.0f * b + 30.0f * c) * x * x
      + (-12.0f * b - 48.0f * c) * x
      + (8.0f * b + 24.0f * c)) / 6.0f;
  }
  return 0.0f;
}

float3 reconstructTapeRaster(float2 uv) {
  const float2 gridSize = float2(activeSamples(), recordedScanlines());
  const float2 tapePosition = uv * gridSize - 0.5f;
  const float2 basePosition = floor(tapePosition);
  float3 reconstructed = 0.0f;
  float totalWeight = 0.0f;

  [unroll]
  for (int y = -1; y <= 2; ++y) {
    const float weightY = cubicWeight(tapePosition.y - (basePosition.y + float(y)), cb.captureAperture, cb.reconstructionSoftness);
    [unroll]
    for (int x = -1; x <= 2; ++x) {
      const float weightX = cubicWeight(tapePosition.x - (basePosition.x + float(x)), cb.captureAperture, cb.reconstructionSoftness);
      const float weight = weightX * weightY;
      const float2 sampleUv = (basePosition + float2(float(x), float(y)) + 0.5f) / gridSize;
      reconstructed += sampleColor(sampleUv) * weight;
      totalWeight += weight;
    }
  }

  return reconstructed / max(totalWeight, 1e-5f);
}

float lowpassWeight(int offset, float cutoffHz, int taps) {
  float halfWidth = 0.5f * float(taps - 1);
  float n = float(offset);
  float normalizedCutoff = saturate(cutoffHz / NYQUIST);
  float window = 0.54f - 0.46f * cos(TAU * (float(offset) + halfWidth) / float(taps - 1));
  float sinc = abs(n) < 1e-4f ? normalizedCutoff : sin(PI * normalizedCutoff * n) / (PI * n);
  return sinc * window;
}

float3 applyVhsTapePath(float2 uv) {
  const float dx = 1.0f / activeSamples();
  float sumY = 0.0f, sumI = 0.0f, sumQ = 0.0f, normY = 0.0f, normC = 0.0f;
  [loop]
  for (int k = -30; k <= 30; ++k) {
    const float2 sampleUv = float2(uv.x + float(k) * dx, uv.y);
    float3 yiq = toYIQ(sampleDisplayColor(sampleUv));

    // Fine luma detail leaks into a broad, slowly drifting colour carrier.
    const float detail = yiq.x - sampleDisplayLuma(float2(sampleUv.x + 2.0f * dx, sampleUv.y));
    const float sampleIndex = floor(sampleUv.x * activeSamples());
    const float line = floor(uv.y * recordedScanlines());
    const float block = floor(fieldNumber() / 30.0f);
    const float blockPhase = hash(float2(block, 83.0f)) * TAU;
    const float phase = TAU * (sampleIndex / 24.0f + line / (recordedScanlines() * 2.0f)) + blockPhase;
    const float leak = detail * saturate(cb.rainbow) * 0.85f;
    yiq.y += leak * cos(phase);
    yiq.z += leak * sin(phase);

    const float yWeight = lowpassWeight(k, max(cb.lumaBW, 0.25f) * 1e6f, 61);
    const float cWeight = lowpassWeight(k, max(cb.colorBW, 1.0f) * 1e3f, 61);
    sumY += yiq.x * yWeight;
    sumI += yiq.y * cWeight;
    sumQ += yiq.z * cWeight;
    normY += yWeight;
    normC += cWeight;
  }

  float y = sumY / max(normY, 1e-4f);
  float lowpassY = 0.0f, normRing = 0.0f;
  [loop]
  for (int k = -15; k <= 15; ++k) {
    const float weight = lowpassWeight(k, 1.5e6f, 31);
    lowpassY += sampleDisplayLuma(float2(uv.x + float(k) / activeSamples(), uv.y)) * weight;
    normRing += weight;
  }
  y += cb.ringing * (y - lowpassY / max(normRing, 1e-4f));

  // Grain is recorded into the composite signal before playback blur.
  const float2 pixel = floor(uv * float2(activeSamples(), recordedScanlines()));
  const float grain = (hash3(float3(pixel, fieldNumber())) - 0.5f) * cb.lumaNoise;
  y += grain * (1.0f - 0.6f * saturate(y));
  return toRGB(float3(y, sumI / max(normC, 1e-4f), sumQ / max(normC, 1e-4f)));
}

float headSmearEnvelope(float line, float frame) {
  const float p1 = hash(float2(frame + 11.0f, 17.0f)) * TAU;
  const float p2 = hash(float2(frame + 23.0f, 29.0f)) * TAU;
  const float p3 = hash(float2(frame + 37.0f, 41.0f)) * TAU;
  return saturate(0.5f * sin(TAU * line / 73.0f + p1) + 0.3f * sin(TAU * line / 127.0f + p2) + 0.2f * sin(TAU * line / 41.0f + p3));
}

float valueNoise(float2 p, float field, float seed) {
  const float2 cell = floor(p);
  float2 blend = frac(p);
  blend = blend * blend * (3.0f - 2.0f * blend);
  const float n00 = hash3(float3(cell, field + seed));
  const float n10 = hash3(float3(cell + float2(1.0f, 0.0f), field + seed));
  const float n01 = hash3(float3(cell + float2(0.0f, 1.0f), field + seed));
  const float n11 = hash3(float3(cell + 1.0f, field + seed));
  return lerp(lerp(n00, n10, blend.x), lerp(n01, n11, blend.x), blend.y) * 2.0f - 1.0f;
}

float3 applyTapeStages(float2 uv) {
  const float dx = 1.0f / activeSamples();
  const float line = floor(uv.y * recordedScanlines());
  float3 color = sampleColor(uv);
  const float2 delayedUv = float2(uv.x - cb.chromaDelay * dx, uv.y);
  const float3 delayedColor = sampleColor(delayedUv);
  const float3 sourceChroma = delayedColor - dot(delayedColor, LUMA);
  float3 blurredChroma = 0.0f;
  [loop]
  for (int k = -3; k <= 3; ++k) {
    const float3 tap = sampleColor(float2(delayedUv.x + float(k) * dx, uv.y));
    blurredChroma += tap - dot(tap, LUMA);
  }
  float y = dot(color, LUMA);

  // VHS chroma is delayed, phase-unstable, and visibly noisier than luma.
  // Keep the noise low-bandwidth so it forms dancing color clouds rather than RGB grain.
  float3 chromaYiq = toYIQ(lerp(sourceChroma, blurredChroma / 7.0f, 0.68f));
  const float field = fieldNumber();
  const float slowPhase = cb.hueDrift * (
    0.70f * sin(cb.time * 0.37f) +
    0.30f * sin(cb.time * 0.11f + 1.70f) +
    0.18f * (hash(float2(floor(field), 193.0f)) - 0.5f));
  const float phaseCos = cos(slowPhase);
  const float phaseSin = sin(slowPhase);
  const float2 rotatedChroma = float2(
    chromaYiq.y * phaseCos - chromaYiq.z * phaseSin,
    chromaYiq.y * phaseSin + chromaYiq.z * phaseCos);
  const float2 noiseUv = uv * float2(activeSamples() / 6.0f, recordedScanlines() * 0.5f);
  const float darkWeight = 0.65f + 0.35f * (1.0f - saturate(y));
  const float2 chromaNoise = float2(
    valueNoise(noiseUv, floor(field), 17.0f),
    valueNoise(noiseUv + float2(11.3f, 7.1f), floor(field), 59.0f));
  const float fieldSaturation = 1.0f + cb.chromaNoise * 1.5f * (hash(float2(floor(field), 227.0f)) - 0.5f);
  chromaYiq.yz = rotatedChroma * fieldSaturation + chromaNoise * cb.chromaNoise * darkWeight;
  const float3 chroma = toRGB(chromaYiq);

  const int maxRadius = int(ceil(saturate(cb.headSmear) * 32.0f));
  const int radius = int(floor(headSmearEnvelope(line, frameNumber()) * float(maxRadius) + 0.5f));
  if (radius > 0) {
    float sum = 0.0f, count = 0.0f;
    [loop]
    for (int k = -32; k <= 32; ++k) {
      if (abs(k) <= radius) {
        sum += sampleLuma(float2(uv.x + float(k) * dx, uv.y));
        count += 1.0f;
      }
    }
    y = sum / max(count, 1.0f);
  }
  return float3(y, y, y) + chroma;
}

float3 applyTapeDropout(float2 uv) {
  const float line = floor(uv.y * recordedScanlines());
  const float frame = frameNumber();
  const float event = hash(float2(line + 47.0f, frame + 53.0f));
  const float start = hash(float2(line + 59.0f, frame + 61.0f)) * activeSamples();
  const float length = max(4.0f, -log(max(hash(float2(line + 71.0f, frame + 73.0f)), 1e-4f)) * cb.dropoutLengthUs * 14.31818f);
  const float x = uv.x * activeSamples();
  if (cb.dropoutRate <= 0.0f || event >= min(cb.dropoutRate / recordedScanlines(), 1.0f) || x < start || x >= start + length)
    return sampleColor(uv);
  if (length < 20.0f)
    return float3(1.0857f + hash(float2(line + 67.0f, frame + 79.0f)) * 0.25f);
  return sampleColor(float2(uv.x, max(uv.y - 1.0f / recordedScanlines(), 0.0f)));
}

float3 applyTapeTrail(float2 uv) {
  const float dx = 1.0f / activeSamples();
  const float alpha = 1.0f - saturate(cb.tapeTrail) * 0.70f;
  const float decay = 1.0f - alpha;
  float trailY = 0.0f, weight = alpha;
  [unroll]
  for (int k = 0; k <= 32; ++k) {
    trailY += weight * sampleLuma(float2(uv.x - float(k) * dx, uv.y));
    weight *= decay;
  }
  const float3 color = sampleColor(uv);
  return float3(trailY, trailY, trailY) + color - dot(color, LUMA);
}

[shader("compute")]
[numthreads(NTSC_VHS_TILE_SIZE, NTSC_VHS_TILE_SIZE, 1)]
void main(uint2 pixelPos : SV_DispatchThreadID) {
  if (any(pixelPos >= cb.imageSize))
    return;

  const float2 uv = (float2(pixelPos) + 0.5f) * cb.invImageSize;
  float3 color;
  if (cb.passIndex == 0) {
    color = reconstructTapeRaster(uv);
  } else if (cb.passIndex == 1) {
    color = applyVhsTapePath(uv);
  } else if (cb.passIndex == 2) {
    color = applyTapeStages(uv);
  } else if (cb.passIndex == 3) {
    color = applyTapeDropout(uv);
  } else {
    color = srgbToLinear(saturate(applyTapeTrail(uv)));
  }
  OutColorTexture[pixelPos] = float4(color, 1.0f);
}
