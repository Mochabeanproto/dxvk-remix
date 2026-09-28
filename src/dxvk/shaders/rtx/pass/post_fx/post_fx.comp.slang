/*
* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx/pass/post_fx/post_fx.h"

layout(binding = POST_FX_INPUT)
Sampler2D InColorSampler;

layout(binding = POST_FX_OUTPUT)
RWTexture2D<float4> OutColorTexture;

layout(push_constant)
ConstantBuffer<PostFxArgs> cb;

// Heat haze (fork): a screen-space shimmer above hot surfaces. Heat rises, so we probe the colour a little
// BELOW this pixel; bright + warm (orange, R >> B) reads as lava/fire, and the air above it wobbles. Returns
// a UV offset to perturb the colour sampling by. No extra buffers -- the mask comes from the image itself.
float2 heatHazeOffset(const float2 uv, const float2 invImageSize, const float time, const float strength)
{
  if (strength <= 0.0f) { return float2(0.0f, 0.0f); }
  const float3 below = InColorSampler.SampleLevel(uv + float2(0.0f, 26.0f * invImageSize.y), 0.0f).rgb;
  const float lum = dot(below, float3(0.299f, 0.587f, 0.114f));
  const float warmth = saturate((below.r - below.b) * 1.5f);
  const float heat = saturate(lum * 0.6f) * warmth;
  if (heat <= 0.001f) { return float2(0.0f, 0.0f); }
  // Animated interference wobble; scrolls upward like rising heat.
  const float2 p = uv * float2(70.0f, 40.0f);
  const float t = time * 2.2f;
  float2 warp;
  warp.x = sin(p.y * 1.0f + t) + 0.5f * sin(p.y * 2.7f - t * 1.3f + p.x);
  warp.y = 0.5f * sin(p.x * 1.7f + t * 1.1f);
  return warp * (heat * strength * 10.0f) * invImageSize;   // pixel-scale UV offset
}

float3 calculateColorFringe(const uint2  pixelPos,
                            const uint2  imageSize,
                            const float2 invImageSize,
                            const float2 chromaticAberrationScale,
                            const float  chromaticCenterAttenuationAmount,
                            const float2 uvOffset)
{
  // Basic center based attenuation, mitigate the chromatic aberration effect from center
  const float2 centerAttenuation = (((float2)pixelPos + 0.5f) * 2.0f - (float2)imageSize) * max(invImageSize.x, invImageSize.y);
  const float  centerAttenuationSquare = dot(centerAttenuation, centerAttenuation);
  const float  chromaticCenterAttenuation = max(0.0f, 1.0f - centerAttenuationSquare * (1.0f - chromaticCenterAttenuationAmount));

  const float2 uv = (pixelPos + 0.5f) * (float2)invImageSize;
  const float2 chromaticAberrationUvScale = 1.0f + chromaticAberrationScale * (1.0f - chromaticCenterAttenuation);

  // Simulate Lens Color Shift
  const float chromaticAberrationR = InColorSampler.SampleLevel((uv - 0.5f) / chromaticAberrationUvScale + 0.5f + uvOffset, 0.0f).r;
  const float chromaticAberrationG = InColorSampler.SampleLevel(uv + uvOffset, 0.0f).g;
  const float chromaticAberrationB = InColorSampler.SampleLevel((uv - 0.5f) * chromaticAberrationUvScale + 0.5f + uvOffset, 0.0f).b;

  return float3(chromaticAberrationR, chromaticAberrationG, chromaticAberrationB);
}

float calculateVignetteAttenuation(const uint2  pixelPos,
                                   const uint2  imageSize,
                                   const float2 invImageSize,
                                   const float  vignetteIntensity,
                                   const float  vignetteRadius,
                                   const float  vignetteSoftness)
{
  const float2 ndc = (((float2)pixelPos + 0.5f) * 2.0f - (float2)imageSize) * min(invImageSize.x, invImageSize.y);

  const float2 maxRadius = length((1.0f - (float2)imageSize) * invImageSize);
  const float2 factor = smoothstep(vignetteRadius.xx, maxRadius, length(ndc));
  const float2 centerAttenuation = lerp(0.0f, 1.0f - vignetteSoftness, factor);
  const float  centerAttenuationSquare = dot(centerAttenuation, centerAttenuation);

  // https://grail.cs.washington.edu/projects/vignette/vign.iccv05.pdf
  return max(0.0f, 1.0f - centerAttenuationSquare * vignetteIntensity);
}

[shader("compute")]
[numthreads(POST_FX_TILE_SIZE, POST_FX_TILE_SIZE, 1)]
void main(uint2 pixelPos : SV_DispatchThreadID)
{
  if (any(pixelPos >= cb.imageSize))
  {
    return;
  }

  float3 result = 0.0f;
  // 0. Heat haze: UV warp above hot (lava/fire) surfaces, folded into the colour sampling below.
  const float2 heatUv = (pixelPos + 0.5f) * cb.invImageSize;
  const float2 heatOffset = heatHazeOffset(heatUv, cb.invImageSize, cb.heatHazeTime, cb.heatHazeStrength);
  // 1. Chromatic Aberration (+ heat-haze offset)
  result = calculateColorFringe(pixelPos, cb.imageSize, cb.invImageSize, cb.chromaticAberrationScale, cb.chromaticCenterAttenuationAmount, heatOffset);

  // 2. Vignette
  result *= calculateVignetteAttenuation(pixelPos, cb.imageSize, cb.invImageSize, cb.vignetteIntensity, cb.vignetteRadius, cb.vignetteSoftness);

  // Write back to final display texture
  OutColorTexture[pixelPos] = float4(result, 1.0f);
}
