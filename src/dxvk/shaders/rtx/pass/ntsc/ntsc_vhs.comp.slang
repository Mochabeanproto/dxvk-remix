/*
 * NTSC/VHS composite post-process. TWO passes (per Kim's note: noise + rainbow
 * must be UNDERNEATH the blur):
 *   pass 0 (encode): NTSC subcarrier decode -> luma/chroma (rainbow + dot crawl),
 *                    edge ringing, and injected luma noise. No blur. -> intermediate
 *   pass 1 (blur):   read the encoded signal and apply the playback blur
 *                    (vertical softness, head-switch smear, tape trail) + ghosting,
 *                    dropout, head-switch band, vignette. -> final
 * Processed on a fixed NTSC grid (NW x NH) regardless of output resolution.
 * Place at: src/dxvk/shaders/rtx/pass/ntsc/ntsc_vhs.comp.slang
 */
#include "rtx/pass/ntsc/ntsc_vhs.h"

layout(binding = NTSC_VHS_INPUT)
Sampler2D InColorSampler;

layout(binding = NTSC_VHS_OUTPUT)
RWTexture2D<float4> OutColorTexture;

layout(push_constant)
ConstantBuffer<NtscVhsArgs> cb;

static const float PI = 3.14159265f;
static const float3 LUMA = float3(0.299f, 0.587f, 0.114f);
static const float NW = 640.0f;   // luma samples per line (NTSC active)
static const float NH = 480.0f;   // visible lines

float  hash(float2 v)  { return frac(sin(dot(v, float2(12.9898f, 78.233f))) * 43758.5453f); }
float  hash3(float3 v) { return frac(sin(dot(v, float3(12.9898f, 78.233f, 37.719f))) * 43758.5453f); }
float3 toYIQ(float3 c) { return float3(dot(c,LUMA), dot(c,float3(0.596f,-0.274f,-0.322f)), dot(c,float3(0.211f,-0.523f,0.312f))); }
float3 toRGB(float3 v) { return float3(dot(v,float3(1.0f,0.956f,0.621f)), dot(v,float3(1.0f,-0.272f,-0.647f)), dot(v,float3(1.0f,-1.106f,1.703f))); }
float3 tex(float2 uv)  { return InColorSampler.SampleLevel(uv, 0.0f).rgb; }

[shader("compute")]
[numthreads(NTSC_VHS_TILE_SIZE, NTSC_VHS_TILE_SIZE, 1)]
void main(uint2 pixelPos : SV_DispatchThreadID)
{
  if (any(pixelPos >= cb.imageSize))
    return;

  float2 uv = (float2(pixelPos) + 0.5f) * cb.invImageSize;
  const float du = 1.0f / NW, dv = 1.0f / NH;
  float ln = floor(uv.y * NH);
  float vy = uv.y, cx = uv.x;
  float field = floor(cb.time * 59.94f);   // NTSC field clock; ALL effects tick on this
  float bb = 1.0f - uv.y;
  float hsw = smoothstep(0.03f, 0.0f, bb);

  // ===================== PASS 0: ENCODE (no blur) =====================
  if (cb.pass == 0u)
  {
    cx += hsw * (hash(float2(ln, field)) - 0.5f) * 0.02f;   // head-switch skew

    // subcarrier: lower density (wider rainbow) + band positions re-randomized every
    // ~30 NTSC fields (interval jittered 20..40 fields), per Kim.
    float blkLen    = 30.0f + (hash(float2(floor(field / 30.0f), 5.0f)) - 0.5f) * 20.0f;
    float bandPhase = hash(float2(floor(field / blkLen), 3.0f)) * 6.2831853f;
    const float fsc = 0.5f;   // subcarrier density (lower = wider rainbow bands)

    // wider luma window notches the subcarrier out of luma -> less dot-crawl crosshatch
    float lumaHalf   = lerp(10.0f, 5.0f, clamp((cb.lumaBW - 1.0f) / 3.0f, 0.0f, 1.0f));
    float chromaHalf = lerp(7.0f, 24.0f, clamp((500.0f - cb.colorBW) / 200.0f, 0.0f, 1.0f));
    float sumY = 0.0f, sumI = 0.0f, sumQ = 0.0f, nY = 0.0f, nC = 0.0f;
    [unroll]
    for (int k = -24; k <= 24; ++k)
    {
      float fk = float(k);
      float3 y = toYIQ(tex(float2(cx + fk * du, vy)));
      float  sx = cx * NW + fk;
      float  p  = PI * 0.5f * fsc * sx + ln * PI + bandPhase;
      float  comp = y.x + y.y * cos(p) + y.z * sin(p);
      float  wY = max(0.0f, 1.0f - abs(fk) / lumaHalf);
      float  wC = max(0.0f, 1.0f - abs(fk) / chromaHalf);
      sumY += comp * wY; nY += wY;
      sumI += comp * cos(p) * wC; sumQ += comp * sin(p) * wC; nC += wC;
    }
    float Y = sumY / nY, I = 1.6f * sumI / nC, Q = 1.6f * sumQ / nC;   // calmer rainbow amplitude

    // edge ringing (unsharp) using sharp source luma
    float Yc = dot(tex(float2(cx, vy)), LUMA);
    Y = Y + cb.ringing * (Yc - Y) * 2.0f;

    // fine PER-PIXEL tape noise injected into the signal (pass 1 blurs it into
    // organic grain). Per-pixel (not the NTSC cell grid) avoids the moire crosshatch.
    float n = (hash3(float3(float(pixelPos.x), float(pixelPos.y), field)) - 0.5f) * cb.lumaNoise;
    Y += n * (1.0f - 0.6f * Y);

    OutColorTexture[pixelPos] = float4(saturate(toRGB(float3(Y, I, Q))), 1.0f);
    return;
  }

  // ===================== PASS 1: BLUR (on top of the signal) =====================
  float3 c = tex(uv);

  // (a) WIDE vertical blur: rainbow bands are horizontal stripes (vary line-to-line),
  //     so a wide vertical blur is what pushes them "underneath the blur" (per Kim).
  float3 vb = float3(0.0f, 0.0f, 0.0f);
  [unroll]
  for (int j = -3; j <= 3; ++j) vb += tex(float2(cx, vy + float(j) * dv));
  vb /= 7.0f;
  c = lerp(c, vb, 0.6f + cb.vertSoften * 0.4f);

  // (b) head-switching smear: horizontal box blur (banded) + a base horizontal softness
  float band = (sin(vy * 180.0f) + sin(vy * 260.0f) + sin(vy * 100.0f)) / 3.0f * 0.5f + 0.5f;
  float rr = (cb.headSmear * (0.4f + band) + 0.5f) * 4.0f * du;
  float3 sm = float3(0.0f, 0.0f, 0.0f);
  [unroll]
  for (int b = -5; b <= 5; ++b) sm += tex(float2(cx + float(b) * rr, vy));
  sm /= 11.0f;
  c = lerp(c, sm, 0.5f + cb.headSmear * 0.5f);

  // (c) tape trail: rightward one-pole IIR comet tail (RGB)
  float a = 0.85f - cb.tapeTrail * 0.70f;
  float3 tr = float3(0.0f, 0.0f, 0.0f); float ww = a, dec = 1.0f - a;
  [unroll]
  for (int t = 0; t <= 20; ++t) { tr += ww * tex(float2(cx - float(t) * du, vy)); ww *= dec; }
  c = lerp(c, tr, 0.85f);

  // (d) ghosting: SMOOTH decaying smear across the offset (not a hard double image)
  float3 gsum = float3(0.0f, 0.0f, 0.0f); float gw = 0.0f;
  [unroll]
  for (int gi = 1; gi <= 6; ++gi) { float w = 1.0f - float(gi) / 7.0f; gsum += tex(float2(cx - 0.028f * float(gi) / 6.0f, vy)) * w; gw += w; }
  c += (gsum / gw) * cb.ghost;

  // (e) dropout: rare, short, not stark white
  float dl = hash(float2(ln * 2.3f, field));
  if (dl > 0.997f)
  {
    float xs = hash(float2(ln, field + 5.0f));
    if (abs(uv.x - xs) < 0.05f) c = lerp(c, float3(0.85f, 0.85f, 0.80f), 0.5f);
  }
  // (f) head-switch noise band at the very bottom
  c = lerp(c, float3(hash3(float3(cx * NW, ln, field))), hsw * 0.5f);
  // (g) mild vignette
  c *= 1.0f - 0.18f * pow(length(uv - 0.5f) * 1.2f, 2.2f);

  OutColorTexture[pixelPos] = float4(saturate(c), 1.0f);
}
