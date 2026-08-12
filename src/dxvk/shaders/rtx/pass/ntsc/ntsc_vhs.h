/*
 * NTSC/VHS composite post-process - shared shader/C++ header.
 * Matches the fork's post_fx.h convention (uint2/float2 resolve on both sides).
 * Place at: src/dxvk/shaders/rtx/pass/ntsc/ntsc_vhs.h
 */
#pragma once

#define NTSC_VHS_INPUT   0   // Sampler2D  : composited LDR color (read)
#define NTSC_VHS_OUTPUT  1   // RWTexture2D: processed output (write)

#define NTSC_VHS_TILE_SIZE 8

struct NtscVhsArgs {
  uint2  imageSize;
  float2 invImageSize;

  float  time;        // seconds (animated noise/dropout)
  float  lumaBW;      // MHz  (SP~3.0, EP/SLP~1.6)
  float  colorBW;     // kHz  (300..500; lower = heavier chroma smudge)
  float  ringing;     // edge peaking gain

  float  ghost;       // multipath ghost mix
  float  tapeTrail;   // 0..1 -> IIR alpha 0.85..0.15 (rightward comet tail)
  float  headSmear;   // banded symmetric luma blur
  float  lumaNoise;   // luminance-dependent grain

  float  vertSoften;  // capture-realism extra (0 = strict Kim match)
  uint   frameIdx;    // for noise animation if time is unavailable
  uint   pass;        // 0 = encode (decode+rainbow+noise), 1 = blur (over the signal)
  float  _pad1;
};
