#pragma once

#define NTSC_VHS_INPUT 0
#define NTSC_VHS_OUTPUT 1
#define NTSC_VHS_TILE_SIZE 8

struct NtscVhsArgs {
  uint2 imageSize;
  float2 invImageSize;

  float time;
  float lumaBW;
  float colorBW;
  float ringing;

  float rainbow;
  float lumaNoise;
  float dropoutRate;
  float dropoutLengthUs;

  float headSmear;
  float tapeTrail;
  float chromaNoise;
  float hueDrift;

  float chromaDelay;
  uint frameIdx;
  uint passIndex;
  float _pad0;

  uint activeSamples;
  uint recordedScanlines;
  float captureAperture;
  float reconstructionSoftness;
};
