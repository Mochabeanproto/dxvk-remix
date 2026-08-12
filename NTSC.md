# NTSC / VHS composite post-process

This fork adds an NTSC/VHS composite look as a Remix post-process pass, applied
after tonemapping on the final LDR image (alongside the existing PostFx lens
effects). Modeled on the signal chain in
[Kim2091/ntsc-simulator](https://github.com/Kim2091/ntsc-simulator).

## What it does
Two compute passes on a fixed NTSC grid (640×480), then upscaled:
1. **Encode** — RGB→YIQ, subcarrier modulation, luma/chroma band-limiting and
   imperfect decode (cross-color / dot-crawl "rainbow"), edge ringing, and
   luminance-dependent tape noise injected *into* the signal.
2. **Blur** — vertical softness, head-switch smear, and a rightward one-pole IIR
   tape-trail applied *on top of* the encoded signal (so the noise and rainbow
   sit underneath the blur), plus smooth multipath ghosting, sparse dropouts, a
   bottom head-switch band, and a vignette.

All animated effects tick on a real-time **59.94 Hz NTSC field clock**.

## Files
- `src/dxvk/shaders/rtx/pass/ntsc/ntsc_vhs.comp.slang` — the two-pass shader
- `src/dxvk/shaders/rtx/pass/ntsc/ntsc_vhs.h` — bindings + push-constant args
- `src/dxvk/rtx_render/rtx_postFx.{h,cpp}` — `DxvkPostFx::dispatchNtsc` + options
- `src/dxvk/rtx_render/rtx_context.cpp` — dispatch call in `dispatchPostFxLensEffects`

## Enable / tune
In `rtx.conf` or the Remix menu (Alt+X, search `ntsc`):

```
rtx.ntsc.ntscEnable    = True
rtx.ntsc.ntscLumaBW    = 2.8     # MHz  (lower = softer)
rtx.ntsc.ntscColorBW   = 420     # kHz  (lower = more chroma smudge)
rtx.ntsc.ntscTapeTrail = 0.66    # rightward comet tail
rtx.ntsc.ntscHeadSmear = 0.22
rtx.ntsc.ntscVertSoften= 0.40
rtx.ntsc.ntscGhost     = 0.10
rtx.ntsc.ntscRinging   = 0.30
rtx.ntsc.ntscLumaNoise = 0.012
```

## Build
Standard fork build (`.\scripts\build.ps1 -Flavor release`). Shaders are
auto-discovered, so no build-file changes are needed. Output `d3d9.dll` goes to
`_Comp64Release/src/d3d9/`.
