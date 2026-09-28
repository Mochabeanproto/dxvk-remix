/*
* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include "rtx/utility/shader_types.h"

// Runtime displacement mapping via geometry tessellation ("SPOM"): subdivides each source triangle
// into an N x N barycentric grid and offsets the generated vertices outward along the interpolated
// normal by the sampled heightmap, producing REAL geometry in the BLAS (true silhouettes + shadows),
// unlike the shading-only POM path. See dispatchTessellateDisplace / rtx_scene_manager integration.

#define BINDING_TD_CONSTANTS        0
#define BINDING_TD_SRC_POSITION     1
#define BINDING_TD_SRC_NORMAL       2
#define BINDING_TD_SRC_TEXCOORD     3
#define BINDING_TD_SRC_INDEX        4
#define BINDING_TD_DST_POSITION     5
#define BINDING_TD_DST_NORMAL       6
#define BINDING_TD_DST_TEXCOORD     7
#define BINDING_TD_HEIGHT_TEXTURE   8
#define BINDING_TD_HEIGHT_SAMPLER   9

struct TessellateDisplaceArgs {
  uint  srcTriangleCount;   // number of triangles in the source geometry (one thread each)
  uint  tessFactor;         // N: each source triangle -> N*N sub-triangles
  float displaceScale;      // world-space outward offset at height == 1.0
  uint  nearestFilter;      // 1 = snap the height sample to the texel center (crisp voxel steps)

  // Source buffers are raw float arrays addressed by (offset + vertexIndex*stride) in BYTES.
  uint  srcPositionOffset;  uint srcPositionStride;
  uint  srcNormalOffset;    uint srcNormalStride;
  uint  srcTexcoordOffset;  uint srcTexcoordStride;

  // Index buffer (ByteAddressBuffer). useIndices==0 => sequential (i*3) source triangles.
  uint  srcIndexOffset;     uint useIndices;
  uint  indexIs32Bit;       uint pad0;

  // Destination is a densely packed triangle soup: 3 verts per sub-triangle, sequential.
  uint  dstPositionOffset;  uint dstPositionStride;
  uint  dstNormalOffset;    uint dstNormalStride;
  uint  dstTexcoordOffset;  uint dstTexcoordStride;

  uint  heightTexWidth;     uint heightTexHeight;   // for nearest texel-center snapping
  float heightBias;         uint pad1;              // subtract from sampled height (e.g. 0.5 = center)
};
