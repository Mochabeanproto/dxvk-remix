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

// Runtime displacement mapping via geometry tessellation. One thread per SOURCE triangle: subdivide
// it into an N x N barycentric grid and emit N*N sub-triangles as a densely packed triangle soup,
// each generated vertex pushed outward along the (source-interpolated) normal by the sampled height.
// The result is REAL displaced geometry that goes into the BLAS — so it produces true silhouettes,
// shadows, and neighbour occlusion, which shading-only POM cannot.

#include "tessellate_displace_binding_indices.h"

layout(binding = BINDING_TD_CONSTANTS)
ConstantBuffer<TessellateDisplaceArgs> cb;

layout(binding = BINDING_TD_SRC_POSITION) StructuredBuffer<float> srcPosition;
layout(binding = BINDING_TD_SRC_NORMAL)   StructuredBuffer<float> srcNormal;
layout(binding = BINDING_TD_SRC_TEXCOORD) StructuredBuffer<float> srcTexcoord;
layout(binding = BINDING_TD_SRC_INDEX)    ByteAddressBuffer       srcIndex;

layout(binding = BINDING_TD_DST_POSITION) RWStructuredBuffer<float> dstPosition;
layout(binding = BINDING_TD_DST_NORMAL)   RWStructuredBuffer<float> dstNormal;
layout(binding = BINDING_TD_DST_TEXCOORD) RWStructuredBuffer<float> dstTexcoord;

layout(binding = BINDING_TD_HEIGHT_TEXTURE) Texture2D<float4> heightTexture;
layout(binding = BINDING_TD_HEIGHT_SAMPLER) SamplerState heightSampler;

float3 readVec3(StructuredBuffer<float> buf, uint byteOffset, uint stride, uint vertexIndex) {
  const uint base = (byteOffset + vertexIndex * stride) / 4u;
  return float3(buf[base + 0u], buf[base + 1u], buf[base + 2u]);
}

float2 readVec2(StructuredBuffer<float> buf, uint byteOffset, uint stride, uint vertexIndex) {
  const uint base = (byteOffset + vertexIndex * stride) / 4u;
  return float2(buf[base + 0u], buf[base + 1u]);
}

uint readIndex(uint i) {
  if (cb.useIndices == 0u) {
    return i;
  }
  if (cb.indexIs32Bit != 0u) {
    return srcIndex.Load(cb.srcIndexOffset + i * 4u);
  }
  // 16-bit indices packed two-per-dword.
  const uint dword = srcIndex.Load(cb.srcIndexOffset + (i / 2u) * 4u);
  return (i & 1u) != 0u ? (dword >> 16u) : (dword & 0xFFFFu);
}

void writeVec3(RWStructuredBuffer<float> buf, uint byteOffset, uint stride, uint vertexIndex, float3 v) {
  const uint base = (byteOffset + vertexIndex * stride) / 4u;
  buf[base + 0u] = v.x;
  buf[base + 1u] = v.y;
  buf[base + 2u] = v.z;
}

void writeVec2(RWStructuredBuffer<float> buf, uint byteOffset, uint stride, uint vertexIndex, float2 v) {
  const uint base = (byteOffset + vertexIndex * stride) / 4u;
  buf[base + 0u] = v.x;
  buf[base + 1u] = v.y;
}

float sampleHeight(float2 uv) {
  float2 sampleUv = uv;
  if (cb.nearestFilter != 0u && cb.heightTexWidth > 0u && cb.heightTexHeight > 0u) {
    const float2 dim = float2(cb.heightTexWidth, cb.heightTexHeight);
    sampleUv = (floor(uv * dim) + 0.5f) / dim; // snap to texel center -> nearest/point sampling
  }
  return heightTexture.SampleLevel(heightSampler, sampleUv, 0).r;
}

// Displaced height (world-space offset magnitude) at the center of grid cell (gx,gy).
float cellHeightAt(float2 uvMin, float2 uvSize, float invN, uint gx, uint gy) {
  const float2 uvc = uvMin + float2((float(gx) + 0.5f) * invN, (float(gy) + 0.5f) * invN) * uvSize;
  return (sampleHeight(uvc) - cb.heightBias) * cb.displaceScale;
}

struct SrcVert {
  float3 pos;
  float3 nrm;
  float2 uv;
};

float2 interpUv(SrcVert a, SrcVert b, SrcVert c, float3 w) {
  return a.uv * w.x + b.uv * w.y + c.uv * w.z;
}
float3 interpPos(SrcVert a, SrcVert b, SrcVert c, float3 w) {
  return a.pos * w.x + b.pos * w.y + c.pos * w.z;
}
float3 interpNrm(SrcVert a, SrcVert b, SrcVert c, float3 w) {
  return normalize(a.nrm * w.x + b.nrm * w.y + c.nrm * w.z);
}

// Barycentric weights for grid node (i,j) of an N-subdivided triangle: w = (1 - (i+j)/N, i/N, j/N).
float3 gridWeights(uint i, uint j, float invN) {
  const float u = i * invN;
  const float v = j * invN;
  return float3(1.0f - u - v, u, v);
}

void writeTri(uint outTriIndex, float3 p0, float3 p1, float3 p2,
              float2 uv0, float2 uv1, float2 uv2, float3 fallbackN) {
  float3 n = cross(p1 - p0, p2 - p0);
  const float len = length(n);
  n = len > 1e-8f ? (n / len) : fallbackN;
  const uint v0 = outTriIndex * 3u;
  writeVec3(dstPosition, cb.dstPositionOffset, cb.dstPositionStride, v0 + 0u, p0);
  writeVec3(dstPosition, cb.dstPositionOffset, cb.dstPositionStride, v0 + 1u, p1);
  writeVec3(dstPosition, cb.dstPositionOffset, cb.dstPositionStride, v0 + 2u, p2);
  writeVec3(dstNormal, cb.dstNormalOffset, cb.dstNormalStride, v0 + 0u, n);
  writeVec3(dstNormal, cb.dstNormalOffset, cb.dstNormalStride, v0 + 1u, n);
  writeVec3(dstNormal, cb.dstNormalOffset, cb.dstNormalStride, v0 + 2u, n);
  writeVec2(dstTexcoord, cb.dstTexcoordOffset, cb.dstTexcoordStride, v0 + 0u, uv0);
  writeVec2(dstTexcoord, cb.dstTexcoordOffset, cb.dstTexcoordStride, v0 + 1u, uv1);
  writeVec2(dstTexcoord, cb.dstTexcoordOffset, cb.dstTexcoordStride, v0 + 2u, uv2);
}

// Barycentric weights (wa,wb,wc) of a UV-space point p within the triangle whose UV corners are
// (a,b,c). Signed division by the triangle's UV area makes the weights positive INSIDE regardless of
// winding. Used to voxelize in TEXTURE space rather than the triangle's own barycentric space.
float3 uvBary(float2 a, float2 b, float2 c, float2 p) {
  const float2 v0 = b - a;
  const float2 v1 = c - a;
  const float2 v2 = p - a;
  const float d = v0.x * v1.y - v1.x * v0.y;
  if (abs(d) < 1e-20f) {
    return float3(1.0f, 0.0f, 0.0f); // degenerate UV mapping
  }
  const float invd = 1.0f / d;
  const float wb = (v2.x * v1.y - v1.x * v2.y) * invd;
  const float wc = (v0.x * v2.y - v2.x * v0.y) * invd;
  return float3(1.0f - wb - wc, wb, wc);
}

// Top-left fill rule for a UV edge direction. A quad's two triangles traverse their shared diagonal in
// OPPOSITE directions, so exactly one classifies it as top-left — the deterministic tie-break that
// assigns each grid cell straddling the diagonal to exactly one triangle (no double coverage, no gap).
bool isTopLeft(float2 e) {
  return (e.y < 0.0f) || (e.y == 0.0f && e.x < 0.0f);
}

// Is a cell-center inside this triangle, applying the top-left rule to the three edges so diagonal
// cells are claimed once. \p w is uvBary of the center; tl* are the precomputed edge flags.
bool insideTopLeft(float3 w, bool tlA, bool tlB, bool tlC) {
  const float EPS = 1e-6f;
  const bool inA = (w.x > EPS) || (abs(w.x) <= EPS && tlA);
  const bool inB = (w.y > EPS) || (abs(w.y) <= EPS && tlB);
  const bool inC = (w.z > EPS) || (abs(w.z) <= EPS && tlC);
  return inA && inB && inC;
}

// Collapse an unused cell slot to zero-area triangles (no RT hits, culled) so leftover buffer slots
// never render garbage. Collapsed onto \p at (a real source vertex) rather than the origin, so the
// degenerates stay inside the mesh's existing bounds and don't inflate the BLAS AABB. Each cell owns a
// fixed [cellIndex*10 .. +9] range.
void emitDegenerate(uint cellIndex, float3 at) {
  const uint base = cellIndex * 10u;
  const float2 z2 = float2(0.0f, 0.0f);
  for (uint t = 0u; t < 10u; ++t) {
    writeTri(base + t, at, at, at, z2, z2, z2, float3(0.0f, 1.0f, 0.0f));
  }
}

// Emit one side wall (2 tris at [slot, slot+1]) of a voxel cell, but ONLY when this cell rises above its
// neighbor on that edge (a real step). The wall spans from the cell's top down to the NEIGHBOR's height
// (\p neighborOff = n * neighborHeight), not to the base — so there are no coincident/overlapping walls
// with the neighbor cell (which emits its own wall only when the roles reverse). When there is no step
// (neighbor >= this cell) the slot is filled with zero-area triangles the BLAS build discards.
void emitWall(uint slot, float3 baseA, float3 baseB, float3 topA, float3 topB,
              float3 neighborOff, bool doEmit, float2 uvA, float2 uvB, float3 n) {
  if (!doEmit) {
    const float2 z2 = float2(0.0f, 0.0f);
    writeTri(slot + 0u, baseA, baseA, baseA, z2, z2, z2, n);
    writeTri(slot + 1u, baseA, baseA, baseA, z2, z2, z2, n);
    return;
  }
  const float3 botA = baseA + neighborOff;
  const float3 botB = baseB + neighborOff;
  writeTri(slot + 0u, topA, topB, botB, uvA, uvB, uvB, n);
  writeTri(slot + 1u, topA, botB, botA, uvA, uvB, uvA, n);
}

// Emit one SQUARE voxel cell: a flat quad top raised by n*h, plus up to 4 side walls. A wall is emitted
// only where this cell steps above its neighbor (hS/hE/hN/hW = neighbor heights; block-edge neighbors are
// 0 = base). Emitting each step's wall exactly once (from the higher side) removes the ~half of wall
// geometry that used to be coincident full-height columns — cutting BLAS build cost for identical looks.
// Still 10 fixed slots [cellIndex*10 .. +9] (unused walls are zero-area) so output layout is deterministic.
void emitVoxelCell(uint cellIndex,
                   float3 b00, float3 b10, float3 b11, float3 b01, float3 n,
                   float h, float hS, float hE, float hN, float hW,
                   float2 uv00, float2 uv10, float2 uv11, float2 uv01) {
  const float3 off = n * h;
  const float3 t00 = b00 + off;
  const float3 t10 = b10 + off;
  const float3 t11 = b11 + off;
  const float3 t01 = b01 + off;
  const uint base = cellIndex * 10u;
  // Flat square top (2 tris).
  writeTri(base + 0u, t00, t10, t11, uv00, uv10, uv11, n);
  writeTri(base + 1u, t00, t11, t01, uv00, uv11, uv01, n);
  // Side walls, each only where this cell is higher than that neighbor (down to the neighbor's height).
  emitWall(base + 2u, b00, b10, t00, t10, n * hS, hS < h, uv00, uv10, n);  // south edge b00-b10
  emitWall(base + 4u, b10, b11, t10, t11, n * hE, hE < h, uv10, uv11, n);  // east  edge b10-b11
  emitWall(base + 6u, b11, b01, t11, t01, n * hN, hN < h, uv11, uv01, n);  // north edge b11-b01
  emitWall(base + 8u, b01, b00, t01, t00, n * hW, hW < h, uv01, uv00, n);  // west  edge b01-b00
}

// Largest power of two <= x (x >= 1). A power-of-two grid over a 2^k-texel sprite puts every cell edge
// exactly on a texel boundary, which is what makes the voxels line up with the texture's pixels.
uint largestPow2LE(uint x) {
  uint p = 1u;
  while ((p << 1u) <= x) {
    p <<= 1u;
  }
  return p;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint tri : SV_DispatchThreadID) {
  if (tri >= cb.srcTriangleCount) {
    return;
  }
  const uint Nalloc = max(cb.tessFactor, 1u); // slot budget the host allocated per source triangle

  const uint i0 = readIndex(tri * 3u + 0u);
  const uint i1 = readIndex(tri * 3u + 1u);
  const uint i2 = readIndex(tri * 3u + 2u);

  SrcVert a, b, c;
  a.pos = readVec3(srcPosition, cb.srcPositionOffset, cb.srcPositionStride, i0);
  b.pos = readVec3(srcPosition, cb.srcPositionOffset, cb.srcPositionStride, i1);
  c.pos = readVec3(srcPosition, cb.srcPositionOffset, cb.srcPositionStride, i2);
  a.nrm = readVec3(srcNormal, cb.srcNormalOffset, cb.srcNormalStride, i0);
  b.nrm = readVec3(srcNormal, cb.srcNormalOffset, cb.srcNormalStride, i1);
  c.nrm = readVec3(srcNormal, cb.srcNormalOffset, cb.srcNormalStride, i2);
  a.uv = readVec2(srcTexcoord, cb.srcTexcoordOffset, cb.srcTexcoordStride, i0);
  b.uv = readVec2(srcTexcoord, cb.srcTexcoordOffset, cb.srcTexcoordStride, i1);
  c.uv = readVec2(srcTexcoord, cb.srcTexcoordOffset, cb.srcTexcoordStride, i2);

  // Voxelize in TEXTURE space, not the triangle's barycentric space. The UV bounding box of a single
  // triangle of a UV quad already spans the full block sprite rect (any 3 of the 4 quad corners cover
  // both min and max in U and V), so an N x N grid over this box is axis-aligned to the texture — the
  // cells follow texel rows/columns instead of the triangle diagonal. That diagonal alignment was the
  // source of the residual "triangle angles"; square texel-aligned cells remove it.
  const float2 uvMin = min(a.uv, min(b.uv, c.uv));
  const float2 uvMax = max(a.uv, max(b.uv, c.uv));
  const float2 uvSize = uvMax - uvMin;

  // Grid resolution derived from the SPRITE'S TEXEL SIZE so voxels line up with the texture's pixels.
  // The sprite spans (uvSize * texDim) texels; we want one cell per texel, but capped at the host slot
  // budget (Nalloc) and snapped DOWN to a power of two. Because a vanilla sprite is 2^k texels and
  // starts on a texel boundary, a pow2 grid puts every cell edge exactly on a texel edge — no drift.
  // If the budget forces fewer cells than texels, each cell is a 2^m-texel block, still edge-aligned.
  const float spriteTexels = max(uvSize.x * float(cb.heightTexWidth),
                                 uvSize.y * float(cb.heightTexHeight));
  const uint desired = max(1u, (uint) round(spriteTexels));
  const uint N = largestPow2LE(min(Nalloc, desired)); // texel-aligned grid resolution actually used
  const float invN = 1.0f / float(N);

  // Precompute top-left flags for the three edges (edge opposite each vertex), for the diagonal tie-
  // break so each grid cell is owned by exactly one of the face's two triangles.
  const bool tlA = isTopLeft(c.uv - b.uv); // edge b->c, opposite a
  const bool tlB = isTopLeft(a.uv - c.uv); // edge c->a, opposite b
  const bool tlC = isTopLeft(b.uv - a.uv); // edge a->b, opposite c

  // This triangle owns ~half of the N*N grid (the quad's OTHER triangle owns the rest, split on the
  // diagonal by the top-left rule). So the host reserves only ceil(Nalloc^2/2)+Nalloc slots per triangle
  // and we SKIP un-owned cells entirely — no zero-area degenerate is emitted for them. That roughly
  // HALVES the tessellated geometry (BLAS memory + ray-traversal cost) for identical visuals; the small
  // reserved margin (+Nalloc) covers the diagonal cells the tie-break can assign to one side.
  const uint slotsPerTri = (Nalloc * Nalloc + 1u) / 2u + Nalloc;
  const uint outBase = tri * slotsPerTri;
  uint emitted = 0u;
  for (uint gy = 0u; gy < N; ++gy) {
    for (uint gx = 0u; gx < N; ++gx) {
      // Cell corner + center UVs on the sprite grid.
      const float2 uv00 = uvMin + float2(float(gx),      float(gy))      * invN * uvSize;
      const float2 uv11 = uvMin + float2(float(gx + 1u), float(gy + 1u)) * invN * uvSize;
      const float2 uv10 = float2(uv11.x, uv00.y);
      const float2 uv01 = float2(uv00.x, uv11.y);
      const float2 uvC  = 0.5f * (uv00 + uv11);

      // Only this triangle's cells (top-left rule on the diagonal); un-owned cells consume no slot.
      const float3 wC = uvBary(a.uv, b.uv, c.uv, uvC);
      if (!insideTopLeft(wC, tlA, tlB, tlC)) {
        continue;
      }
      if (emitted >= slotsPerTri) {
        break; // safety: never write past the reserved slots (the bound guarantees we don't get here)
      }
      const uint cellIndex = outBase + emitted;
      ++emitted;

      // Cell height + its 4 neighbor heights (block-edge neighbors = 0 = base, so the sprite border gets
      // a full wall). Emitting each step's wall only from the higher side removes coincident walls. The
      // face is planar, so mapping the cell's corner UVs back through barycentric gives correct surface
      // positions even where a corner extrapolates just past the triangle edge.
      const float3 nC = interpNrm(a, b, c, wC);
      const float h  = cellHeightAt(uvMin, uvSize, invN, gx, gy);
      const float hS = (gy > 0u)     ? cellHeightAt(uvMin, uvSize, invN, gx, gy - 1u) : 0.0f;
      const float hN = (gy + 1u < N) ? cellHeightAt(uvMin, uvSize, invN, gx, gy + 1u) : 0.0f;
      const float hW = (gx > 0u)     ? cellHeightAt(uvMin, uvSize, invN, gx - 1u, gy) : 0.0f;
      const float hE = (gx + 1u < N) ? cellHeightAt(uvMin, uvSize, invN, gx + 1u, gy) : 0.0f;
      const float3 b00 = interpPos(a, b, c, uvBary(a.uv, b.uv, c.uv, uv00));
      const float3 b10 = interpPos(a, b, c, uvBary(a.uv, b.uv, c.uv, uv10));
      const float3 b11 = interpPos(a, b, c, uvBary(a.uv, b.uv, c.uv, uv11));
      const float3 b01 = interpPos(a, b, c, uvBary(a.uv, b.uv, c.uv, uv01));

      emitVoxelCell(cellIndex, b00, b10, b11, b01, nC, h, hS, hE, hN, hW, uv00, uv10, uv11, uv01);
    }
  }

  // Zero any unused reserved slots so leftover buffer space never renders stale geometry.
  for (uint s = emitted; s < slotsPerTri; ++s) {
    emitDegenerate(outBase + s, a.pos);
  }
}
