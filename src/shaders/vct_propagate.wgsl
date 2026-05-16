// =============================================================================
// QubePixel — VCT Light Propagation Compute Shader
// =============================================================================
// Flood-fill propagation: each air voxel takes the MAX of its 6 neighbours
// and decays it.  Solid voxels block propagation (write 0).  Emissive solids
// continuously re-inject their emission to stay bright.
//
// MAX-based (vs. averaging) preserves energy along propagation paths — no
// 1/r² loss, so lights illuminate the full PROPAGATION_STEPS distance (~64 blocks).
//
// Ping-ponged: reads from radiance_in, writes to radiance_out.
//
// Shared memory tile for Z-neighbors: Z-stride in a 128^3 texture is 128KB,
// causing L1 cache misses on every ±Z access.  We load the workgroup's 8×8×4
// slab plus the two adjacent Z-slices into workgroup shared memory (6 Z-layers
// total), so all Z-neighbor reads hit shared memory instead of global memory.

struct PropagateParams {
    volume_size: vec4<u32>,   // x = side length (128)
    config:      vec4<f32>,   // x = decay (0..1), y = iteration, zw = 0
};

@group(0) @binding(0) var<uniform> params:       PropagateParams;
@group(0) @binding(1) var           voxel_data:   texture_3d<f32>;   // opacity in .a
@group(0) @binding(2) var           voxel_emission: texture_3d<f32>; // emission source
@group(0) @binding(3) var           radiance_in:  texture_3d<f32>;   // read prev iteration
@group(0) @binding(4) var           radiance_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var           voxel_tint:   texture_3d<f32>;   // glass tint/opacity

// Glass voxel marker (matches VOXEL_ALPHA_GLASS = 180/255 in voxel_volume.rs).
const GLASS_ALPHA_MIN: f32 = 0.55;
const GLASS_ALPHA_MAX: f32 = 0.85;

// Shared memory tile: 8×8 XY slab, 6 Z-layers (1 border + 4 real + 1 border).
// Index = lx + ly*8 + lz_shifted*64  where lz_shifted ∈ [0..5].
var<workgroup> z_tile: array<vec4<f32>, 8u * 8u * 6u>;  // 384 × 16B = 6 KB

@compute @workgroup_size(8, 8, 4)
fn main(
    @builtin(global_invocation_id)   gid: vec3<u32>,
    @builtin(local_invocation_id)    lid: vec3<u32>,
) {
    let size  = params.volume_size.x;
    let isize = i32(size);

    let pos = vec3<i32>(gid);

    // ── Load Z-tile into workgroup shared memory ─────────────────────────────
    // Threads fill their own slot (lid.z + 1 = shifted index 1..4).
    // Threads at the Z boundary also fill the border slots (shifted 0 and 5).
    let base_xy = lid.x + lid.y * 8u;
    let sz      = lid.z + 1u;             // shifted Z index for this thread (1..=4)

    // Current thread's own radiance value
    let self_rad = textureLoad(radiance_in, pos, 0);
    z_tile[base_xy + sz * 64u] = self_rad;

    // Bottom Z border: only the threads in the first Z-layer load it.
    if (lid.z == 0u) {
        if (pos.z > 0) {
            z_tile[base_xy] = textureLoad(radiance_in, pos - vec3<i32>(0, 0, 1), 0);
        } else {
            z_tile[base_xy] = vec4<f32>(0.0);
        }
    }

    // Top Z border: only the threads in the last Z-layer load it.
    if (lid.z == 3u) {
        if (pos.z + 1 < isize) {
            z_tile[base_xy + 5u * 64u] = textureLoad(radiance_in, pos + vec3<i32>(0, 0, 1), 0);
        } else {
            z_tile[base_xy + 5u * 64u] = vec4<f32>(0.0);
        }
    }

    workgroupBarrier();

    // ── Early-out bounds check (after barrier so all threads participate) ────
    if (any(gid >= vec3<u32>(size))) { return; }

    let data    = textureLoad(voxel_data, pos, 0);
    let opacity = data.a;
    let em      = textureLoad(voxel_emission, pos, 0);
    let is_glass = opacity >= GLASS_ALPHA_MIN && opacity <= GLASS_ALPHA_MAX;

    // --- Fully opaque solid voxel: keep emission, block propagation ---
    if (opacity > 0.85) {
        if (em.a > 0.001) {
            textureStore(radiance_out, pos, vec4<f32>(em.rgb * em.a * 2.0, 1.0));
        } else {
            textureStore(radiance_out, pos, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        }
        return;
    }

    // --- Model block: keep emission if bright, but let light propagate ---
    if (opacity > 0.2 && opacity < 0.5) {
        if (em.a > 0.001) {
            textureStore(radiance_out, pos, vec4<f32>(em.rgb * em.a * 2.0, 1.0));
        }
    }

    // --- Air / glass voxel: MAX-based flood fill ---
    let decay = params.config.x;

    var best = vec3<f32>(0.0);

    // X-neighbors: stride=1 texel — cache-friendly, read from global texture
    if (pos.x + 1 < isize) {
        best = max(best, textureLoad(radiance_in, pos + vec3<i32>(1, 0, 0), 0).rgb);
    }
    if (pos.x - 1 >= 0) {
        best = max(best, textureLoad(radiance_in, pos + vec3<i32>(-1, 0, 0), 0).rgb);
    }
    // Y-neighbors: stride=128 texels — still from global texture
    if (pos.y + 1 < isize) {
        best = max(best, textureLoad(radiance_in, pos + vec3<i32>(0, 1, 0), 0).rgb);
    }
    if (pos.y - 1 >= 0) {
        best = max(best, textureLoad(radiance_in, pos + vec3<i32>(0, -1, 0), 0).rgb);
    }
    // Z-neighbors: stride=128×128 texels (128KB!) — read from shared memory
    best = max(best, z_tile[base_xy + (sz + 1u) * 64u].rgb);  // +Z
    best = max(best, z_tile[base_xy + (sz - 1u) * 64u].rgb);  // -Z

    var result = best * decay;

    // Glass tinting
    if (is_glass) {
        let tint = textureLoad(voxel_tint, pos, 0);
        let op   = tint.a;
        let mult = mix(vec3<f32>(1.0), tint.rgb, op);
        result   = result * mult * (1.0 - 0.5 * op);
    }

    textureStore(radiance_out, pos, vec4<f32>(result, 0.0));
}
