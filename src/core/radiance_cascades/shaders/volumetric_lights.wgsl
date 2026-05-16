// =============================================================================
// QubePixel — Volumetric Light Shader
// =============================================================================
//
// Two-component visual effect for emissive blocks:
//
//   1. HALO  — a soft Gaussian glow sphere rendered as a camera-facing billboard
//              quad around the block centre.  Uses additive blending.
//
//   2. RAY   — thin god-ray quads radiating outward from the block centre in N
//              configurable directions.  Gradient bright-at-source → transparent-
//              at-tip; slow time-based wobble for a living flame look.
//
// Both components share the same uniform layout but are drawn in two separate
// draw calls (halo pass first, then ray pass).  The depth buffer is read (no
// write) so both passes are occluded by solid geometry.
//
// GPU struct alignment (std140 / WGSL storage):
//   VolumetricUniforms  — 128 bytes (4 × mat4 / vec4)
//   HaloInstance        — 32 bytes  (2 × vec4)
//   RayInstance         — 64 bytes  (4 × vec4)
// =============================================================================

// ---------------------------------------------------------------------------
// Shared uniforms (updated once per frame)
// ---------------------------------------------------------------------------

struct VolumetricUniforms {
    view_proj:   mat4x4<f32>,   // [0..64]
    cam_right:   vec4<f32>,     // [64..80]   .xyz = world-space camera-right
    cam_up:      vec4<f32>,     // [80..96]   .xyz = world-space camera-up
    time_params: vec4<f32>,     // [96..112]  .x = elapsed seconds
    _pad:        vec4<f32>,     // [112..128]
};

@group(0) @binding(0) var<uniform> u: VolumetricUniforms;

// ---------------------------------------------------------------------------
// PASS 1 — Halo (soft additive billboard)
// ---------------------------------------------------------------------------

// One entry per visible emissive block with halo_enabled = true.
struct HaloInstance {
    world_pos_radius: vec4<f32>,   // .xyz = world centre, .w = halo_radius
    color_intensity:  vec4<f32>,   // .rgb = emission colour, .w = halo_intensity
};

@group(1) @binding(0) var<storage, read> halo_instances: array<HaloInstance>;

struct HaloVOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,     // -1..1 in both axes
    @location(1)       color_i:  vec4<f32>,     // .rgb = color, .w = intensity
};

// Quad corners in local space (-1,-1) … (1,1)
fn halo_corner(vid: u32) -> vec2<f32> {
    // vid: 0 = (-1,-1), 1 = (1,-1), 2 = (1,1), 3 = (-1,1)
    let x = select(-1.0, 1.0, (vid & 1u) != 0u);
    let y = select(-1.0, 1.0, (vid & 2u) != 0u);
    return vec2<f32>(x, y);
}

@vertex
fn vs_halo(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32,
) -> HaloVOut {
    let inst   = halo_instances[iid];
    let pos    = inst.world_pos_radius.xyz;
    let radius = inst.world_pos_radius.w;

    let uv     = halo_corner(vid);

    // Billboard: expand in camera-right and camera-up directions
    let world_pos = pos
        + u.cam_right.xyz * uv.x * radius
        + u.cam_up.xyz    * uv.y * radius;

    var out: HaloVOut;
    out.clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.uv       = uv;
    out.color_i  = inst.color_intensity;
    return out;
}

@fragment
fn fs_halo(in: HaloVOut) -> @location(0) vec4<f32> {
    // Radial distance from billboard centre (0 = centre, 1 = edge)
    let r = length(in.uv);
    if r > 1.0 { discard; }

    // Gaussian falloff: smooth halo, nearly zero at r = 1
    let falloff = exp(-3.0 * r * r);
    let alpha   = falloff * in.color_i.w;

    // Premultiplied alpha (for additive blend: src = SrcAlpha, dst = One)
    return vec4<f32>(in.color_i.rgb * alpha, alpha);
}

// ---------------------------------------------------------------------------
// PASS 2 — God rays (thin quads radiating outward from block centre)
// ---------------------------------------------------------------------------

// One entry per ray (N rays per emissive block — packed by CPU).
struct RayInstance {
    origin_len:   vec4<f32>,   // .xyz = ray origin (block centre), .w = ray length
    dir_width:    vec4<f32>,   // .xyz = ray direction (normalised), .w = ray width
    color_int:    vec4<f32>,   // .rgb = colour, .w = intensity
    falloff_time: vec4<f32>,   // .x = falloff exponent, .y = per-ray phase offset
};

@group(1) @binding(0) var<storage, read> ray_instances: array<RayInstance>;

struct RayVOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       t:        f32,       // 0 = origin, 1 = tip
    @location(1)       color_i:  vec4<f32>,
    @location(2)       falloff:  f32,
};

// Build a right-vector perpendicular to `dir` and in the plane of `dir`+camera.
// If dir is nearly parallel to cam_right we fall back to cam_up.
fn ray_right(dir: vec3<f32>, cam_right: vec3<f32>, cam_up: vec3<f32>) -> vec3<f32> {
    let d_dot_r = abs(dot(dir, cam_right));
    let ref_vec = select(cam_right, cam_up, d_dot_r > 0.9);
    return normalize(cross(dir, ref_vec));
}

@vertex
fn vs_ray(
    @builtin(vertex_index)   vid: u32,
    @builtin(instance_index) iid: u32,
) -> RayVOut {
    let inst   = ray_instances[iid];
    let origin = inst.origin_len.xyz;
    let length = inst.origin_len.w;
    let dir    = normalize(inst.dir_width.xyz);
    let width  = inst.dir_width.w;

    // Gentle time-based wobble (makes rays look alive)
    let phase   = inst.falloff_time.y;
    let wobble  = sin(u.time_params.x * 1.3 + phase) * 0.06;
    let wobble2 = cos(u.time_params.x * 0.9 + phase) * 0.04;

    // Build TBN: forward = dir, right = perpendicular to dir facing camera
    let cam_r   = u.cam_right.xyz;
    let cam_u   = u.cam_up.xyz;
    let right   = ray_right(dir, cam_r, cam_u);

    // Quad layout (vid 0..3):
    //   0 = base-left   (t=0, x=-0.5)
    //   1 = base-right  (t=0, x= 0.5)
    //   2 = tip-right   (t=1, x= 0.5)
    //   3 = tip-left    (t=1, x=-0.5)
    let local_x = select(-0.5, 0.5, (vid & 1u) != 0u);
    let local_t = select(0.0, 1.0, (vid & 2u) != 0u);

    // Taper: slightly wider at base, zero at tip
    let taper_w = width * (1.0 - local_t * 0.6);

    let world_pos = origin
        + dir   * (local_t * length)
        + right * (local_x * taper_w + wobble * local_t)
        + cam_u * (wobble2  * local_t);

    var out: RayVOut;
    out.clip_pos = u.view_proj * vec4<f32>(world_pos, 1.0);
    out.t        = local_t;
    out.color_i  = inst.color_int;
    out.falloff  = inst.falloff_time.x;
    return out;
}

@fragment
fn fs_ray(in: RayVOut) -> @location(0) vec4<f32> {
    // Gradient: full intensity at base, fades to 0 at tip
    let alpha = pow(max(1.0 - in.t, 0.0), in.falloff) * in.color_i.w;
    if alpha < 0.003 { discard; }

    return vec4<f32>(in.color_i.rgb * alpha, alpha);
}
