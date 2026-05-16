# Key Invariants

**Silent GPU corruption prevention — read before touching shaders or GPU structs:**

- **WGSL struct layout must exactly match Rust `bytemuck` types.** Any change to `RadianceInterval`, cascade uniforms, or block LUT structs in Rust must be reflected in all `.wgsl` files, and vice versa.
- Voxel texture is 128³ world-space blocks centered on camera. Block ID `0xFFFF` = unloaded air (transparent).
- Block LUT index 0 is reserved for air (fully transparent, no emission).
