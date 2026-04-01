// =============================================================================
// QubePixel — UploadWorker  (parallel CPU staging + main-thread GPU upload)
// =============================================================================
//
// Offloads CPU-heavy mesh data packing to a background thread while the
// render thread stays free for drawing.  GPU buffer creation and upload
// (create_buffer + write_buffer) happen on the main thread because they are
// lightweight in wgpu (just record commands, actual GPU work is async).
//
// Flow:
//   Main thread                    Worker thread (CPU only)
//   ──────────                    ──────────────────────────
//   submit(jobs)  ──────────────►  pack vertices/indices
//                                  into contiguous staging buffer
//   poll()  ◄──────────────────  send UploadPacked
//   create_buffer + write_buffer
//   insert into pipeline
// =============================================================================

use std::sync::mpsc;
use std::thread;
use std::time::Instant;
use crate::{debug_log, flow_debug_log, ext_debug_log};
use crate::screens::game_3d_pipeline::Vertex3D;

// ---------------------------------------------------------------------------
// UploadJob — raw mesh data sent from render thread to worker
// ---------------------------------------------------------------------------
pub struct UploadJob {
    pub key:       (i32, i32, i32),
    pub vertices:  Vec<Vertex3D>,
    pub indices:   Vec<u32>,
    pub aabb_min:  [f32; 3],
    pub aabb_max:  [f32; 3],
}

// ---------------------------------------------------------------------------
// UploadPacked — CPU-packed mesh data ready for GPU buffer creation
// ---------------------------------------------------------------------------
pub struct UploadPacked {
    pub key:          (i32, i32, i32),
    pub vertex_data:  Vec<u8>,
    pub index_data:   Vec<u8>,
    pub index_count:  u32,
    pub aabb_min:     [f32; 3],
    pub aabb_max:     [f32; 3],
}

// ---------------------------------------------------------------------------
// UploadWorker
// ---------------------------------------------------------------------------
pub struct UploadWorker {
    job_tx:    mpsc::Sender<UploadJob>,
    packed_rx: mpsc::Receiver<UploadPacked>,
    thread:    Option<thread::JoinHandle<()>>,
    pending:   usize,
}

impl UploadWorker {
    /// Spawn the background packing thread.  No GPU resources needed.
    pub fn new() -> Self {
        let (job_tx, job_rx)       = mpsc::channel::<UploadJob>();
        let (packed_tx, packed_rx) = mpsc::channel::<UploadPacked>();

        let handle = thread::Builder::new()
            .name("gpu-staging".into())
            .spawn(move || {
                let vert_stride = std::mem::size_of::<Vertex3D>();
                let idx_stride  = std::mem::size_of::<u32>();

                // Reusable staging buffer — avoids per-batch allocation
                let mut staging: Vec<u8> = Vec::with_capacity(4 * 1024 * 1024);
                let mut total_packed = 0usize;

                loop {
                    // Block until at least one job arrives
                    let first = match job_rx.recv() {
                        Ok(job) => job,
                        Err(_)  => break, // channel closed → shutdown
                    };

                    // Drain every additional pending job into a single batch
                    let mut batch = Vec::with_capacity(64);
                    batch.push(first);
                    while let Ok(job) = job_rx.try_recv() {
                        batch.push(job);
                    }

                    let t0        = Instant::now();
                    let batch_len = batch.len();

                    // ── Calculate total staging size ───────────────────────
                    let mut needed: usize = 0;
                    for job in &batch {
                        needed += job.vertices.len() * vert_stride;
                        needed += job.indices.len()  * idx_stride;
                    }

                    // Grow staging buffer only if necessary (amortised)
                    if staging.capacity() < needed {
                        let grow = (needed * 2).max(4 * 1024 * 1024);
                        staging.reserve(grow - staging.len());
                    }
                    staging.clear();
                    staging.resize(needed, 0);

                    // ── Pack each job's data into staging ──────────────────
                    let mut offset = 0usize;
                    let mut packed = 0usize;

                    for job in batch {
                        if job.vertices.is_empty() || job.indices.is_empty() {
                            continue;
                        }

                        let vert_bytes = job.vertices.len() * vert_stride;
                        let idx_bytes  = job.indices.len()  * idx_stride;

                        // Copy vertex data
                        let vert_off = offset;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                job.vertices.as_ptr() as *const u8,
                                staging[vert_off..].as_mut_ptr(),
                                vert_bytes,
                            );
                        }
                        offset += vert_bytes;

                        // Copy index data
                        let idx_off = offset;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                job.indices.as_ptr() as *const u8,
                                staging[idx_off..].as_mut_ptr(),
                                idx_bytes,
                            );
                        }
                        offset += idx_bytes;

                        // Extract packed slices as owned Vecs
                        let vertex_data: Vec<u8> =
                            staging[vert_off..vert_off + vert_bytes].to_vec();
                        let index_data: Vec<u8> =
                            staging[idx_off..idx_off + idx_bytes].to_vec();

                        packed += 1;
                        total_packed += 1;

                        let _ = packed_tx.send(UploadPacked {
                            key:         job.key,
                            vertex_data,
                            index_data,
                            index_count: job.indices.len() as u32,
                            aabb_min:    job.aabb_min,
                            aabb_max:    job.aabb_max,
                        });
                    }

                    let us = t0.elapsed().as_micros();

                    flow_debug_log!(
                        "UploadWorker", "thread",
                        "[PERF] batch={} packed={} total={} time={:.2}ms staging={:.1}KB",
                        batch_len, packed, total_packed,
                        us as f64 / 1000.0,
                        needed as f64 / 1024.0
                    );
                }

                debug_log!("UploadWorker", "thread", "Background thread exiting");
            })
            .expect("Failed to spawn gpu-staging thread");

        debug_log!("UploadWorker", "new", "Spawned staging worker thread");

        Self {
            job_tx,
            packed_rx,
            thread: Some(handle),
            pending: 0,
        }
    }

    // -------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------

    /// Submit a batch of mesh jobs for async CPU packing.  Non-blocking.
    pub fn submit(&mut self, jobs: Vec<UploadJob>) {
        self.pending += jobs.len();
        for job in jobs {
            let _ = self.job_tx.send(job);
        }
    }

    /// Non-blocking poll for packed results.  Call once per frame.
    pub fn poll(&mut self) -> Vec<UploadPacked> {
        let mut results = Vec::new();
        while let Ok(r) = self.packed_rx.try_recv() {
            self.pending -= 1;
            results.push(r);
        }
        results
    }

    pub fn pending_count(&self) -> usize { self.pending }
    #[allow(dead_code)]
    pub fn is_busy(&self)     -> bool   { self.pending > 0 }
}

impl Drop for UploadWorker {
    fn drop(&mut self) {
        debug_log!("UploadWorker", "drop", "Shutting down staging worker");
        let _ = self.thread.take().and_then(|h| Some(h.join()));
    }
}