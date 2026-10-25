//! A minimal executor for hosts whose calls complete immediately (Python): the
//! pipeline future is polled until it finishes, never actually waiting.

use std::future::Future;
use std::task::{Context, Poll, Waker};

pub fn block_on<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

/// Little-endian bytes <-> f64/f32/u32 slices: how arrays cross both bindings.
pub fn f64s(b: &[u8]) -> Vec<f64> { b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }
pub fn f32s(b: &[u8]) -> Vec<f32> { b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
pub fn u32s(b: &[u8]) -> Vec<u32> { b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect() }
pub fn bytes_f64(v: &[f64]) -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() }

use crate::mesh::V3;
use crate::pipeline::Assets;

/// Assets from flat arrays: table (V x 55), angles (V x 2), template, canon.
pub fn assets(table: &[f64], angles: &[f64], template: &[f64], canon: &[f64]) -> Assets {
    let v3 = |a: &[f64]| a.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect::<Vec<V3>>();
    Assets {
        table: table.chunks_exact(55).map(|c| c.try_into().unwrap()).collect(),
        angles: angles.chunks_exact(2).map(|c| [c[0], c[1]]).collect(),
        template: v3(template),
        canon: v3(canon),
    }
}
