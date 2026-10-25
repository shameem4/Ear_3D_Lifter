//! numpy's default_rng, reproduced: SeedSequence -> PCG64 (XSL-RR 128/64),
//! and the Generator methods the skin draws with (random, uniform,
//! standard_normal by numpy's ziggurat). Same seed, same numbers as
//! np.random.default_rng(seed), so the Rust skin is the Python skin.

use crate::ziggurat::{FI, KI, WI};

const MULT: u128 = 0x2360ED051FC65DA44385DF649FCCF645;

pub struct Rng { state: u128, inc: u128 }

/// SeedSequence(entropy).generate_state(4, uint64), for a non-negative int seed.
fn seed_state(seed: u64) -> [u64; 4] {
    const INIT_A: u32 = 0x43b0d7e5;
    const MULT_A: u32 = 0x931e8875;
    const INIT_B: u32 = 0x8b51f9dd;
    const MULT_B: u32 = 0x58f38ded;
    const MIX_L: u32 = 0xca01f9dd;
    const MIX_R: u32 = 0x4973f715;
    let mut ent: Vec<u32> = Vec::new();
    let mut s = seed;
    while s > 0 { ent.push(s as u32); s >>= 32; }
    if ent.is_empty() { ent.push(0); }
    let mut hc = INIT_A;
    let mut hashmix = |v: u32| { let mut v = v ^ hc; hc = hc.wrapping_mul(MULT_A); v = v.wrapping_mul(hc); v ^ (v >> 16) };
    let mix = |x: u32, y: u32| { let r = MIX_L.wrapping_mul(x).wrapping_sub(MIX_R.wrapping_mul(y)); r ^ (r >> 16) };
    let mut pool = [0u32; 4];
    for i in 0..4 { pool[i] = hashmix(if i < ent.len() { ent[i] } else { 0 }); }
    for src in 0..4 { for dst in 0..4 { if src != dst { let h = hashmix(pool[src]); pool[dst] = mix(pool[dst], h); } } }
    for &e in ent.iter().skip(4) { for dst in 0..4 { let h = hashmix(e); pool[dst] = mix(pool[dst], h); } }
    let mut hb = INIT_B;
    let mut w = [0u32; 8];
    for (i, x) in w.iter_mut().enumerate() {
        let mut d = pool[i % 4] ^ hb;
        hb = hb.wrapping_mul(MULT_B);
        d = d.wrapping_mul(hb);
        *x = d ^ (d >> 16);
    }
    [0, 1, 2, 3].map(|k| w[2 * k] as u64 | (w[2 * k + 1] as u64) << 32)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let s = seed_state(seed);
        let initstate = (s[0] as u128) << 64 | s[1] as u128;
        let initseq = (s[2] as u128) << 64 | s[3] as u128;
        let mut r = Rng { state: 0, inc: initseq << 1 | 1 };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }
    fn step(&mut self) { self.state = self.state.wrapping_mul(MULT).wrapping_add(self.inc); }
    pub fn u64(&mut self) -> u64 {
        self.step();
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((hi >> 58) as u32)
    }
    /// Uniform in [0, 1): Generator.random().
    pub fn random(&mut self) -> f64 { (self.u64() >> 11) as f64 * (1.0 / 9007199254740992.0) }
    /// Generator.uniform(lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * self.random() }
    /// Generator.standard_normal(): numpy's ziggurat.
    pub fn normal(&mut self) -> f64 {
        const R: f64 = 3.6541528853610088;
        const INV_R: f64 = 0.27366123732975828;
        loop {
            let mut r = self.u64();
            let idx = (r & 0xff) as usize;
            r >>= 8;
            let sign = r & 1;
            let rabs = (r >> 1) & 0x000fffffffffffff;
            let mut x = rabs as f64 * WI[idx];
            if sign == 1 { x = -x; }
            if rabs < KI[idx] { return x; }
            if idx == 0 {
                loop {
                    let xx = -INV_R * (-self.random()).ln_1p();
                    let yy = -(-self.random()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 1 == 1 { -(R + xx) } else { R + xx };
                    }
                }
            } else if (FI[idx - 1] - FI[idx]) * self.random() + FI[idx] < (-0.5 * x * x).exp() {
                return x;
            }
        }
    }
}
