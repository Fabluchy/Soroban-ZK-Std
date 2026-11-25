//! PLONK KZG Evaluation Proof Verifier (Stellar Integration Layer).
//!
//! This module wires the pure-math [`kzg_eval_proof_points`] function from
//! `soroban-zk-core` to the Soroban `bn254_multi_pairing_check` host function
//! via the existing [`pairing_check`] abstraction.
//!
//! ## Protocol
//!
//! The final KZG check in PLONK verification reduces to a two-pair equation:
//!
//! ```text
//! e(W_ζ + u·W_{ζω},  [τ]₂)
//! · e(-(ζ·W_ζ + u·ζω·W_{ζω} + F - [E]₁),  G₂)
//! == 1
//! ```
//!
//! where:
//! - `[τ]₂` is the SRS G2 element (the trusted-setup key)
//! - `G₂` is the BN254 G2 generator
//! - `F` is the batched polynomial commitment (see [`KzgEvalProofInputs`])
//! - `[E]₁` is the batched evaluation encoded as a G1 point
//!
//! [`kzg_eval_proof_points`] handles all G1 arithmetic and returns the two
//! G1 points for the equation above.  This module pairs them against the
//! appropriate G2 points using the Soroban host's native pairing engine.
//!
//! ## Security Notes
//!
//! - All input scalars are validated against the BN254 scalar field modulus
//!   before any curve operation (inside [`kzg_eval_proof_points`]).
//! - G1 points coming from the proof (`w_zeta`, `w_zeta_omega`, and the
//!   commitments) **must** be validated by the caller before constructing
//!   [`KzgEvalProofInputs`].  Use [`crate::host::g1_from_host_bytes`] or
//!   check [`soroban_zk_core::Bn254::is_valid_g1_subgroup`] directly.
//! - G2 points (`srs_g2`, generator) are validated inside [`pairing_check`].

use ethnum::u256;
use soroban_sdk::Env;
use soroban_zk_core::{
    barycentric_pi_evaluation, evaluate_linearization, kzg_eval_proof_points,
    plonk_batch_verify_points, split_quotient_commitment, Bn254, CommitEvaluations, DomainParams,
    G1Affine, KzgEvalProofInputs, PlonkTranscript, SelectorEvaluations, ZkError,
};

use crate::groth16::g2_from_bytes;
use crate::pairing::{g1_to_bytes, pairing_check, validate_g2_coords, G2Affine};

/// Re-exports the canonical proof type so callers can reach
/// [`PlonkProof::from_bytes`] / [`PlonkProof::to_bytes`] through this module.
pub use soroban_zk_core::PlonkProof;

#[inline(always)]
fn is_g1_identity(p: G1Affine) -> bool {
    p.x == u256::from(0u8) && p.y == u256::from(0u8)
}

/// Evaluates `e(lhs, srs_g2) · e(-rhs, G₂) == 1`, handling identity G1 terms
/// (`e(𝒪, Q) = 1`) cleanly after validating `srs_g2`.
fn check_kzg_pairing_equation(
    env: &Env,
    lhs: G1Affine,
    rhs: G1Affine,
    srs_g2: G2Affine,
) -> Result<bool, ZkError> {
    if !validate_g2_coords(&srs_g2) {
        return Err(ZkError::InvalidInput);
    }
    let g2_gen = G2Affine::generator();
    match (is_g1_identity(lhs), is_g1_identity(rhs)) {
        (true, true) => Ok(true),
        (false, true) => pairing_check(env, &[(lhs, srs_g2)]),
        (true, false) => pairing_check(env, &[(neg_g1(rhs), g2_gen)]),
        (false, false) => pairing_check(env, &[(lhs, srs_g2), (neg_g1(rhs), g2_gen)]),
    }
}

/// Negates a G1Affine point: `(x, y) -> (x, -y mod Fq)`.
///
/// Returns the identity unchanged.
#[inline(always)]
fn neg_g1(p: G1Affine) -> G1Affine {
    if p.x == u256::from(0u8) && p.y == u256::from(0u8) {
        p
    } else {
        G1Affine {
            x: p.x,
            y: Bn254::sub_fq(u256::from(0u8), p.y),
        }
    }
}

/// Verifies the batched KZG evaluation proofs for PLONK at `ζ` and `ζω`.
///
/// This is the entry point for the "opening check" step of a PLONK verifier.
/// It performs the full two-pair pairing equation using the Soroban
/// `bn254_multi_pairing_check` host function (CAP-0075).
///
/// # Arguments
///
/// * `env` – Soroban execution environment (needed for the pairing host call).
/// * `inputs` – All evaluations, commitments, and opening proofs (see
///   [`KzgEvalProofInputs`]).
/// * `srs_g2` – The SRS G2 element `[τ]₂` from the trusted setup.  In
///   practice this is stored in the verifying key.
///
/// # Returns
///
/// * `Ok(true)` if the pairing equation holds (proof accepted).
/// * `Ok(false)` if the pairing equation fails (proof rejected).
/// * `Err(ZkError::InvalidInput)` / `Err(ZkError::InvalidFieldElement)` on
///   malformed inputs (propagated from [`kzg_eval_proof_points`]).
/// * `Err(ZkError::HostError)` if the host pairing call fails unexpectedly.
///
/// # Gas Guidance
///
/// This call performs exactly two pairings via `bn254_multi_pairing_check`.
/// At Protocol 25 rates each pair costs ~40 M instructions; total budget for
/// this call is ~80 M instructions.  The G1 scalar-mul operations in
/// [`kzg_eval_proof_points`] add roughly 10–30 M depending on the batch size.
///
/// # Example
///
/// ```rust,ignore
/// use soroban_zk_std::plonk_kzg::verify_plonk_kzg;
/// use soroban_zk_core::KzgEvalProofInputs;
/// use soroban_zk_std::pairing::G2Affine;
///
/// let accepted = verify_plonk_kzg(&env, &inputs, srs_g2)?;
/// assert!(accepted, "proof verification failed");
/// ```
pub fn verify_plonk_kzg(
    env: &Env,
    inputs: &KzgEvalProofInputs<'_>,
    srs_g2: G2Affine,
) -> Result<bool, ZkError> {
    // 1. Compute the two G1 points from the batched polynomial arithmetic.
    let (lhs, rhs) = kzg_eval_proof_points(inputs)?;

    // 2. Check:  e(lhs, srs_g2) · e(-rhs, G2_gen) == 1
    check_kzg_pairing_equation(env, lhs, rhs, srs_g2)
}

/// Verifies `M` PLONK proofs with a **single** pairing call (Issue #427).
///
/// Each entry of `inputs` is a full [`KzgEvalProofInputs`] for one proof (the
/// same bundle consumed by [`verify_plonk_kzg`]). Instead of running one
/// two-pair `bn254_multi_pairing_check` per proof (`2·M` pairings total), the
/// per-proof `(lhs_j, rhs_j)` points are combined with random batch scalars
/// `rho_j` derived from `batch_seed` and folded into a single two-pair check:
///
/// ```text
/// e( sum_j rho_j·lhs_j , [tau]_2 ) · e( -sum_j rho_j·rhs_j , G₂ ) == 1
/// ```
///
/// This collapses the dominant on-chain cost (pairings) from `2·M` to `2`, so
/// verifying a batch is far cheaper than verifying each proof independently.
/// The G1 arithmetic is delegated to
/// [`plonk_batch_verify_points`] (allocation-free).
///
/// # Arguments
///
/// * `env` – Soroban execution environment (needed for the pairing host call).
/// * `inputs` – One [`KzgEvalProofInputs`] bundle per proof. Must be
///   non-empty.
/// * `batch_seed` – 32-byte seed used to derive the batch scalars `rho_j`.
/// * `srs_g2` – The SRS G2 element `[tau]₂` from the trusted setup.
///
/// # Returns
///
/// * `Ok(true)` if the aggregated pairing holds (all proofs accepted).
/// * `Ok(false)` if it fails (at least one proof is invalid).
/// * `Err(...)` on malformed inputs (empty batch, out-of-range scalars, or a
///   per-proof bundle failing validation) or a host pairing failure.
///
/// # Security
///
/// `batch_seed` **must** be unpredictable to the prover and not controlled by
/// it — e.g. derived from a transcript that binds every proof in the batch, or
/// an on-chain randomness beacon. A fixed or prover-chosen seed lets a malicious
/// prover cheat the randomized combination and breaks soundness.
///
/// As with [`verify_plonk_kzg`], the G1 points inside each bundle
/// (`w_zeta`, `w_zeta_omega`, commitments) **must** be subgroup-validated by
/// the caller before constructing [`KzgEvalProofInputs`].
///
/// # Example
///
/// ```rust,ignore
/// use soroban_zk_std::plonk_kzg::verify_plonk_kzg_batch;
///
/// let accepted = verify_plonk_kzg_batch(&env, &inputs, &batch_seed, srs_g2)?;
/// assert!(accepted, "batch verification failed");
/// ```
pub fn verify_plonk_kzg_batch(
    env: &Env,
    inputs: &[KzgEvalProofInputs<'_>],
    batch_seed: &[u8; 32],
    srs_g2: G2Affine,
) -> Result<bool, ZkError> {
    // 1. Fold every proof into the two batched G1 points (single accumulator
    //    pass, no heap allocation).
    let (lhs, rhs) = plonk_batch_verify_points(inputs, batch_seed)?;

    // 2. One two-pair check replaces the M individual checks:
    //        e(lhs, [tau]_2) · e(-rhs, G₂) == 1
    check_kzg_pairing_equation(env, lhs, rhs, srs_g2)
}

// ============================================================================
// End-to-end PLONK verifier
// ============================================================================

/// Canonical verifying key for [`verify_plonk_full`].
///
/// Holds the trusted-setup `[tau]_2`, the three permutation-polynomial
/// commitments (needed to batch the openings at `zeta`), and the evaluation
/// domain parameters (`n`, `omega`, `k1`, `k2`).
///
/// The selector evaluations at the derived `zeta` are *not* part of the key:
/// they are supplied per proof to [`verify_plonk_full`], so callers must take
/// them from the preprocessed circuit description (or from the on-chain copy of
/// it), never from the untrusted proof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlonkVerifyingKey {
    /// Trusted-setup G2 element `[tau]_2`.
    pub srs_g2: G2Affine,
    /// Commitments to `sigma_1, sigma_2, sigma_3`.
    pub s_sigma_commitments: [G1Affine; 3],
    /// Generator of the `n`-element evaluation domain.
    pub omega: u256,
    /// Domain size (a power of two).
    pub n: u32,
    /// First coset shift for the permutation argument.
    pub k1: u256,
    /// Second coset shift for the permutation argument.
    pub k2: u256,
}

impl PlonkVerifyingKey {
    /// Canonical byte length of [`PlonkVerifyingKey::to_bytes`].
    ///
    /// `[tau]_2` (128) ‖ `sigma_1, sigma_2, sigma_3` (192) ‖ `omega` (32) ‖
    /// `n` (4, big-endian `u32`) ‖ `k1` (32) ‖ `k2` (32).
    pub const BYTE_LEN: usize = 420;

    /// Decodes a verifying key from its canonical encoding.
    ///
    /// Rejects wrong lengths, non-canonical scalars, off-curve or
    /// small-subgroup points, an invalid G2 SRS element, and inconsistent
    /// domain parameters (`n` not a power of two, `omega^n != 1`, zero shifts).
    ///
    /// # Errors
    ///
    /// [`ZkError::DeserializationError`] for any of the above.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ZkError> {
        if bytes.len() != Self::BYTE_LEN {
            return Err(ZkError::DeserializationError);
        }

        let srs_g2 = g2_from_bytes(&bytes[0..128])?;

        let mut s_sigma_commitments = [G1Affine {
            x: u256::from(0u8),
            y: u256::from(0u8),
        }; 3];
        for (index, slot) in s_sigma_commitments.iter_mut().enumerate() {
            let offset = 128 + index * 64;
            *slot = decode_g1_allow_identity(&bytes[offset..offset + 64])?;
        }

        let omega = decode_fr(&bytes[320..352])?;
        let mut n_bytes = [0u8; 4];
        n_bytes.copy_from_slice(&bytes[352..356]);
        let n = u32::from_be_bytes(n_bytes);
        let k1 = decode_fr(&bytes[356..388])?;
        let k2 = decode_fr(&bytes[388..420])?;

        let key = Self {
            srs_g2,
            s_sigma_commitments,
            omega,
            n,
            k1,
            k2,
        };
        if !key.has_canonical_domain() {
            return Err(ZkError::DeserializationError);
        }
        Ok(key)
    }

    /// Encodes the key in the canonical fixed-width form documented at
    /// [`PlonkVerifyingKey::BYTE_LEN`].
    pub fn to_bytes(&self) -> [u8; Self::BYTE_LEN] {
        let mut out = [0u8; Self::BYTE_LEN];
        out[0..128].copy_from_slice(&self.srs_g2.to_bytes());
        for (index, point) in self.s_sigma_commitments.iter().enumerate() {
            let offset = 128 + index * 64;
            out[offset..offset + 64].copy_from_slice(&g1_to_bytes(point));
        }
        out[320..352].copy_from_slice(&self.omega.to_be_bytes());
        out[352..356].copy_from_slice(&self.n.to_be_bytes());
        out[356..388].copy_from_slice(&self.k1.to_be_bytes());
        out[388..420].copy_from_slice(&self.k2.to_be_bytes());
        out
    }

    /// Domain sanity: `n` is a non-zero power of two, `omega` has order exactly
    /// `n`, and both coset shifts are non-zero.
    fn has_canonical_domain(&self) -> bool {
        if self.n == 0 || !self.n.is_power_of_two() {
            return false;
        }
        if self.omega == u256::from(0u8) || self.k1 == u256::from(0u8) || self.k2 == u256::from(0u8)
        {
            return false;
        }
        if self.n > 1 && self.omega == u256::from(1u8) {
            return false;
        }
        Bn254::pow(self.omega, u256::from(self.n as u64)) == u256::from(1u8)
    }
}

fn decode_g1_allow_identity(bytes: &[u8]) -> Result<G1Affine, ZkError> {
    if bytes.len() != 64 {
        return Err(ZkError::DeserializationError);
    }
    let mut x_bytes = [0u8; 32];
    let mut y_bytes = [0u8; 32];
    x_bytes.copy_from_slice(&bytes[..32]);
    y_bytes.copy_from_slice(&bytes[32..64]);
    let x = u256::from_be_bytes(x_bytes);
    let y = u256::from_be_bytes(y_bytes);
    if x == u256::from(0u8) && y == u256::from(0u8) {
        return Ok(G1Affine { x, y });
    }
    if !Bn254::is_valid_g1_subgroup(x, y) {
        return Err(ZkError::DeserializationError);
    }
    Ok(G1Affine { x, y })
}

fn decode_fr(bytes: &[u8]) -> Result<u256, ZkError> {
    if bytes.len() != 32 {
        return Err(ZkError::DeserializationError);
    }
    let mut raw = [0u8; 32];
    raw.copy_from_slice(bytes);
    Bn254::fr_from_bytes(raw).ok_or(ZkError::DeserializationError)
}

/// Runs the complete PLONK verification pipeline.
///
/// This is the end-to-end entry point that was previously missing:
///
/// 1. **Fiat-Shamir** — a [`PlonkTranscript`] absorbs the verifying key, the
///    public inputs, every commitment and every evaluation, then squeezes
///    `beta, gamma, alpha, zeta, v, u`.
/// 2. **`PI(zeta)`** — [`barycentric_pi_evaluation`] evaluates the public-input
///    polynomial in `O(k)` with a single batch inversion.
/// 3. **Linearization** — [`evaluate_linearization`] recomputes `L(zeta)` from
///    the wire/permutation evaluations and rejects a proof whose declared
///    `L(zeta)` (or `t(zeta)`) does not match.
/// 4. **Split quotient** — [`split_quotient_commitment`] folds
///    `[t_lo] + zeta^n [t_mid] + zeta^{2n} [t_hi]`.
/// 5. **Batched KZG opening** — the folded commitments/proofs are checked with
///    one two-pair `bn254_multi_pairing_check` via [`verify_plonk_kzg`].
///
/// # Arguments
///
/// * `vk` – The verifying key (decode it with
///   [`PlonkVerifyingKey::from_bytes`] for untrusted bytes).
/// * `proof` – The proof (decode it with [`PlonkProof::from_bytes`]).
/// * `public_inputs` – The statement; `len() <= vk.n`.
/// * `selectors` – Selector evaluations at the derived `zeta`, taken from the
///   preprocessed circuit description.  These are **trusted** inputs.
///
/// # Returns
///
/// `Ok(true)` only when the quotient identity, the linearization evaluation and
/// the batched KZG opening all hold.  Malformed inputs produce an error; a
/// well-formed but inconsistent proof produces `Ok(false)`.
pub fn verify_plonk_full(
    env: &Env,
    vk: &PlonkVerifyingKey,
    proof: &PlonkProof,
    public_inputs: &[u256],
    selectors: &SelectorEvaluations,
) -> Result<bool, ZkError> {
    let n = vk.n as usize;
    if n == 0 || public_inputs.len() > n {
        return Err(ZkError::InvalidInput);
    }
    if !vk.has_canonical_domain() {
        return Err(ZkError::InvalidInput);
    }

    // ── 1. Reject non-canonical scalars before any arithmetic ──────────────
    let scalars = [
        proof.wire_evaluations[0],
        proof.wire_evaluations[1],
        proof.wire_evaluations[2],
        proof.sigma_evaluations[0],
        proof.sigma_evaluations[1],
        proof.z_omega_evaluation,
        proof.z_evaluation,
        proof.s_sigma3_evaluation,
        proof.quotient_evaluation,
        proof.linearization_evaluation,
        selectors.q_l,
        selectors.q_r,
        selectors.q_o,
        selectors.q_m,
        selectors.q_c,
    ];
    for value in scalars {
        if !Bn254::is_valid_scalar(value) {
            return Err(ZkError::InvalidFieldElement);
        }
    }
    for value in public_inputs {
        if !Bn254::is_valid_scalar(*value) {
            return Err(ZkError::InvalidFieldElement);
        }
    }

    // ── 2. Transcript: beta, gamma, alpha, zeta, v, u ──────────────────────
    let mut transcript = PlonkTranscript::new(b"plonk-full-verifier-v1");
    transcript.absorb_bytes(b"vk", &vk.to_bytes());
    transcript.absorb_usize(b"public_inputs", public_inputs.len());
    for value in public_inputs {
        transcript.absorb_scalar(b"public_input", *value);
    }
    for commitment in &proof.wire_commitments {
        transcript.absorb_g1(b"wire_commitment", commitment);
    }
    transcript.absorb_g1(b"z_commitment", &proof.z_commitment);
    for commitment in &proof.quotient_commitments {
        transcript.absorb_g1(b"quotient_commitment", commitment);
    }
    transcript.absorb_g1(b"w_zeta", &proof.w_zeta);
    transcript.absorb_g1(b"w_zeta_omega", &proof.w_zeta_omega);
    for evaluation in &proof.wire_evaluations {
        transcript.absorb_scalar(b"wire_evaluation", *evaluation);
    }
    for evaluation in &proof.sigma_evaluations {
        transcript.absorb_scalar(b"sigma_evaluation", *evaluation);
    }
    transcript.absorb_scalar(b"z_omega_evaluation", proof.z_omega_evaluation);
    transcript.absorb_scalar(b"z_evaluation", proof.z_evaluation);
    transcript.absorb_scalar(b"s_sigma3_evaluation", proof.s_sigma3_evaluation);
    transcript.absorb_scalar(b"quotient_evaluation", proof.quotient_evaluation);
    transcript.absorb_scalar(b"linearization_evaluation", proof.linearization_evaluation);
    let challenges = transcript.derive_plonk_challenges();

    // ── 3. PI(zeta) via barycentric evaluation ──────────────────────────────
    let pi_zeta = barycentric_pi_evaluation(public_inputs, challenges.zeta, vk.omega, n)?;

    // ── 4. Linearization identity: t(zeta) = L(zeta) + PI(zeta) ─────────────
    let commits = CommitEvaluations {
        z_zeta: proof.z_evaluation,
        s_sigma3_zeta: proof.s_sigma3_evaluation,
    };
    let domain = DomainParams {
        n,
        k1: vk.k1,
        k2: vk.k2,
    };
    let linearization = evaluate_linearization(
        proof,
        &challenges.linearization_challenges(),
        selectors,
        &commits,
        &domain,
    )?;
    if linearization != proof.linearization_evaluation {
        return Ok(false);
    }
    if Bn254::add(linearization, pi_zeta) != proof.quotient_evaluation {
        return Ok(false);
    }

    // ── 5. Split-quotient folding and batched KZG opening ──────────────────
    let quotient_commitment =
        split_quotient_commitment(&proof.quotient_commitments, challenges.zeta, n)?;

    let commitments = [
        proof.wire_commitments[0],
        proof.wire_commitments[1],
        proof.wire_commitments[2],
        proof.z_commitment,
        quotient_commitment,
        vk.s_sigma_commitments[0],
        vk.s_sigma_commitments[1],
    ];
    let evaluations = [
        proof.wire_evaluations[0],
        proof.wire_evaluations[1],
        proof.wire_evaluations[2],
        proof.z_evaluation,
        proof.quotient_evaluation,
        proof.sigma_evaluations[0],
        proof.sigma_evaluations[1],
    ];

    let inputs = KzgEvalProofInputs {
        commitments: &commitments,
        evaluations_at_zeta: &evaluations,
        commitment_at_zeta_omega: proof.z_commitment,
        evaluation_at_zeta_omega: proof.z_omega_evaluation,
        w_zeta: proof.w_zeta,
        w_zeta_omega: proof.w_zeta_omega,
        v: challenges.v,
        u: challenges.u,
        zeta: challenges.zeta,
        omega: vk.omega,
    };
    verify_plonk_kzg(env, &inputs, vk.srs_g2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ethnum::u256;
    use soroban_sdk::Env;
    use soroban_zk_core::G1Affine;

    // ─── helpers ────────────────────────────────────────────────────────────

    fn g1_gen() -> G1Affine {
        G1Affine {
            x: u256::from(1u8),
            y: u256::from(2u8),
        }
    }

    fn g2_gen() -> G2Affine {
        G2Affine::generator()
    }

    // ─── unit tests ─────────────────────────────────────────────────────────

    /// Calling `verify_plonk_kzg` with well-formed inputs must not panic and
    /// must return a `bool` (true or false) rather than an error.
    ///
    /// Note: Because the fixture is *not* a valid proof, the pairing will
    /// return `false`.  We only assert the absence of errors here.
    #[test]
    fn verify_plonk_kzg_completes_without_error_for_valid_inputs() {
        let env = Env::default();
        let commitments = [g1_gen()];
        let evaluations = [u256::from(3u8)];

        let inputs = KzgEvalProofInputs {
            commitments: &commitments,
            evaluations_at_zeta: &evaluations,
            commitment_at_zeta_omega: g1_gen(),
            evaluation_at_zeta_omega: u256::from(5u8),
            w_zeta: g1_gen(),
            w_zeta_omega: g1_gen(),
            v: u256::from(7u8),
            u: u256::from(11u8),
            zeta: u256::from(13u8),
            omega: u256::from(17u8),
        };

        // A random G2 point used as the mock SRS element.
        let srs_g2 = g2_gen();

        // Must return Ok(bool) — not panic, not Err.
        match verify_plonk_kzg(&env, &inputs, srs_g2) {
            Ok(_accepted) => { /* pass — result is mathematically undefined for dummy data */ }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    /// `verify_plonk_kzg` must propagate `ZkError::InvalidInput` when the
    /// commitment and evaluation slices have different lengths.
    #[test]
    fn verify_plonk_kzg_propagates_invalid_input_error() {
        let env = Env::default();
        let commitments = [g1_gen()];
        let evaluations = [u256::from(1u8), u256::from(2u8)]; // wrong length

        let inputs = KzgEvalProofInputs {
            commitments: &commitments,
            evaluations_at_zeta: &evaluations,
            commitment_at_zeta_omega: g1_gen(),
            evaluation_at_zeta_omega: u256::from(5u8),
            w_zeta: g1_gen(),
            w_zeta_omega: g1_gen(),
            v: u256::from(7u8),
            u: u256::from(11u8),
            zeta: u256::from(13u8),
            omega: u256::from(17u8),
        };

        assert_eq!(
            verify_plonk_kzg(&env, &inputs, g2_gen()),
            Err(ZkError::InvalidInput)
        );
    }

    /// `verify_plonk_kzg` must propagate `ZkError::InvalidFieldElement` when
    /// any challenge scalar is ≥ the BN254 scalar field modulus.
    #[test]
    fn verify_plonk_kzg_propagates_field_element_error() {
        let env = Env::default();
        let commitments = [g1_gen()];
        let evaluations = [u256::from(3u8)];

        let inputs = KzgEvalProofInputs {
            commitments: &commitments,
            evaluations_at_zeta: &evaluations,
            commitment_at_zeta_omega: g1_gen(),
            evaluation_at_zeta_omega: u256::from(5u8),
            w_zeta: g1_gen(),
            w_zeta_omega: g1_gen(),
            v: u256::from(7u8),
            u: u256::from(11u8),
            zeta: Bn254::FR_MODULUS, // out-of-range zeta
            omega: u256::from(17u8),
        };

        assert_eq!(
            verify_plonk_kzg(&env, &inputs, g2_gen()),
            Err(ZkError::InvalidFieldElement)
        );
    }

    /// Structural test: verify the full call chain reaches the pairing engine.
    /// With u = 0 and carefully chosen inputs, we confirm no panic/error occurs.
    /// See inline comments for the mathematical reasoning.
    #[test]
    fn verify_plonk_kzg_trivial_self_cancelling_proof() {
        let env = Env::default();

        // With u = 0: lhs = W_ζ
        // With zeta = 0: zeta*W_ζ = identity
        // F = C_0 = G1_gen, [E]_1 = eval_0 * G1_gen = 1 * G1_gen = G1_gen
        // → F - [E]_1 = G1_gen - G1_gen = identity
        // → rhs = identity
        //
        // Pairing check: e(G1_gen, srs_g2) · e(-identity, G2_gen) = e(G1_gen, srs_g2) ≠ 1
        // So the proof is NOT accepted — but no error should be returned.
        let commitments = [g1_gen()];
        let evaluations = [u256::from(1u8)];

        let inputs = KzgEvalProofInputs {
            commitments: &commitments,
            evaluations_at_zeta: &evaluations,
            commitment_at_zeta_omega: g1_gen(),
            evaluation_at_zeta_omega: u256::from(0u8),
            w_zeta: g1_gen(),
            w_zeta_omega: g1_gen(),
            v: u256::from(1u8),
            u: u256::from(0u8),
            zeta: u256::from(0u8),
            omega: u256::from(1u8),
        };

        let result = verify_plonk_kzg(&env, &inputs, g2_gen());
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }

    // ─── multi-proof batch tests (Issue #427) ───────────────────────────────

    /// Builds a well-formed [`KzgEvalProofInputs`] bundle for batch tests.
    fn batch_bundle() -> KzgEvalProofInputs<'static> {
        static C: &[G1Affine] = &[G1Affine {
            x: u256::from_words(0, 1),
            y: u256::from_words(0, 2),
        }];
        static E: &[u256] = &[u256::from_words(0, 3)];
        KzgEvalProofInputs {
            commitments: C,
            evaluations_at_zeta: E,
            commitment_at_zeta_omega: g1_gen(),
            evaluation_at_zeta_omega: u256::from(5u8),
            w_zeta: g1_gen(),
            w_zeta_omega: g1_gen(),
            v: u256::from(7u8),
            u: u256::from(11u8),
            zeta: u256::from(13u8),
            omega: u256::from(17u8),
        }
    }

    /// An empty batch is a structural error, never a pairing call.
    #[test]
    fn verify_plonk_kzg_batch_rejects_empty_batch() {
        let env = Env::default();
        assert_eq!(
            verify_plonk_kzg_batch(&env, &[], &[7u8; 32], g2_gen()),
            Err(ZkError::InvalidInput)
        );
    }

    /// A batch of well-formed bundles must run the single aggregated pairing
    /// and return `Ok(bool)` (the dummy data is not a real proof, so we only
    /// assert the absence of errors).
    #[test]
    fn verify_plonk_kzg_batch_completes_without_error() {
        let env = Env::default();
        let inputs = [batch_bundle(), batch_bundle()];
        match verify_plonk_kzg_batch(&env, &inputs, &[9u8; 32], g2_gen()) {
            Ok(_) => { /* pass — result undefined for dummy data */ }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    /// A zero `batch_seed` reduces the first batch scalar to zero and must be
    /// rejected before any pairing is attempted.
    #[test]
    fn verify_plonk_kzg_batch_rejects_zero_seed() {
        let env = Env::default();
        let inputs = [batch_bundle()];
        assert_eq!(
            verify_plonk_kzg_batch(&env, &inputs, &[0u8; 32], g2_gen()),
            Err(ZkError::InvalidFieldElement)
        );
    }

    // ─── full verifier tests ────────────────────────────────────────────────

    fn identity() -> G1Affine {
        G1Affine {
            x: u256::from(0u8),
            y: u256::from(0u8),
        }
    }

    /// A structurally valid, fully consistent "empty circuit" proof: every
    /// commitment is infinity, every evaluation is zero, so the quotient
    /// identity holds and the batched opening is the zero polynomial's opening.
    fn zero_proof() -> PlonkProof {
        PlonkProof {
            wire_commitments: [identity(); 3],
            z_commitment: identity(),
            quotient_commitments: [identity(); 3],
            w_zeta: identity(),
            w_zeta_omega: identity(),
            wire_evaluations: [u256::from(0u8); 3],
            sigma_evaluations: [u256::from(0u8); 2],
            z_omega_evaluation: u256::from(0u8),
            z_evaluation: u256::from(0u8),
            s_sigma3_evaluation: u256::from(0u8),
            quotient_evaluation: u256::from(0u8),
            linearization_evaluation: u256::from(0u8),
        }
    }

    fn zero_selectors() -> SelectorEvaluations {
        SelectorEvaluations {
            q_l: u256::from(0u8),
            q_r: u256::from(0u8),
            q_o: u256::from(0u8),
            q_m: u256::from(0u8),
            q_c: u256::from(0u8),
        }
    }

    /// `n = 1` domain; `omega = 1` is a valid order-1 generator.
    fn zero_vk() -> PlonkVerifyingKey {
        PlonkVerifyingKey {
            srs_g2: g2_gen(),
            s_sigma_commitments: [identity(); 3],
            omega: u256::from(1u8),
            n: 1,
            k1: u256::from(1u8),
            k2: u256::from(2u8),
        }
    }

    #[test]
    fn plonk_verifying_key_bytes_round_trip() {
        let vk = zero_vk();
        let bytes = vk.to_bytes();
        assert_eq!(bytes.len(), PlonkVerifyingKey::BYTE_LEN);
        assert_eq!(PlonkVerifyingKey::from_bytes(&bytes), Ok(vk));
    }

    #[test]
    fn plonk_verifying_key_from_bytes_rejects_malformed() {
        let bytes = zero_vk().to_bytes();
        assert_eq!(
            PlonkVerifyingKey::from_bytes(&bytes[..PlonkVerifyingKey::BYTE_LEN - 1]),
            Err(ZkError::DeserializationError)
        );

        // n = 0 is not a valid domain size.
        let mut bad_n = bytes;
        bad_n[352..356].copy_from_slice(&0u32.to_be_bytes());
        assert_eq!(
            PlonkVerifyingKey::from_bytes(&bad_n),
            Err(ZkError::DeserializationError)
        );

        // n = 3 is not a power of two.
        let mut bad_n = bytes;
        bad_n[352..356].copy_from_slice(&3u32.to_be_bytes());
        assert_eq!(
            PlonkVerifyingKey::from_bytes(&bad_n),
            Err(ZkError::DeserializationError)
        );

        // omega^n != 1 for n = 2 and omega = 2.
        let mut bad_omega = bytes;
        bad_omega[352..356].copy_from_slice(&2u32.to_be_bytes());
        bad_omega[320..352].copy_from_slice(&u256::from(2u8).to_be_bytes());
        assert_eq!(
            PlonkVerifyingKey::from_bytes(&bad_omega),
            Err(ZkError::DeserializationError)
        );

        // A non-canonical scalar field element.
        let mut bad_k1 = bytes;
        bad_k1[356..388].copy_from_slice(&Bn254::FR_MODULUS.to_be_bytes());
        assert_eq!(
            PlonkVerifyingKey::from_bytes(&bad_k1),
            Err(ZkError::DeserializationError)
        );
    }

    #[test]
    fn plonk_verifying_key_from_bytes_rejects_off_curve_sigma() {
        let mut bytes = zero_vk().to_bytes();
        bytes[128 + 32..128 + 64].copy_from_slice(&u256::from(3u8).to_be_bytes());
        assert_eq!(
            PlonkVerifyingKey::from_bytes(&bytes),
            Err(ZkError::DeserializationError)
        );
    }

    #[test]
    fn verify_plonk_full_accepts_consistent_proof_through_canonical_bytes() {
        let env = Env::default();
        let vk = PlonkVerifyingKey::from_bytes(&zero_vk().to_bytes()).expect("canonical vk");
        let proof = PlonkProof::from_bytes(&zero_proof().to_bytes()).expect("canonical proof");
        assert_eq!(
            verify_plonk_full(&env, &vk, &proof, &[], &zero_selectors()),
            Ok(true)
        );
    }

    #[test]
    fn verify_plonk_full_rejects_inconsistent_quotient() {
        let env = Env::default();
        let vk = zero_vk();
        let mut proof = zero_proof();
        proof.quotient_evaluation = u256::from(1u8);
        assert_eq!(
            verify_plonk_full(&env, &vk, &proof, &[], &zero_selectors()),
            Ok(false)
        );
    }

    #[test]
    fn verify_plonk_full_rejects_inconsistent_linearization() {
        let env = Env::default();
        let vk = zero_vk();
        let mut proof = zero_proof();
        proof.linearization_evaluation = u256::from(1u8);
        assert_eq!(
            verify_plonk_full(&env, &vk, &proof, &[], &zero_selectors()),
            Ok(false)
        );
    }

    #[test]
    fn verify_plonk_full_rejects_nonzero_public_input_against_zero_quotient() {
        let env = Env::default();
        let vk = zero_vk();
        let proof = zero_proof();
        assert_eq!(
            verify_plonk_full(&env, &vk, &proof, &[u256::from(7u8)], &zero_selectors()),
            Ok(false)
        );
    }

    #[test]
    fn verify_plonk_full_rejects_malformed_boundaries() {
        let env = Env::default();
        let vk = zero_vk();
        let proof = zero_proof();
        let selectors = zero_selectors();

        // More public inputs than the domain holds.
        assert_eq!(
            verify_plonk_full(
                &env,
                &vk,
                &proof,
                &[u256::from(1u8), u256::from(2u8)],
                &selectors
            ),
            Err(ZkError::InvalidInput)
        );

        // Non-canonical scalar in the proof.
        let mut bad = proof;
        bad.sigma_evaluations[0] = Bn254::FR_MODULUS;
        assert_eq!(
            verify_plonk_full(&env, &vk, &bad, &[], &selectors),
            Err(ZkError::InvalidFieldElement)
        );

        // Non-canonical selector evaluation.
        let mut bad_selectors = selectors;
        bad_selectors.q_c = Bn254::FR_MODULUS;
        assert_eq!(
            verify_plonk_full(&env, &vk, &proof, &[], &bad_selectors),
            Err(ZkError::InvalidFieldElement)
        );

        // Degenerate verifying-key domain.
        let mut bad_vk = vk;
        bad_vk.n = 0;
        assert_eq!(
            verify_plonk_full(&env, &bad_vk, &proof, &[], &selectors),
            Err(ZkError::InvalidInput)
        );
    }

    #[test]
    fn verify_plonk_full_rejects_tampered_opening_proof() {
        let env = Env::default();
        let vk = zero_vk();
        let mut proof = zero_proof();
        // A non-zero opening proof no longer opens the zero polynomial.
        proof.w_zeta = g1_gen();
        assert_eq!(
            verify_plonk_full(&env, &vk, &proof, &[], &zero_selectors()),
            Ok(false)
        );
    }
}
