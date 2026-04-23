// SPDX-License-Identifier: GPL-3.0-only
//
// Pickle Protocol 4 byte-grammar scanner for Grok-1 JAX shards. Extracts
// `(dtype, shape, offset, nbytes)` tuples and `QuantizedWeight8bit` pairing
// without running a real unpickler or decoding tensor bodies.
//
// This module is intentionally free of schema/inventory concerns: it speaks
// only in terms of raw tensor anchors. The higher layers map these anchors
// into `schema::TensorInfo` and assign semantics.
//
// Stable pickle-4 invariants we rely on (documented in the original
// single-file CLI):
//
//   1. Every `numpy.ndarray` is reconstructed via a state tuple whose dtype
//      is built by `numpy.dtype('f4' | 'i1', False, True)`. That guarantees
//      the byte signature `<dtype class push> \x8c\x02XX [\x94] \x89 \x88
//      \x87 [\x94] R`.
//   2. The array payload is always emitted immediately after the fortran
//      bool: `[\x88|\x89] <C|B|\x8e> <len> <raw bytes>`.

use std::cmp::min;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use memchr::memmem;
use memmap2::Mmap;

use crate::schema::{TensorDType, TensorRole, TensorShape};

// --- Pickle protocol 4 opcodes we touch ------------------------------------

const OP_PROTO: u8 = 0x80;

const OP_SHORT_BINUNICODE: u8 = 0x8c;
const OP_MEMOIZE: u8 = 0x94;
const OP_STACK_GLOBAL: u8 = 0x93;

const OP_BINGET: u8 = b'h';      // 0x68  u8 index
const OP_LONG_BINGET: u8 = b'j'; // 0x6a  u32 index

const OP_SHORT_BINBYTES: u8 = b'C'; // 0x43  u8 len
const OP_BINBYTES: u8 = b'B';       // 0x42  u32 len
const OP_BINBYTES8: u8 = 0x8e;      // u64 len

const OP_MARK: u8 = b'(';
const OP_TUPLE: u8 = b't';
const OP_TUPLE1: u8 = 0x85;
const OP_TUPLE2: u8 = 0x86;
const OP_TUPLE3: u8 = 0x87;
const OP_EMPTY_TUPLE: u8 = b')';

const OP_BININT1: u8 = b'K'; // 0x4b  u8
const OP_BININT2: u8 = b'M'; // 0x4d  u16
const OP_BININT: u8 = b'J';  // 0x4a  i32

const OP_NEWTRUE: u8 = 0x88;
const OP_NEWFALSE: u8 = 0x89;

// --- Anchor bytes ----------------------------------------------------------

const DTYPE_TAG_F32: &[u8] = b"\x8c\x02f4";
const DTYPE_TAG_I8: &[u8] = b"\x8c\x02i1";

const SIG_QW8_STRICT: &[u8] = b"\x8c\x08__main__\x94\x8c\x13QuantizedWeight8bit\x94";
const SIG_QW8_LOOSE_MODULE: &[u8] = b"\x8c\x08__main__";
const SIG_QW8_CLASS_TAG: &[u8] = b"\x8c\x13QuantizedWeight8bit";

// --- Public API ------------------------------------------------------------

/// Raw tensor record as produced by the byte-grammar scanner. No semantic
/// classification is applied here; that happens in `inventory::classify`.
#[derive(Clone, Debug)]
pub struct RawTensor {
    pub role: TensorRole,
    pub dtype: TensorDType,
    pub shape: TensorShape,
    /// Absolute byte offset of the raw payload within the shard file.
    pub offset: u64,
    pub nbytes: u64,
}

/// Memory-map `path` and extract every tensor anchor it contains. Returns
/// the tensors sorted by `offset`. `QuantizedWeight8bit` sites are detected
/// and the adjacent int8/f32 pair is labeled with the matching
/// `TensorRole::QuantWeight` / `QuantScales`.
pub fn dissect_shard(path: &Path) -> Result<Vec<RawTensor>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // Safety: the file is not mutated while the mmap is live.
    let mm = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    let bytes: &[u8] = &mm;

    if bytes.len() < 2 || bytes[0] != OP_PROTO || bytes[1] != 0x04 {
        bail!(
            "not a pickle protocol 4 stream (magic={:02x?})",
            &bytes[..min(2, bytes.len())]
        );
    }

    let mut anchors = find_dtype_anchors(bytes);
    anchors.sort_by_key(|a| a.tag_pos);

    let mut tensors: Vec<RawTensor> = Vec::with_capacity(anchors.len());
    for a in &anchors {
        match extract_tensor(bytes, a) {
            Ok(t) => tensors.push(t),
            Err(err) => {
                // Non-fatal: one bad anchor never aborts a shard.
                eprintln!("  skip anchor @ {:#x} in {}: {:#}", a.tag_pos, path.display(), err);
            }
        }
    }

    let qw8_sites = find_qw8_sites(bytes);
    assign_qw8_roles(&mut tensors, &qw8_sites);

    // Stable order for all downstream layers.
    tensors.sort_by_key(|t| t.offset);
    Ok(tensors)
}

// --- Anchor discovery ------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct DtypeAnchor {
    /// Byte index of the `\x8c\x02XX` short_binunicode naming the dtype.
    tag_pos: usize,
    /// First byte AFTER the 4-byte tag.
    after_tag: usize,
    dtype: TensorDType,
}

fn find_dtype_anchors(bytes: &[u8]) -> Vec<DtypeAnchor> {
    let mut out: Vec<DtypeAnchor> = Vec::new();
    for (tag, dtype) in [
        (DTYPE_TAG_F32, TensorDType::F32),
        (DTYPE_TAG_I8, TensorDType::I8),
    ] {
        for pos in memmem::find_iter(bytes, tag) {
            let after = pos + tag.len();
            if has_dtype_postamble(bytes, after) {
                out.push(DtypeAnchor { tag_pos: pos, after_tag: after, dtype });
            }
        }
    }
    out
}

fn has_dtype_postamble(bytes: &[u8], after_tag: usize) -> bool {
    let mut i = after_tag;
    if bytes.get(i) == Some(&OP_MEMOIZE) {
        i += 1;
    }
    bytes.get(i..i + 3) == Some(&[OP_NEWFALSE, OP_NEWTRUE, OP_TUPLE3])
}

// --- Per-anchor extraction -------------------------------------------------

fn extract_tensor(bytes: &[u8], anchor: &DtypeAnchor) -> Result<RawTensor> {
    let shape_dims = parse_shape_backward(bytes, anchor.tag_pos)?;
    let (offset, nbytes) = find_payload_forward(bytes, anchor.after_tag)?;
    let expected = anchor.dtype.itemsize() as u64
        * shape_dims.iter().copied().fold(1u64, |a, d| a.saturating_mul(d));
    if expected != 0 && expected != nbytes {
        return Err(anyhow!(
            "shape/payload mismatch: shape={:?} dtype={} nbytes={} expected={}",
            shape_dims,
            anchor.dtype.label(),
            nbytes,
            expected
        ));
    }
    Ok(RawTensor {
        role: TensorRole::Tensor,
        dtype: anchor.dtype,
        shape: TensorShape::new(shape_dims),
        offset,
        nbytes,
    })
}

fn parse_shape_backward(bytes: &[u8], tag_pos: usize) -> Result<Vec<u64>> {
    let mut p = tag_pos;

    // Skip MEMOIZE that may sit directly after STACK_GLOBAL.
    p = skip_memoize_back(bytes, p);

    if p >= 1 && bytes[p - 1] == OP_STACK_GLOBAL {
        // Fresh dtype class push: push("numpy") push("dtype") STACK_GLOBAL.
        p -= 1;
        p = skip_string_push_back(bytes, p)?; // "dtype"
        p = skip_string_push_back(bytes, p)?; // "numpy"
    } else {
        // Pre-memoized class: single BINGET / LONG_BINGET.
        p = skip_binget_back(bytes, p)?;
    }

    p = skip_memoize_back(bytes, p);
    decode_shape_tuple_backward(bytes, p)
}

fn skip_memoize_back(bytes: &[u8], p: usize) -> usize {
    if p >= 1 && bytes[p - 1] == OP_MEMOIZE { p - 1 } else { p }
}

fn skip_string_push_back(bytes: &[u8], p: usize) -> Result<usize> {
    let p = skip_memoize_back(bytes, p);

    if p >= 2 && bytes[p - 2] == OP_BINGET {
        return Ok(p - 2);
    }
    if p >= 5 && bytes[p - 5] == OP_LONG_BINGET {
        return Ok(p - 5);
    }
    for len in 1..=32usize {
        if p < len + 2 {
            break;
        }
        if bytes[p - len - 2] == OP_SHORT_BINUNICODE && bytes[p - len - 1] as usize == len {
            return Ok(p - len - 2);
        }
    }
    Err(anyhow!("cannot unwind string push ending at {:#x}", p.saturating_sub(1)))
}

fn skip_binget_back(bytes: &[u8], p: usize) -> Result<usize> {
    if p >= 2 && bytes[p - 2] == OP_BINGET {
        return Ok(p - 2);
    }
    if p >= 5 && bytes[p - 5] == OP_LONG_BINGET {
        return Ok(p - 5);
    }
    Err(anyhow!(
        "expected BINGET/LONG_BINGET before dtype tag at {:#x}",
        p.saturating_sub(1)
    ))
}

fn decode_shape_tuple_backward(bytes: &[u8], end_exclusive: usize) -> Result<Vec<u64>> {
    if end_exclusive == 0 {
        bail!("no room for shape tuple at offset {:#x}", end_exclusive);
    }
    let term = bytes[end_exclusive - 1];
    match term {
        OP_EMPTY_TUPLE => Ok(Vec::new()),
        OP_TUPLE1 => {
            let (v, _) = read_int_backward(bytes, end_exclusive - 1)?;
            Ok(vec![v])
        }
        OP_TUPLE2 => {
            let (v1, p1) = read_int_backward(bytes, end_exclusive - 1)?;
            let (v0, _) = read_int_backward(bytes, p1)?;
            Ok(vec![v0, v1])
        }
        OP_TUPLE3 => {
            let (v2, p2) = read_int_backward(bytes, end_exclusive - 1)?;
            let (v1, p1) = read_int_backward(bytes, p2)?;
            let (v0, _) = read_int_backward(bytes, p1)?;
            Ok(vec![v0, v1, v2])
        }
        OP_TUPLE => {
            let mut dims = Vec::new();
            let mut cursor = end_exclusive - 1;
            loop {
                if cursor == 0 {
                    bail!("ran off start walking MARK..TUPLE shape");
                }
                if bytes[cursor - 1] == OP_MARK {
                    dims.reverse();
                    return Ok(dims);
                }
                let (v, next) = read_int_backward(bytes, cursor)?;
                dims.push(v);
                cursor = next;
            }
        }
        other => Err(anyhow!(
            "unexpected shape terminator opcode 0x{:02x} at offset {:#x}",
            other,
            end_exclusive - 1
        )),
    }
}

fn read_int_backward(bytes: &[u8], end_exclusive: usize) -> Result<(u64, usize)> {
    if end_exclusive == 0 {
        bail!("int parse: empty range");
    }
    if end_exclusive >= 5 && bytes[end_exclusive - 5] == OP_BININT {
        let v = i32::from_le_bytes(bytes[end_exclusive - 4..end_exclusive].try_into().unwrap());
        if v < 0 {
            bail!("negative BININT in shape dim");
        }
        return Ok((v as u64, end_exclusive - 5));
    }
    if end_exclusive >= 3 && bytes[end_exclusive - 3] == OP_BININT2 {
        let v = u16::from_le_bytes(bytes[end_exclusive - 2..end_exclusive].try_into().unwrap());
        return Ok((v as u64, end_exclusive - 3));
    }
    if end_exclusive >= 2 && bytes[end_exclusive - 2] == OP_BININT1 {
        return Ok((bytes[end_exclusive - 1] as u64, end_exclusive - 2));
    }
    Err(anyhow!("no BININT* opcode ending at offset {:#x}", end_exclusive - 1))
}

fn find_payload_forward(bytes: &[u8], after_tag: usize) -> Result<(u64, u64)> {
    let start = after_tag;
    let stop = min(bytes.len().saturating_sub(1), start + 256);
    let mut i = start;
    while i < stop {
        let b = bytes[i];
        if (b == OP_NEWTRUE || b == OP_NEWFALSE) && i + 1 < bytes.len() {
            let opcode = bytes[i + 1];
            match opcode {
                OP_SHORT_BINBYTES => {
                    let hdr = i + 2;
                    if hdr >= bytes.len() {
                        bail!("truncated SHORT_BINBYTES at {:#x}", i + 1);
                    }
                    let len = bytes[hdr] as u64;
                    let data = (hdr + 1) as u64;
                    check_payload_bounds(bytes, data, len)?;
                    return Ok((data, len));
                }
                OP_BINBYTES => {
                    let hdr = i + 2;
                    if hdr + 4 > bytes.len() {
                        bail!("truncated BINBYTES header at {:#x}", i + 1);
                    }
                    let len = u32::from_le_bytes(bytes[hdr..hdr + 4].try_into().unwrap()) as u64;
                    let data = (hdr + 4) as u64;
                    check_payload_bounds(bytes, data, len)?;
                    return Ok((data, len));
                }
                OP_BINBYTES8 => {
                    let hdr = i + 2;
                    if hdr + 8 > bytes.len() {
                        bail!("truncated BINBYTES8 header at {:#x}", i + 1);
                    }
                    let len = u64::from_le_bytes(bytes[hdr..hdr + 8].try_into().unwrap());
                    let data = (hdr + 8) as u64;
                    check_payload_bounds(bytes, data, len)?;
                    return Ok((data, len));
                }
                _ => {}
            }
        }
        i += 1;
    }
    bail!(
        "no fortran+payload marker found within {} bytes of dtype tag @ {:#x}",
        stop - start,
        after_tag
    )
}

fn check_payload_bounds(bytes: &[u8], data: u64, len: u64) -> Result<()> {
    if data.saturating_add(len) > bytes.len() as u64 {
        bail!(
            "payload {} bytes overruns file (data_off={:#x}, file_len={})",
            len,
            data,
            bytes.len()
        );
    }
    Ok(())
}

// --- QuantizedWeight8bit grouping -----------------------------------------

fn find_qw8_sites(bytes: &[u8]) -> Vec<usize> {
    let mut sites: Vec<usize> = memmem::find_iter(bytes, SIG_QW8_STRICT).collect();
    for pos in memmem::find_iter(bytes, SIG_QW8_LOOSE_MODULE) {
        let window_end = min(bytes.len(), pos + SIG_QW8_LOOSE_MODULE.len() + 24);
        if memmem::find(&bytes[pos..window_end], SIG_QW8_CLASS_TAG).is_some()
            && !sites.contains(&pos)
        {
            sites.push(pos);
        }
    }
    sites.sort();
    sites.dedup();
    sites
}

fn assign_qw8_roles(tensors: &mut [RawTensor], sites: &[usize]) {
    for &site in sites {
        let site_u64 = site as u64;
        let mut weight_idx = None;
        let mut scales_idx = None;
        for (idx, t) in tensors.iter().enumerate() {
            if t.offset < site_u64 {
                continue;
            }
            match (t.dtype, weight_idx, scales_idx) {
                (TensorDType::I8, None, _) => weight_idx = Some(idx),
                (TensorDType::F32, _, None) => scales_idx = Some(idx),
                _ => {}
            }
            if weight_idx.is_some() && scales_idx.is_some() {
                break;
            }
        }
        if let Some(i) = weight_idx {
            tensors[i].role = TensorRole::QuantWeight;
        }
        if let Some(i) = scales_idx {
            tensors[i].role = TensorRole::QuantScales;
        }
    }
}
