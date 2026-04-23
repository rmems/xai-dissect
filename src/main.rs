// SPDX-License-Identifier: GPL-3.0-only
//
// xai-dissect: zero-copy metadata extractor for Grok-1 raw JAX/Pickle shards.
// Copyright (C) 2026 xai-dissect contributors.
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details. You should have received a copy of the
// GNU General Public License along with this program. If not, see
// <https://www.gnu.org/licenses/>.
//
// The shards are `pickle.dump(obj, protocol=4)` blobs where `obj` is either
// a bare `numpy.ndarray` or a `__main__.QuantizedWeight8bit` dataclass
// wrapping two ndarrays (`weight: int8`, `scales: float32`).
//
// We do NOT run a real unpickler. We memory-map each shard and exploit two
// stable pickle-4 invariants:
//
//   1. Every numpy ndarray is reconstructed via a state tuple whose dtype is
//      built by `numpy.dtype('f4' | 'i1', False, True)`. That guarantees the
//      immediate byte signature is:
//
//          <dtype-class push>  \x8c\x02XX  [\x94]  \x89 \x88 \x87  [\x94]  R
//
//      where `XX` is `f4` or `i1`. Anchor on `\x8c\x02XX` and validate the
//      tight post-amble `\x89 \x88 \x87`.
//
//   2. The array payload is always emitted immediately after the fortran-order
//      bool: `[\x88|\x89] <C|B|\x8e> <len> <raw bytes>`. Scanning forward from
//      the dtype tag for the first such pair lands exactly on the payload.
//
// Backward from the dtype tag we walk a tiny state machine that unwinds the
// "dtype class push" (full STACK_GLOBAL form or BINGET/LONG_BINGET memo
// re-fetch) to find the shape tuple's terminator opcode.

use std::cmp::min;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use comfy_table::{presets::UTF8_FULL, Cell, ContentArrangement, Table};
use memchr::memmem;
use memmap2::Mmap;

// --- Pickle protocol 4 opcodes we touch -------------------------------------

const OP_PROTO: u8 = 0x80;

const OP_SHORT_BINUNICODE: u8 = 0x8c;
const OP_MEMOIZE: u8 = 0x94;
const OP_STACK_GLOBAL: u8 = 0x93;

const OP_BINGET: u8 = b'h';        // 0x68  u8 index
const OP_LONG_BINGET: u8 = b'j';   // 0x6a  u32 index

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

// --- Anchor bytes -----------------------------------------------------------

const DTYPE_TAG_F32: &[u8] = b"\x8c\x02f4";
const DTYPE_TAG_I8: &[u8] = b"\x8c\x02i1";

// QuantizedWeight8bit class reference (strict and loose). The dumper Grok-1
// used emits both MEMOIZE bytes, but the spec asks us to be lax against a
// missing trailing \x94 when the Pickler reuses a memo slot.
const SIG_QW8_STRICT: &[u8] =
    b"\x8c\x08__main__\x94\x8c\x13QuantizedWeight8bit\x94";
const SIG_QW8_LOOSE_MODULE: &[u8] = b"\x8c\x08__main__";
const SIG_QW8_CLASS_TAG: &[u8] = b"\x8c\x13QuantizedWeight8bit";

// --- Types ------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Dtype {
    F32,
    I8,
}

impl Dtype {
    fn itemsize(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::I8 => 1,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::I8 => "int8",
        }
    }
}

#[derive(Debug, Clone)]
enum Role {
    Tensor,
    QuantWeight,
    QuantScales,
}

impl Role {
    fn label(&self) -> &'static str {
        match self {
            Role::Tensor => "tensor",
            Role::QuantWeight => "quant.weight",
            Role::QuantScales => "quant.scales",
        }
    }
}

#[derive(Debug, Clone)]
struct TensorEntry {
    role: Role,
    dtype: Dtype,
    shape: Vec<usize>,
    offset: usize, // absolute byte offset of raw payload within the shard file
    nbytes: usize,
}

// --- CLI --------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "xai-dissect",
    version,
    about = "Dissect Grok-1 JAX/Pickle tensor shards without decoding weights"
)]
struct Cli {
    /// Directory containing `tensor*` shard files (non-recursive).
    path: PathBuf,

    /// Only process the first N shards (sorted by filename). Useful for
    /// verifying extraction against `tensor00000_000` before sweeping a full
    /// checkpoint directory.
    #[arg(long)]
    limit: Option<usize>,

    /// Filename prefix filter. Defaults to `tensor`.
    #[arg(long, default_value = "tensor")]
    prefix: String,
}

// --- Entry point ------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    let md = std::fs::metadata(&cli.path)
        .with_context(|| format!("stat {}", cli.path.display()))?;
    if !md.is_dir() {
        bail!("{} is not a directory", cli.path.display());
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(&cli.path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with(&cli.prefix))
                    .unwrap_or(false)
        })
        .collect();
    shards.sort();

    if shards.is_empty() {
        bail!(
            "no shards found under {} with prefix '{}'",
            cli.path.display(),
            cli.prefix
        );
    }

    if let Some(n) = cli.limit {
        shards.truncate(n);
    }

    for shard in &shards {
        match dissect_shard(shard) {
            Ok(entries) => render_table(shard, &entries),
            Err(e) => eprintln!("warn: {}: {:#}", shard.display(), e),
        }
    }

    Ok(())
}

// --- Per-shard driver -------------------------------------------------------

fn dissect_shard(path: &Path) -> Result<Vec<TensorEntry>> {
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

    let mut entries = Vec::with_capacity(anchors.len());
    for a in &anchors {
        match extract_tensor(bytes, a) {
            Ok(e) => entries.push(e),
            Err(err) => eprintln!(
                "  skip anchor @ {:#x}: {:#}",
                a.tag_pos, err
            ),
        }
    }

    let qw8_sites = find_qw8_sites(bytes);
    assign_qw8_roles(&mut entries, &qw8_sites);

    Ok(entries)
}

// --- Anchor discovery -------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct DtypeAnchor {
    /// Byte index of the `\x8c\x02XX` short_binunicode naming the dtype.
    tag_pos: usize,
    /// First byte AFTER the 4-byte tag (pointer to optional trailing MEMOIZE).
    after_tag: usize,
    dtype: Dtype,
}

/// Locate every dtype tag whose forward post-amble matches numpy's
/// `np.dtype(str, False, True)` pickle shape. This is the core invariant that
/// works whether or not any individual MEMOIZE opcode is elided.
///
/// Valid post-amble forms (with `P` = after_tag):
///   * strict: `\x94 \x89 \x88 \x87` at P..P+4
///   * loose:        `\x89 \x88 \x87` at P..P+3
fn find_dtype_anchors(bytes: &[u8]) -> Vec<DtypeAnchor> {
    let mut out: Vec<DtypeAnchor> = Vec::new();

    for (tag, dtype) in [(DTYPE_TAG_F32, Dtype::F32), (DTYPE_TAG_I8, Dtype::I8)] {
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

// --- Per-anchor extraction --------------------------------------------------

fn extract_tensor(bytes: &[u8], anchor: &DtypeAnchor) -> Result<TensorEntry> {
    let shape = parse_shape_backward(bytes, anchor.tag_pos)?;
    let (offset, nbytes) = find_payload_forward(bytes, anchor.after_tag)?;
    let expected = anchor.dtype.itemsize() * shape.iter().product::<usize>();
    if expected != 0 && expected != nbytes {
        return Err(anyhow!(
            "shape/payload mismatch: shape={:?} dtype={} nbytes={} expected={}",
            shape,
            anchor.dtype.label(),
            nbytes,
            expected
        ));
    }
    Ok(TensorEntry {
        role: Role::Tensor,
        dtype: anchor.dtype,
        shape,
        offset,
        nbytes,
    })
}

/// Walk backward from the dtype tag through the variable-length "dtype class
/// push" to reach the shape tuple's terminator byte, then decode the shape.
///
/// The dtype class is either:
///   a) Freshly constructed: push("numpy") push("dtype") STACK_GLOBAL
///      with each string push being either a literal SHORT_BINUNICODE (with
///      optional MEMOIZE) or a BINGET/LONG_BINGET re-fetch of a memoized
///      string; STACK_GLOBAL itself may or may not be followed by MEMOIZE.
///   b) Pre-memoized and re-fetched: a single BINGET or LONG_BINGET.
///
/// We greedily unwind the post-class MEMOIZE, then disambiguate (a) vs (b)
/// by looking for `\x93` (STACK_GLOBAL) within a 2-byte lookback.
fn parse_shape_backward(bytes: &[u8], tag_pos: usize) -> Result<Vec<usize>> {
    let mut p = tag_pos;

    // Skip MEMOIZE that may sit directly after STACK_GLOBAL (case a).
    p = skip_memoize_back(bytes, p);

    if p >= 1 && bytes[p - 1] == OP_STACK_GLOBAL {
        // Case (a): consume STACK_GLOBAL and the two string pushes behind it.
        p -= 1;
        p = skip_string_push_back(bytes, p)?; // "dtype"
        p = skip_string_push_back(bytes, p)?; // "numpy"
    } else {
        // Case (b): a single BINGET / LONG_BINGET of the memoized class.
        p = skip_binget_back(bytes, p)?;
    }

    // Between the class push and the shape tuple there may be a shape-MEMOIZE.
    p = skip_memoize_back(bytes, p);

    decode_shape_tuple_backward(bytes, p)
}

fn skip_memoize_back(bytes: &[u8], p: usize) -> usize {
    if p >= 1 && bytes[p - 1] == OP_MEMOIZE {
        p - 1
    } else {
        p
    }
}

/// Step over either `h XX` (BINGET) or `j XX XX XX XX` (LONG_BINGET) or
/// `\x8c LEN <chars>` (SHORT_BINUNICODE, MEMOIZE-less) ending at `p - 1`.
/// Leaves the cursor on the byte after the consumed sequence (i.e. the new
/// logical end of the preceding tuple area).
fn skip_string_push_back(bytes: &[u8], p: usize) -> Result<usize> {
    // Optional trailing MEMOIZE for the string itself.
    let p = skip_memoize_back(bytes, p);

    if p >= 2 && bytes[p - 2] == OP_BINGET {
        return Ok(p - 2);
    }
    if p >= 5 && bytes[p - 5] == OP_LONG_BINGET {
        return Ok(p - 5);
    }
    // Literal SHORT_BINUNICODE: last char at p-1, length byte at p-len-1,
    // opcode at p-len-2. We bound len to <= 32 which covers "numpy"/"dtype".
    for len in 1..=32usize {
        if p < len + 2 {
            break;
        }
        if bytes[p - len - 2] == OP_SHORT_BINUNICODE
            && bytes[p - len - 1] as usize == len
        {
            return Ok(p - len - 2);
        }
    }
    Err(anyhow!(
        "cannot unwind string push ending at {:#x}",
        p.saturating_sub(1)
    ))
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

fn decode_shape_tuple_backward(bytes: &[u8], end_exclusive: usize) -> Result<Vec<usize>> {
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

/// Consume one integer opcode whose final byte sits at `end_exclusive - 1` and
/// return (value, index_of_first_byte_of_the_opcode).
fn read_int_backward(bytes: &[u8], end_exclusive: usize) -> Result<(usize, usize)> {
    if end_exclusive == 0 {
        bail!("int parse: empty range");
    }
    // BININT (J) - 5 bytes
    if end_exclusive >= 5 && bytes[end_exclusive - 5] == OP_BININT {
        let v = i32::from_le_bytes(bytes[end_exclusive - 4..end_exclusive].try_into().unwrap());
        if v < 0 {
            bail!("negative BININT in shape dim");
        }
        return Ok((v as usize, end_exclusive - 5));
    }
    // BININT2 (M) - 3 bytes
    if end_exclusive >= 3 && bytes[end_exclusive - 3] == OP_BININT2 {
        let v = u16::from_le_bytes(bytes[end_exclusive - 2..end_exclusive].try_into().unwrap());
        return Ok((v as usize, end_exclusive - 3));
    }
    // BININT1 (K) - 2 bytes
    if end_exclusive >= 2 && bytes[end_exclusive - 2] == OP_BININT1 {
        return Ok((bytes[end_exclusive - 1] as usize, end_exclusive - 2));
    }
    Err(anyhow!(
        "no BININT* opcode ending at offset {:#x}",
        end_exclusive - 1
    ))
}

/// Scan forward from just after the dtype tag for the `[\x88|\x89] <payload>`
/// fortran-bool + payload-opcode boundary. Handles SHORT_BINBYTES (C, u8
/// len), BINBYTES (B, u32 len), and BINBYTES8 (\x8e, u64 len).
fn find_payload_forward(bytes: &[u8], after_tag: usize) -> Result<(usize, usize)> {
    let start = after_tag;
    // The dtype state tuple between the tag and the payload is short and
    // predictable; 256 bytes is a generous ceiling.
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
                    let len = bytes[hdr] as usize;
                    let data = hdr + 1;
                    check_payload_bounds(bytes, data, len)?;
                    return Ok((data, len));
                }
                OP_BINBYTES => {
                    let hdr = i + 2;
                    if hdr + 4 > bytes.len() {
                        bail!("truncated BINBYTES header at {:#x}", i + 1);
                    }
                    let len = u32::from_le_bytes(bytes[hdr..hdr + 4].try_into().unwrap()) as usize;
                    let data = hdr + 4;
                    check_payload_bounds(bytes, data, len)?;
                    return Ok((data, len));
                }
                OP_BINBYTES8 => {
                    let hdr = i + 2;
                    if hdr + 8 > bytes.len() {
                        bail!("truncated BINBYTES8 header at {:#x}", i + 1);
                    }
                    let len = u64::from_le_bytes(bytes[hdr..hdr + 8].try_into().unwrap()) as usize;
                    let data = hdr + 8;
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

fn check_payload_bounds(bytes: &[u8], data: usize, len: usize) -> Result<()> {
    if data.saturating_add(len) > bytes.len() {
        bail!(
            "payload {} bytes overruns file (data_off={:#x}, file_len={})",
            len,
            data,
            bytes.len()
        );
    }
    Ok(())
}

// --- QuantizedWeight8bit grouping ------------------------------------------

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

/// For each QuantizedWeight8bit site, label the first int8 tensor that starts
/// after it as `QuantWeight` and the first f32 tensor after it as `QuantScales`.
/// This matches the dataclass field order used in the Grok-1 dumper.
fn assign_qw8_roles(entries: &mut [TensorEntry], sites: &[usize]) {
    for &site in sites {
        let mut weight_idx = None;
        let mut scales_idx = None;
        for (idx, e) in entries.iter().enumerate() {
            if e.offset < site {
                continue;
            }
            match (e.dtype, weight_idx, scales_idx) {
                (Dtype::I8, None, _) => weight_idx = Some(idx),
                (Dtype::F32, _, None) => scales_idx = Some(idx),
                _ => {}
            }
            if weight_idx.is_some() && scales_idx.is_some() {
                break;
            }
        }
        if let Some(i) = weight_idx {
            entries[i].role = Role::QuantWeight;
        }
        if let Some(i) = scales_idx {
            entries[i].role = Role::QuantScales;
        }
    }
}

// --- Output -----------------------------------------------------------------

fn render_table(path: &Path, entries: &[TensorEntry]) {
    println!("\n{}", path.display());
    if entries.is_empty() {
        println!("  (no tensors found)");
        return;
    }
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["Idx", "Role", "Dtype", "Shape", "Offset", "Nbytes"]);

    for (i, e) in entries.iter().enumerate() {
        let shape_str = match e.shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", e.shape[0]),
            _ => format!(
                "({})",
                e.shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        table.add_row(vec![
            Cell::new(i),
            Cell::new(e.role.label()),
            Cell::new(e.dtype.label()),
            Cell::new(shape_str),
            Cell::new(format!("{:#x}", e.offset)),
            Cell::new(e.nbytes),
        ]);
    }
    println!("{table}");
}
