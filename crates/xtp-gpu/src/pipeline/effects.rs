//! The title's effects, read from its shader pack: which programs each pass
//! of each technique runs. `xtp precook` reads them to know, from the game
//! files alone, the shader sets of the title's pipelines, and how many of
//! them the descriptions sessions kept cover ([`super::precook`]).
//!
//! The pack's layout, as read here (little-endian; read from the pack, no
//! game bytes in the repository):
//!
//! - The pack holds SGD5 files one after another, each from its magic.
//! - A file's records (`xtp_game::sgd5`: one program container each) come
//!   in sections, one per stage list ([`LISTS`]) in the lists' order, empty
//!   ones left out. A section's record count is the u32 right before its
//!   first record (before the length byte of the record's name).
//! - After a file's last record come its effects, one after another:
//!   - the name: u8 length (with its NUL), the name, NUL; then a u32;
//!   - one list per stage ([`LISTS`]): u32 count, then each program's
//!     index in its section (u32);
//!   - the effect's state blocks, whose size is not decoded here;
//!   - the techniques: u32 count, then for each the u16 offset of its name
//!     in the effect's string table (bit 15 a flag), a byte, and its pass
//!     count (u8); each technique's passes follow the one before's;
//!   - the passes: u32 count (the techniques' sum), then 24 bytes each,
//!     whose first [`LISTS`] say the program of each list, by its position
//!     in it from 1 (0: none). The other 18 hold the pass's render state,
//!     not read here: the title overrides it when it creates a pipeline
//!     (613 pipelines of a session gave up to 12 rasterizer states for one
//!     value), so the pack does not give a pipeline's state, only its
//!     shaders;
//!   - the effect's variables, of a size not decoded here, then its string
//!     table: u32 size, then NUL-terminated names (the techniques', then
//!     the passes'), which ends where the next effect starts.
//! - The technique table is found after the state blocks as the first
//!   place, within [`STATE_BYTES`], where a technique count, name offsets
//!   that start at 1 and increase, pass counts of 1 or more and a pass
//!   count that is their sum agree, and whose string table is found after
//!   the passes, within [`VARIABLE_BYTES`], with every technique's name in
//!   it. In the title's pack (2026-11-25) the state blocks took 96 to
//!   1,487 bytes and the variables up to 30,730; its 913 effects in two
//!   files hold 28,798 passes.
//! - Which section a list's indices are in: the lists that some effect
//!   uses, in order, are the file's sections, in order (in the title's
//!   pack the vertex, pixel, geometry and compute lists; lists 3 and 4 are
//!   empty everywhere, their stages unknown).
//! - A pass that names a program past its effect's list names one that is
//!   not in the pack ([`Pass::outside`]).

use std::ops::Range;

/// An effect's program lists, one per stage, in the order of the file's
/// sections and of a pass's first bytes.
pub const LISTS: usize = 6;
/// The vertex list: vertex or mesh programs (the vertex slot of the
/// title's pipelines, which it fills with a mesh program in mesh ones).
pub const VERTEX: usize = 0;
pub const PIXEL: usize = 1;
pub const GEOMETRY: usize = 2;
pub const COMPUTE: usize = 5;

/// How far after an effect's lists its technique table may start (its
/// state blocks), and how far after its passes its string table may
/// start (its variables).
pub const STATE_BYTES: usize = 16 << 10;
pub const VARIABLE_BYTES: usize = 256 << 10;

/// How far after a file's last record its first effect may start.
const FIRST_EFFECT: usize = 64 << 10;

/// A record of the pack, as `xtp_game::sgd5` finds it: the offset of its
/// marker, and the length of its name (without its NUL), if it has one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record {
    pub marker: usize,
    pub name: Option<usize>,
}

/// The pack's effects, and all their passes.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Effects {
    pub effects: Vec<Effect>,
    /// Every pass of every effect, in the pack's order.
    pub passes: Vec<Pass>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect {
    pub name: String,
    pub techniques: Vec<Technique>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Technique {
    pub name: String,
    /// Its passes, in [`Effects::passes`].
    pub passes: Range<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    /// Its effect, in [`Effects::effects`].
    pub effect: usize,
    /// Each list's program: its record, in the records given to [`read`].
    pub programs: [Option<usize>; LISTS],
    /// The rest of its entry: its render state, as the effect gives it
    /// (the title overrides it at creation).
    pub state: [u8; 24 - LISTS],
    /// It names programs its effect does not list: they are not in the
    /// pack (in the title's, 18 effects list no programs at all, among them
    /// the `_bvh8` variants of the ray-tracing ones, while their 670
    /// passes name some), and [`Pass::programs`] holds none.
    pub outside: bool,
}

impl Pass {
    /// Whether it draws (a vertex or mesh program), rather than dispatches.
    pub fn draws(&self) -> bool {
        self.programs[VERTEX].is_some()
    }
}

/// The effects of the pack `pack`, whose records `records` are in the
/// order `xtp_game::sgd5::records` gives them (the file's order); why not,
/// if the pack is not laid out as the module says.
pub fn read(pack: &[u8], records: &[Record]) -> Result<Effects, String> {
    let files = files(pack);
    if files.is_empty() {
        return Err("no SGD5 file".into());
    }
    let mut out = Effects::default();
    for (k, &start) in files.iter().enumerate() {
        let end = files.get(k + 1).copied().unwrap_or(pack.len());
        let first = records.partition_point(|r| r.marker < start);
        let last = records.partition_point(|r| r.marker < end);
        if first == last {
            continue;
        }
        file(pack, &records[first..last], first, end, &mut out)
            .map_err(|e| format!("the SGD5 file at {start:#x}: {e}"))?;
    }
    Ok(out)
}

/// An SGD5 file's first bytes.
const MAGIC: &[u8; 4] = b"SGD5";

/// Where each SGD5 file of `pack` starts.
fn files(pack: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(i) = pack[at..].iter().position(|&b| b == MAGIC[0]) {
        let i = at + i;
        if pack[i..].starts_with(MAGIC) {
            out.push(i);
        }
        at = i + 1;
    }
    out
}

fn u32_at(b: &[u8], at: usize) -> Option<usize> {
    let bytes = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes(bytes.try_into().ok()?) as usize)
}

fn u16_at(b: &[u8], at: usize) -> Option<usize> {
    let bytes = b.get(at..at.checked_add(2)?)?;
    Some(usize::from(u16::from_le_bytes(bytes.try_into().ok()?)))
}

/// One SGD5 file's effects, whose records are `records` (the first one is
/// record `base` of the pack's) and which ends at `end`, added to `out`.
fn file(
    pack: &[u8],
    records: &[Record],
    base: usize,
    end: usize,
    out: &mut Effects,
) -> Result<(), String> {
    let sections = sections(pack, records)?;
    let last = records.last().expect("a file with records");
    let size = u32_at(pack, last.marker.wrapping_sub(4)).ok_or("its last record is cut")?;
    let tables = last.marker + size;
    let found = (tables..end.min(tables + FIRST_EFFECT))
        .find_map(|at| effect(pack, at, end, records.len()));
    let Some(mut effect_found) = found else {
        return Err("no effect after its records".into());
    };
    let mut raw = Vec::new();
    loop {
        let next = effect_found.next;
        raw.push(effect_found);
        match effect(pack, next, end, records.len()) {
            Some(e) => effect_found = e,
            None => break,
        }
    }
    // The lists some effect uses are the sections, in order.
    let used: Vec<usize> = (0..LISTS)
        .filter(|&l| raw.iter().any(|e| !e.lists[l].is_empty()))
        .collect();
    if used.len() != sections.len() {
        return Err(format!(
            "{} sections of records but {} stage lists used",
            sections.len(),
            used.len()
        ));
    }
    let mut section: [Option<Range<usize>>; LISTS] = Default::default();
    for (&l, s) in used.iter().zip(&sections) {
        section[l] = Some(s.clone());
    }
    for e in raw {
        let index = out.effects.len();
        let mut techniques = Vec::with_capacity(e.techniques.len());
        let mut at = 0;
        for (name, count) in e.techniques {
            let first = out.passes.len();
            for bytes in &e.passes[at..at + count] {
                let mut programs = [None; LISTS];
                let mut outside = false;
                for (l, program) in programs.iter_mut().enumerate() {
                    let position = usize::from(bytes[l]);
                    if position == 0 {
                        continue;
                    }
                    let Some(&i) = e.lists[l].get(position - 1) else {
                        outside = true;
                        continue;
                    };
                    let s = section[l].as_ref().expect("a used list has its section");
                    if i >= s.len() {
                        return Err(format!(
                            "effect {}: list {l} names program {i} of a section of {}",
                            e.name,
                            s.len()
                        ));
                    }
                    *program = Some(base + s.start + i);
                }
                out.passes.push(Pass {
                    effect: index,
                    programs: if outside { [None; LISTS] } else { programs },
                    state: bytes[LISTS..].try_into().expect("18 bytes"),
                    outside,
                });
            }
            at += count;
            techniques.push(Technique {
                name,
                passes: first..out.passes.len(),
            });
        }
        out.effects.push(Effect {
            name: e.name,
            techniques,
        });
    }
    Ok(())
}

/// The sections of a file's `records`: each its records, from the count
/// before its first one.
fn sections(pack: &[u8], records: &[Record]) -> Result<Vec<Range<usize>>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < records.len() {
        let r = records[i];
        let name = r.name.ok_or_else(|| format!("record {i} has no name"))?;
        // The record's name length byte, then the count before it.
        let count = r
            .marker
            .checked_sub(11 + name)
            .and_then(|at| u32_at(pack, at))
            .ok_or_else(|| format!("no section count before record {i}"))?;
        if count == 0 || i + count > records.len() {
            return Err(format!(
                "a section of {count} records from record {i}, of {}",
                records.len()
            ));
        }
        out.push(i..i + count);
        i += count;
    }
    if out.len() > LISTS {
        return Err(format!("{} sections", out.len()));
    }
    Ok(out)
}

/// An effect as the file holds it.
struct RawEffect {
    name: String,
    lists: [Vec<usize>; LISTS],
    /// Each technique's name and pass count.
    techniques: Vec<(String, usize)>,
    passes: Vec<[u8; 24]>,
    /// Where the next effect would start.
    next: usize,
}

/// The effect at `at` (its name's length byte), before `end`, whose lists
/// index at most `records` programs.
fn effect(pack: &[u8], at: usize, end: usize, records: usize) -> Option<RawEffect> {
    let len = usize::from(*pack.get(at)?);
    let name = pack.get(at + 1..at + len)?;
    if len < 2 || *pack.get(at + len)? != 0 || !identifier(name) {
        return None;
    }
    let mut p = at + 1 + len + 4;
    let mut lists: [Vec<usize>; LISTS] = Default::default();
    for list in &mut lists {
        let n = u32_at(pack, p)?;
        if n > records {
            return None;
        }
        p += 4;
        for k in 0..n {
            let i = u32_at(pack, p + 4 * k)?;
            if i >= records {
                return None;
            }
            list.push(i);
        }
        p += 4 * n;
    }
    // The technique table after the state blocks: the first place that
    // reads as one, with its string table after the passes.
    (p..end.min(p + STATE_BYTES)).find_map(|q| {
        let (techniques, passes, after) = techniques(pack, q, end)?;
        let (names, next) = strings(pack, after, end, &techniques)?;
        Some(RawEffect {
            name: String::from_utf8(name.to_vec()).ok()?,
            lists: lists.clone(),
            techniques: names
                .into_iter()
                .zip(techniques.iter().map(|&(_, count)| count))
                .collect(),
            passes,
            next,
        })
    })
}

/// A technique table as read: each technique's name offset and pass count,
/// the passes, and where they end.
type Table = (Vec<(usize, usize)>, Vec<[u8; 24]>, usize);

/// The technique table at `q`, if one is there.
fn techniques(pack: &[u8], q: usize, end: usize) -> Option<Table> {
    let t = u32_at(pack, q)?;
    if !(1..=4096).contains(&t) || u16_at(pack, q + 4)? & 0x7fff != 1 {
        return None;
    }
    let mut techniques = Vec::with_capacity(t);
    for k in 0..t {
        let at = q + 4 + 4 * k;
        let offset = u16_at(pack, at)? & 0x7fff;
        let count = usize::from(*pack.get(at + 3)?);
        if count == 0 || techniques.last().is_some_and(|&(o, _)| offset <= o) {
            return None;
        }
        techniques.push((offset, count));
    }
    let n = u32_at(pack, q + 4 + 4 * t)?;
    if n != techniques.iter().map(|&(_, c)| c).sum::<usize>() {
        return None;
    }
    let start = q + 8 + 4 * t;
    let after = start.checked_add(24 * n)?;
    if after > end {
        return None;
    }
    let passes = pack[start..after].as_chunks::<24>().0.to_vec();
    Some((techniques, passes, after))
}

/// The string table after `after` (within [`VARIABLE_BYTES`]) that holds a
/// name at each of `techniques`' offsets: those names, and where the table
/// ends.
fn strings(
    pack: &[u8],
    after: usize,
    end: usize,
    techniques: &[(usize, usize)],
) -> Option<(Vec<String>, usize)> {
    let last = techniques.last()?.0;
    (after..end.min(after + VARIABLE_BYTES)).find_map(|t| {
        let size = u32_at(pack, t)?;
        if !(3..=1 << 16).contains(&size) || last >= size {
            return None;
        }
        let table = pack.get(t + 4..t + 4 + size)?;
        if table[0] != 0 || table[size - 1] != 0 || table[size - 2] == 0 {
            return None;
        }
        if !table.iter().all(|&c| c == 0 || c.is_ascii_graphic()) {
            return None;
        }
        let names = techniques
            .iter()
            .map(|&(offset, _)| {
                if table[offset - 1] != 0 {
                    return None;
                }
                let name = table[offset..].split(|&c| c == 0).next()?;
                identifier(name).then(|| String::from_utf8_lossy(name).into_owned())
            })
            .collect::<Option<Vec<_>>>()?;
        Some((names, t + 4 + size))
    })
}

/// A name as the pack's are: letters, digits and underscores.
fn identifier(name: &[u8]) -> bool {
    !name.is_empty() && name.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'_')
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pack being built: its bytes and records.
    #[derive(Default)]
    struct Pack {
        bytes: Vec<u8>,
        records: Vec<Record>,
    }

    impl Pack {
        fn u32(&mut self, v: usize) {
            self.bytes.extend((v as u32).to_le_bytes());
        }

        /// An SGD5 file's magic and a header.
        fn file(&mut self) {
            self.bytes.extend(b"SGD5");
            self.bytes.extend([0x5a; 12]);
        }

        /// A section of `n` records, the count before its first.
        fn section(&mut self, n: usize) {
            for k in 0..n {
                let name = format!("P_{}_{k}", self.records.len());
                if k == 0 {
                    self.u32(n);
                }
                // Some bytes of the record before (its name's length, the
                // name, NUL, kind, size), then the marker and a body.
                self.bytes.push(name.len() as u8 + 1);
                self.bytes.extend(name.as_bytes());
                self.bytes.extend([0, 0x32]);
                self.u32(16);
                let marker = self.bytes.len();
                self.bytes.extend(0x000D_DCE7_u32.to_le_bytes());
                self.u32(16);
                self.u32(4);
                self.bytes.extend([1, 2, 3, 4]);
                self.records.push(Record {
                    marker,
                    name: Some(name.len()),
                });
                // Metadata between records.
                self.bytes.extend([0; 9]);
            }
        }

        /// An effect: its name, lists, techniques (name, passes) and
        /// state blocks of `state` bytes.
        fn effect(
            &mut self,
            name: &str,
            lists: [&[usize]; LISTS],
            techniques: &[(&str, &[[u8; LISTS]])],
            state: usize,
        ) {
            self.bytes.push(name.len() as u8 + 1);
            self.bytes.extend(name.as_bytes());
            self.bytes.push(0);
            self.u32(0x1234);
            for list in lists {
                self.u32(list.len());
                for &i in list {
                    self.u32(i);
                }
            }
            // State blocks: bytes that do not read as a technique table.
            self.bytes.extend((0..state).map(|i| (i % 7) as u8 + 2));
            let mut strings = vec![0u8];
            let mut offsets = Vec::new();
            for (t, _) in techniques {
                offsets.push(strings.len());
                strings.extend(t.as_bytes());
                strings.push(0);
            }
            strings.extend(b"p0\0");
            self.u32(techniques.len());
            for (k, (_, passes)) in techniques.iter().enumerate() {
                let flag = if k % 2 == 1 { 0x8000 } else { 0 };
                self.bytes
                    .extend(((offsets[k] | flag) as u16).to_le_bytes());
                self.bytes.push(k as u8 / 2);
                self.bytes.push(passes.len() as u8);
            }
            self.u32(techniques.iter().map(|(_, p)| p.len()).sum());
            for (_, passes) in techniques {
                for pass in *passes {
                    self.bytes.extend(pass);
                    self.bytes.extend([
                        0, 0, 0, 0xff, 0, 0, 0x40, 1, 0x38, 1, 0, 2, 0xff, 0xff, 0, 0, 0, 0,
                    ]);
                }
            }
            // Variables, then the string table.
            self.bytes.extend([0, 0, 0, 5, 0x93, 0x71, 0x8b, 0xbe]);
            self.u32(strings.len());
            self.bytes.extend(strings);
        }
    }

    /// Two files: the first with vertex, pixel and compute sections and
    /// two effects, the second with a pixel-free one. Each pass names its
    /// programs' records.
    #[test]
    fn effects_name_their_passes_programs() {
        let mut p = Pack::default();
        p.bytes.extend([0xee; 40]);
        p.file();
        p.section(3); // vertex: records 0-2
        p.section(2); // pixel: 3-4
        p.section(2); // compute: 5-6
        p.effect(
            "blit",
            [&[0, 2], &[1, 0], &[], &[], &[], &[1]],
            &[
                ("draw", &[[1, 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0]]),
                ("depth", &[[2, 0, 0, 0, 0, 0]]),
                ("resolve", &[[0, 0, 0, 0, 0, 1]]),
            ],
            200,
        );
        p.effect(
            "other",
            [&[1], &[], &[], &[], &[], &[0]],
            &[("only", &[[1, 0, 0, 0, 0, 1]])],
            96,
        );
        // Programs that are not in the pack.
        p.effect(
            "elsewhere",
            [&[], &[], &[], &[], &[], &[]],
            &[("away", &[[1, 2, 0, 0, 0, 0]])],
            96,
        );
        p.bytes.extend([0; 64]);
        p.file();
        p.section(1); // vertex: record 7
        p.section(1); // compute: record 8
        p.effect(
            "ui",
            [&[0], &[], &[], &[], &[], &[0]],
            &[("t", &[[1, 0, 0, 0, 0, 0]])],
            100,
        );
        p.bytes.extend([0xab; 300]);

        let e = read(&p.bytes, &p.records).unwrap();
        let names: Vec<&str> = e.effects.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["blit", "other", "elsewhere", "ui"]);
        let blit = &e.effects[0];
        let techniques: Vec<(&str, Range<usize>)> = blit
            .techniques
            .iter()
            .map(|t| (t.name.as_str(), t.passes.clone()))
            .collect();
        assert_eq!(
            techniques,
            [("draw", 0..2), ("depth", 2..3), ("resolve", 3..4)]
        );
        let programs: Vec<[Option<usize>; LISTS]> = e.passes.iter().map(|p| p.programs).collect();
        assert_eq!(
            programs,
            [
                [Some(0), Some(4), None, None, None, None],
                [Some(2), Some(3), None, None, None, None],
                [Some(2), None, None, None, None, None],
                [None, None, None, None, None, Some(6)],
                [Some(1), None, None, None, None, Some(5)],
                [None; LISTS],
                [Some(7), None, None, None, None, None],
            ]
        );
        assert!(e.passes[0].draws() && !e.passes[3].draws());
        let outside: Vec<bool> = e.passes.iter().map(|p| p.outside).collect();
        assert_eq!(outside, [false, false, false, false, false, true, false]);
        assert!(!e.passes[5].draws());
        assert_eq!(e.passes[0].state[..6], [0, 0, 0, 0xff, 0, 0]);
        assert_eq!(e.passes[6].effect, 3);
    }

    /// A list naming a program past its section, or more sections than
    /// lists used, is not this layout.
    #[test]
    fn a_pack_laid_out_otherwise_is_refused() {
        let mut p = Pack::default();
        p.file();
        p.section(1);
        p.section(1);
        p.effect(
            "e",
            [&[0], &[1], &[], &[], &[], &[]],
            &[("t", &[[1, 1, 0, 0, 0, 0]])],
            96,
        );
        let e = read(&p.bytes, &p.records).unwrap_err();
        assert!(e.contains("names program 1 of a section of 1"), "{e}");

        let mut p = Pack::default();
        p.file();
        p.section(1);
        p.section(1);
        p.effect(
            "e",
            [&[0], &[], &[], &[], &[], &[]],
            &[("t", &[[1, 0, 0, 0, 0, 0]])],
            96,
        );
        let e = read(&p.bytes, &p.records).unwrap_err();
        assert!(
            e.contains("2 sections of records but 1 stage lists used"),
            "{e}"
        );

        assert_eq!(read(b"no pack", &[]).unwrap_err(), "no SGD5 file");
    }
}
