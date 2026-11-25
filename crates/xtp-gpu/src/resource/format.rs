//! The host formats of title formats: what the host GPU creates a title's
//! textures, views, targets and clear values in. The console's GPU has
//! formats PC D3D12 lacks (its own numbers 116, 117, 189, 190, 191) and
//! uses a PC driver refuses: BTP 6's 2560x1440 B5G5R5A1 target
//! that is also unordered access, and its B4G4R4A4 target (E_INVALIDARG on
//! an NVIDIA driver, first run), and R9G9B9E5 targets (GPUs have none).
//!
//! 1. [`TABLE`] lists the title formats the host may lack, each with its
//!    stand-in: a host format holding the same values with as many bits or
//!    more, and how title-written contents convert into it. A format of the
//!    table is replaced when the host (asked once, `CheckFeatureSupport`)
//!    lacks any use the console gives it ([`CONSOLE_USES`]): then for every
//!    texture, view, target and clear value of it. One host format per
//!    title format keeps copies between textures, and pipelines' targets,
//!    consistent whatever order the title creates them in.
//! 2. Every other format is the host's own. A texture is created only if the
//!    host supports its format for its dimension, flags and sample count
//!    ([`refusal`]); otherwise no driver call is made, and the refusal is
//!    counted by reason for the frame log.
//!
//! A stand-in with an alpha channel the title's format lacks (R9G9B9E5 as
//! RGBA16F, B5G6R5 as RGBA8, [`alpha_one`]) reads alpha as one, as the
//! console does: shader views force it to one, and pipelines rendering to
//! such a target neither write alpha nor blend with it (`pipeline/host.rs`,
//! `blend_without_alpha`).
//!
//! shortcut: a stand-in has more precision than the console's format, so
//! what a pass renders keeps bits the console would round away (B5G5R5A1's
//! one-bit alpha keeps its fraction). R9G9B9E5 holds no negative, infinite
//! or NaN value and its writes clamp them; RGBA16F keeps them, so a pass
//! writing one passes it on (to bloom or temporal passes, say) where the
//! console reads 0: clamping needs the pixel and compute programs to clamp
//! their outputs to such targets (a shader translation change). Unordered
//! access views cannot remap alpha: a typed load of such a texture reads
//! the stored alpha.
//!
//! Typed buffer views keep the title's element size ([`buffer_format`]):
//! their bytes are the title's, never converted, so a stand-in's wider
//! elements would read other bytes. A buffer view in a format of the table
//! is made in the title's own format where the host has it for buffers;
//! where it does not (the console's own numbers, and formats the host lacks
//! for buffers) the view is made null, and counted for the frame log.
//!
//! A view of a typed texture in another format of the texture's family
//! keeps the view's format where D3D12 casts it ([`castable`], [`view_cast`]:
//! sRGB and linear, UINT and UNORM, and so on); views D3D12 does not cast
//! take the texture's own. Each such pair is counted for the frame log
//! ([`note_view_cast`]), render targets with them.

/// `D3D12_FORMAT_SUPPORT1` bits.
pub(crate) mod uses {
    pub(crate) const BUFFER: u32 = 0x1;
    pub(crate) const VERTEX_BUFFER: u32 = 0x2;
    pub(crate) const TEXTURE_1D: u32 = 0x10;
    pub(crate) const TEXTURE_2D: u32 = 0x20;
    pub(crate) const TEXTURE_3D: u32 = 0x40;
    pub(crate) const SHADER_LOAD: u32 = 0x100;
    pub(crate) const SHADER_SAMPLE: u32 = 0x200;
    pub(crate) const RENDER_TARGET: u32 = 0x4000;
    pub(crate) const BLENDABLE: u32 = 0x8000;
    pub(crate) const DEPTH_STENCIL: u32 = 0x1_0000;
    pub(crate) const TYPED_UNORDERED_ACCESS: u32 = 0x200_0000;
}

/// What the console lets a color format of [`TABLE`] do: textures sampled,
/// rendered to with blending, and written as typed unordered access views.
pub(crate) const CONSOLE_USES: u32 = uses::TEXTURE_2D
    | uses::SHADER_SAMPLE
    | uses::RENDER_TARGET
    | uses::BLENDABLE
    | uses::TYPED_UNORDERED_ACCESS;

/// How title-written texels convert into a stand-in (linear rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Convert {
    /// B5G6R5_UNORM to R8G8B8A8_UNORM (alpha one).
    Bgr565,
    /// B5G5R5A1_UNORM to R8G8B8A8_UNORM.
    Bgr5a1,
    /// B4G4R4A4_UNORM to R8G8B8A8_UNORM.
    Bgra4,
    /// R4G4_UNORM (the console's 190) to R8G8_UNORM.
    Rg4,
    /// R10G10B10_SNORM_A2_UNORM (the console's 189) to R16G16B16A16_SNORM.
    Snorm10a2,
    /// R9G9B9E5_SHAREDEXP to R16G16B16A16_FLOAT (alpha one), exactly.
    Rgb9e5,
}

impl Convert {
    /// Bytes of a texel in the title's format, then in the stand-in.
    pub(crate) fn bytes(self) -> (usize, usize) {
        match self {
            Convert::Bgr565 | Convert::Bgr5a1 | Convert::Bgra4 => (2, 4),
            Convert::Rg4 => (1, 2),
            Convert::Snorm10a2 | Convert::Rgb9e5 => (4, 8),
        }
    }

    /// Converts the texels of `from` (title format) into `to` (stand-in),
    /// as many as both hold.
    pub(crate) fn row(self, from: &[u8], to: &mut [u8]) {
        let (a, b) = self.bytes();
        for (from, to) in from.chunks_exact(a).zip(to.chunks_exact_mut(b)) {
            match self {
                Convert::Bgr565 | Convert::Bgr5a1 | Convert::Bgra4 => {
                    let v = u32::from(u16::from_le_bytes([from[0], from[1]]));
                    let [r, g, b, a] = match self {
                        Convert::Bgr565 => [
                            unorm8(v >> 11, 5),
                            unorm8(v >> 5 & 0x3f, 6),
                            unorm8(v & 0x1f, 5),
                            255,
                        ],
                        Convert::Bgr5a1 => [
                            unorm8(v >> 10 & 0x1f, 5),
                            unorm8(v >> 5 & 0x1f, 5),
                            unorm8(v & 0x1f, 5),
                            unorm8(v >> 15, 1),
                        ],
                        _ => [
                            unorm8(v >> 8 & 0xf, 4),
                            unorm8(v >> 4 & 0xf, 4),
                            unorm8(v & 0xf, 4),
                            unorm8(v >> 12, 4),
                        ],
                    };
                    to.copy_from_slice(&[r, g, b, a]);
                }
                Convert::Rg4 => {
                    let v = u32::from(from[0]);
                    to.copy_from_slice(&[unorm8(v & 0xf, 4), unorm8(v >> 4, 4)]);
                }
                Convert::Snorm10a2 => {
                    let v = u32::from_le_bytes([from[0], from[1], from[2], from[3]]);
                    let channels = [
                        snorm16_of_snorm10(v & 0x3ff),
                        snorm16_of_snorm10(v >> 10 & 0x3ff),
                        snorm16_of_snorm10(v >> 20 & 0x3ff),
                        // A two-bit UNORM in [0, 1]: k/3 as a 16-bit SNORM.
                        (((v >> 30) * 32767 + 1) / 3) as i32,
                    ];
                    for (to, c) in to.as_chunks_mut::<2>().0.iter_mut().zip(channels) {
                        *to = (c as u16).to_le_bytes();
                    }
                }
                Convert::Rgb9e5 => {
                    let v = u32::from_le_bytes([from[0], from[1], from[2], from[3]]);
                    let exponent = v >> 27;
                    let channels = [
                        half_of_9e5(v & 0x1ff, exponent),
                        half_of_9e5(v >> 9 & 0x1ff, exponent),
                        half_of_9e5(v >> 18 & 0x1ff, exponent),
                        HALF_ONE,
                    ];
                    for (to, c) in to.as_chunks_mut::<2>().0.iter_mut().zip(channels) {
                        *to = c.to_le_bytes();
                    }
                }
            }
        }
    }
}

/// 1.0 as a 16-bit float.
const HALF_ONE: u16 = 0x3c00;

/// A channel of R9G9B9E5_SHAREDEXP, `mantissa / 2^9 * 2^(exponent - 15)`
/// (no implicit one), as a 16-bit float. Every such value is one exactly:
/// at most 9 significant bits, from 2^-24 (the smallest subnormal) to
/// 65,408 (below the largest, 65,504).
fn half_of_9e5(mantissa: u32, exponent: u32) -> u16 {
    if mantissa == 0 {
        return 0;
    }
    // The value is 2^(top + exponent - 24) * (mantissa / 2^top).
    let top = 31 - mantissa.leading_zeros();
    let biased = (top + exponent) as i32 - 9;
    if biased >= 1 {
        (biased as u16) << 10 | ((mantissa << (10 - top)) & 0x3ff) as u16
    } else {
        // Subnormal: multiples of 2^-24, `mantissa * 2^exponent` of them
        // (below 2^10 here).
        (mantissa << exponent) as u16
    }
}

/// A `bits`-bit UNORM value as an 8-bit one: the same value, rounded to
/// nearest as the GPU converts.
fn unorm8(v: u32, bits: u32) -> u8 {
    let max = (1 << bits) - 1;
    ((v * 255 + max / 2) / max) as u8
}

/// A 10-bit SNORM value as a 16-bit one (both: -max is -1, the lowest
/// code too).
fn snorm16_of_snorm10(v: u32) -> i32 {
    let s = ((v << 22) as i32) >> 22;
    let s = s.max(-511);
    let wide = s * 32767;
    // Rounded to nearest, away from zero at halves.
    (wide + if wide < 0 { -255 } else { 255 }) / 511
}

/// A title format the host may lack.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Entry {
    pub(crate) title: u32,
    pub(crate) name: &'static str,
    /// Whether the PC's format of this number is the console's (else the
    /// number means another format, or none, on PC).
    pub(crate) on_pc: bool,
    /// The stand-in and how title-written contents convert into it (none:
    /// they are not uploaded); no stand-in: textures of it are refused.
    pub(crate) stand_in: Option<(u32, Option<Convert>)>,
}

const R16G16B16A16_FLOAT: u32 = 10;
const R16G16B16A16_SNORM: u32 = 13;
const R8G8B8A8_UNORM: u32 = 28;
const R8G8_UNORM: u32 = 49;

/// The substitution table.
pub(crate) const TABLE: [Entry; 9] = [
    // GPUs neither render to nor write R9G9B9E5 (the RTX 4080 and the
    // VIctory testers' cards; WARP does): the title's HDR scene and lighting
    // targets fail there without a stand-in (VIctory #492).
    // RGBA16F holds every value exactly (xcl takes R11G11B10_FLOAT, which
    // rounds 9-bit mantissas to 6 and 5 bits), at twice the memory.
    Entry {
        title: 67,
        name: "R9G9B9E5_SHAREDEXP",
        on_pc: true,
        stand_in: Some((R16G16B16A16_FLOAT, Some(Convert::Rgb9e5))),
    },
    Entry {
        title: 85,
        name: "B5G6R5_UNORM",
        on_pc: true,
        stand_in: Some((R8G8B8A8_UNORM, Some(Convert::Bgr565))),
    },
    Entry {
        title: 86,
        name: "B5G5R5A1_UNORM",
        on_pc: true,
        stand_in: Some((R8G8B8A8_UNORM, Some(Convert::Bgr5a1))),
    },
    Entry {
        title: 115,
        name: "B4G4R4A4_UNORM",
        on_pc: true,
        stand_in: Some((R8G8B8A8_UNORM, Some(Convert::Bgra4))),
    },
    // shortcut: the console's 10-bit floats are not decoded on upload, so
    // title-written textures of them are not uploaded (targets are the
    // GPU's own).
    Entry {
        title: 116,
        name: "R10G10B10_7E3_A2_FLOAT",
        on_pc: false,
        stand_in: Some((R16G16B16A16_FLOAT, None)),
    },
    Entry {
        title: 117,
        name: "R10G10B10_6E4_A2_FLOAT",
        on_pc: false,
        stand_in: Some((R16G16B16A16_FLOAT, None)),
    },
    Entry {
        title: 189,
        name: "R10G10B10_SNORM_A2_UNORM",
        on_pc: false,
        stand_in: Some((R16G16B16A16_SNORM, Some(Convert::Snorm10a2))),
    },
    Entry {
        title: 190,
        name: "R4G4_UNORM",
        on_pc: false,
        stand_in: Some((R8G8_UNORM, Some(Convert::Rg4))),
    },
    // 32 bits a texel (24 and 8): what it holds is not known, and the PC's
    // 191 (A4B4G4R4) is another format.
    Entry {
        title: 191,
        name: "console format 191",
        on_pc: false,
        stand_in: None,
    },
];

/// Whether title format `title` has no alpha while the host's format for
/// it does: R9G9B9E5 and B5G6R5 where they stand in. The console reads such
/// a format's alpha as one.
pub(crate) fn alpha_one(title: u32) -> bool {
    matches!(title, 67 | 85) && matches!(HostCaps.choice(title), Some(Choice::StandIn(..)))
}

/// A view's `Shader4ComponentMapping` with every component that reads the
/// texture's alpha reading one instead (D3D12's encoding: three bits per
/// component, 3 the source alpha, 5 the value one).
pub(crate) fn mapping_alpha_one(mapping: u32) -> u32 {
    const ALWAYS: u32 = 1 << 12;
    const SOURCE_ALPHA: u32 = 3;
    const ONE: u32 = 5;
    (0..4).fold(ALWAYS, |out, i| {
        let from = mapping >> (3 * i) & 7;
        out | (if from == SOURCE_ALPHA { ONE } else { from }) << (3 * i)
    })
}

/// The bits of each channel of host format `format`, in the order of a
/// clear's four values (red, green, blue, alpha); 0 for a channel it lacks.
/// `None` for a format without such channels (depth with stencil,
/// block-compressed, shared exponent, video).
pub(crate) fn channel_bits(format: u32) -> Option<[u32; 4]> {
    Some(match format {
        1..=4 => [32; 4],
        5..=8 => [32, 32, 32, 0],
        9..=14 => [16; 4],
        15..=18 => [32, 32, 0, 0],
        23..=25 | 89 => [10, 10, 10, 2],
        26 => [11, 11, 10, 0],
        27..=32 | 87 | 90 | 91 => [8; 4],
        33..=38 => [16, 16, 0, 0],
        39..=43 => [32, 0, 0, 0],
        48..=52 => [8, 8, 0, 0],
        53..=59 => [16, 0, 0, 0],
        60..=64 => [8, 0, 0, 0],
        65 => [0, 0, 0, 8],
        85 => [5, 6, 5, 0],
        86 => [5, 5, 5, 1],
        88 | 92 | 93 => [8, 8, 8, 0],
        115 => [4; 4],
        _ => return None,
    })
}

/// The values of an integer UAV clear (`ClearUnorderedAccessViewUint`) of
/// a view in host format `format`, each cut to the low bits its channel
/// holds, which is what D3D documents and WARP writes. Hosts differ past
/// that: the L4's driver (NVIDIA 596.86) saturates each value to its
/// channel's largest unsigned value, signed formats included (0x104 gave
/// 0xff in an 8-bit channel, and so did -128 in a signed one), and writes
/// a value that fits as it is (docs/analysis/hardware-tests.md). A format
/// without plain channels keeps its values.
// shortcut: a stand-in (TABLE) has wider channels than the title's format,
// and the clear keeps the stand-in's bits, not the title's.
pub(crate) fn uint_clear(format: u32, values: [u32; 4]) -> [u32; 4] {
    let Some(bits) = channel_bits(format) else {
        return values;
    };
    std::array::from_fn(|i| match bits[i] {
        0 | 32.. => values[i],
        n => values[i] & ((1 << n) - 1),
    })
}

/// What the host uses for a format of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Choice {
    /// The format itself.
    Own,
    /// Its stand-in.
    StandIn(u32, Option<Convert>),
    /// Nothing: textures of it are refused.
    Refused,
}

/// The choice for `entry` given the host's `support` of each format.
pub(crate) fn choose(entry: &Entry, support: impl Fn(u32) -> u32) -> Choice {
    if entry.on_pc && support(entry.title) & CONSOLE_USES == CONSOLE_USES {
        return Choice::Own;
    }
    match entry.stand_in {
        Some((format, convert)) => Choice::StandIn(format, convert),
        None if entry.on_pc => Choice::Own,
        None => Choice::Refused,
    }
}

/// The table's entry for `format`.
pub(crate) fn entry(format: u32) -> Option<&'static Entry> {
    TABLE.iter().find(|e| e.title == format)
}

/// The host device's choices for the table, made once (none without a
/// host: stand-ins for all).
fn choices() -> &'static [Choice; TABLE.len()] {
    static CHOICES: std::sync::OnceLock<[Choice; TABLE.len()]> = std::sync::OnceLock::new();
    CHOICES.get_or_init(|| {
        let choices = TABLE.map(|e| choose(&e, |f| HostCaps.support1(f)));
        for (e, c) in TABLE.iter().zip(&choices) {
            if *c != Choice::Own {
                tracing::info!(
                    format = e.title,
                    name = e.name,
                    support = format_args!("{:#x}", HostCaps.support1(e.title)),
                    choice = ?c,
                    "host format stands in for a title format"
                );
            }
        }
        choices
    })
}

/// How contents of title format `title` convert into the host format
/// `host`, if `host` stands in for it on the host device: `Some(None)`
/// when they cannot.
pub(crate) fn conversion(title: u32, host: u32) -> Option<Option<Convert>> {
    match HostCaps.choice(title)? {
        Choice::StandIn(format, convert) if format == host => Some(convert),
        _ => None,
    }
}

/// How a typed format's channels read, for D3D12's casting rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reads {
    Float,
    /// UNORM, sRGB included.
    Unorm,
    Snorm,
    Uint,
    Sint,
}

/// How the channels of typed color format `format` read; `None` for a
/// typeless format, depth and stencil ones (D3D12 lists casts of them among
/// its known failures), BC6H, and formats outside the families.
fn reads(format: u32) -> Option<Reads> {
    Some(match format {
        2 | 6 | 10 | 16 | 34 | 41 | 54 => Reads::Float,
        11 | 24 | 28 | 29 | 35 | 49 | 56 | 61 | 65 | 71 | 72 | 74 | 75 | 77 | 78 | 80 | 83 | 87
        | 88 | 91 | 93 | 98 | 99 => Reads::Unorm,
        13 | 31 | 37 | 51 | 58 | 63 | 81 | 84 => Reads::Snorm,
        3 | 7 | 12 | 17 | 25 | 30 | 36 | 42 | 50 | 57 | 62 => Reads::Uint,
        4 | 8 | 14 | 18 | 32 | 38 | 43 | 52 | 59 | 64 => Reads::Sint,
        _ => return None,
    })
}

/// Whether D3D12 lets a view in format `view` read or write a resource
/// created in typed format `own` (host formats), where the host casts
/// fully typed formats ([`Caps::fully_typed_casting`]): another format of
/// the same family, unless one of the two is a float format and the other
/// not, or one is UNORM and the other SNORM (D3D12's relaxed casting rules,
/// DirectX-Specs "RelaxedCasting"). Depth and stencil formats, BC6H and
/// typeless formats are left out ([`reads`]).
pub(crate) fn castable(own: u32, view: u32) -> bool {
    if own == view {
        return true;
    }
    let (Some(a), Some(b)) = (reads(own), reads(view)) else {
        return false;
    };
    super::family(own) == super::family(view)
        && (a == Reads::Float) == (b == Reads::Float)
        && !matches!(
            (a, b),
            (Reads::Unorm, Reads::Snorm) | (Reads::Snorm, Reads::Unorm)
        )
}

/// Whether host views of typed textures cast at all (`XTP_VIEW_CAST=0`
/// gives every such view the texture's format, as before, to compare).
fn casts_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XTP_VIEW_CAST").map_or(true, |v| v != "0");
        if !on {
            tracing::warn!(
                "XTP_VIEW_CAST=0: views of typed textures in another format of their family take the texture's format"
            );
        }
        on
    })
}

/// Whether a shader view (an unordered access one if `unordered`) in host
/// format `view` of a host texture in typed format `own` keeps its format:
/// D3D12 casts it ([`castable`]) on this host, which can read (or write)
/// that format through such a view.
pub(crate) fn view_cast(own: u32, view: u32, unordered: bool) -> bool {
    view_cast_with(&HostCaps, own, view, unordered) && casts_on()
}

/// [`view_cast`] on a device that can do `caps`.
fn view_cast_with(caps: &impl Caps, own: u32, view: u32, unordered: bool) -> bool {
    let needs = if unordered {
        uses::TYPED_UNORDERED_ACCESS
    } else {
        uses::SHADER_LOAD
    };
    own == view
        || castable(own, view) && caps.fully_typed_casting() && caps.support1(view) & needs != 0
}

/// The host format of a typed buffer view (an unordered access one if
/// `unordered`) of title format `title`: the same number but for a format
/// of [`TABLE`], which keeps the title's element size (a buffer's bytes are
/// the title's: a stand-in's wider elements would read other bytes). Such
/// a format is the title's own where the host has it for buffer views;
/// else the typeless format of the title's element size, of which the host
/// makes no typed view (the slot reads zeros), counted for the frame log.
pub(crate) fn buffer_format(title: u32, unordered: bool) -> u32 {
    buffer_format_with(&HostCaps, title, unordered)
}

/// [`buffer_format`] on a device that can do `caps`.
fn buffer_format_with(caps: &impl Caps, title: u32, unordered: bool) -> u32 {
    // DXGI ne propose aucun élément typé de trois octets. UNKNOWN sans
    // stride donne une vue nulle, y compris quand le format 45 est disponible.
    if (118..=120).contains(&title) {
        note_buffer_without_format(title, unordered);
        return 0;
    }
    let Some(e) = entry(title) else {
        return title;
    };
    let needs = if unordered {
        uses::BUFFER | uses::TYPED_UNORDERED_ACCESS
    } else {
        uses::BUFFER
    };
    if e.on_pc && caps.support1(title) & needs == needs {
        return title;
    }
    note_buffer_without_format(title, unordered);
    match super::element(title).map(|(bytes, _)| bytes) {
        Some(1) => 60,
        Some(2) => 53,
        Some(8) => 15,
        _ => 39,
    }
}

/// What a view in another format of its texture's family is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ViewUse {
    /// A shader resource view.
    Shader,
    /// An unordered access view.
    Unordered,
    /// A render target view.
    Target,
    /// A format the title lists as castable when it creates the resource.
    Listed,
}

/// Counts a view in host format `view` of a host texture in typed format
/// `own` (another format of its family), and whether it keeps its own
/// format (`cast`: the host casts it; for a target or a listed castable
/// format, whether the host can view the texture so at all). The first of
/// each kind is logged.
pub(crate) fn note_view_cast(own: u32, view: u32, how: ViewUse, cast: bool) {
    let first = {
        let mut counts = counts();
        let n = counts.casts.entry((how, own, view, cast)).or_default();
        *n += 1;
        *n == 1
    };
    if first {
        tracing::info!(
            ?how,
            texture = own,
            view,
            cast,
            "a view in another format of its texture's family (first of its kind)"
        );
    }
}

/// Counts a shader view of a whole typeless host copy that the title made
/// without a description, made in format `read` (it was null before).
pub(crate) fn note_whole_view(read: u32) {
    *counts().wholes.entry(read).or_default() += 1;
}

/// Counts a typed buffer view in title format `title` the host has no
/// buffer format for ([`buffer_format`]); the first of each is logged.
fn note_buffer_without_format(title: u32, unordered: bool) {
    let first = {
        let mut counts = counts();
        let n = counts.buffers.entry((title, unordered)).or_default();
        *n += 1;
        *n == 1
    };
    if first {
        tracing::warn!(
            format = title,
            unordered,
            "a typed buffer view in a title format the host has no buffer format for: null"
        );
    }
}

/// What a device can do with formats.
pub(crate) trait Caps {
    /// `D3D12_FORMAT_SUPPORT1` of `format` (0: none).
    fn support1(&self, format: u32) -> u32;

    /// Whether `count` samples of `format` exist (1: always).
    fn samples(&self, format: u32, count: u32) -> bool;

    /// Whether views of a typed resource may name another format of its
    /// family (`CastingFullyTypedFormatSupported`, D3D12 options 3; the
    /// runtime reports it for every driver since Windows 10 1703), within
    /// the rules of [`castable`].
    fn fully_typed_casting(&self) -> bool {
        false
    }

    /// The choice for a format of [`TABLE`]; `None` for the others.
    fn choice(&self, format: u32) -> Option<Choice> {
        entry(format).map(|e| choose(e, |f| self.support1(f)))
    }

    /// The device's support of a resource format: its own, and its
    /// family's.
    fn support(&self, format: u32) -> Support {
        let own = self.support1(format);
        let head = super::family(format);
        let family = match super::family_members(head) {
            Some(members) => {
                members.fold(own | self.support1(head), |all, f| all | self.support1(f))
            }
            None => own,
        };
        Support {
            own,
            family,
            typeless: head == format && super::family_members(format).is_some(),
        }
    }

    /// The device's format for title format `format`: its stand-in if it
    /// has one there.
    fn host_format(&self, format: u32) -> u32 {
        match self.choice(format) {
            Some(Choice::StandIn(host, _)) => host,
            _ => super::pc_vertex_format(format),
        }
    }
}

/// The host device's, asked once per format.
pub(crate) struct HostCaps;

impl Caps for HostCaps {
    fn support1(&self, format: u32) -> u32 {
        #[cfg(windows)]
        {
            static SUPPORT: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
            let table = SUPPORT.get_or_init(|| match crate::host::host() {
                Some(host) => (0..=LAST_FORMAT).map(|f| host.device.support1(f)).collect(),
                None => Vec::new(),
            });
            table.get(format as usize).copied().unwrap_or(0)
        }
        #[cfg(not(windows))]
        {
            let _ = format;
            0
        }
    }

    fn samples(&self, format: u32, count: u32) -> bool {
        #[cfg(windows)]
        {
            count <= 1 || crate::host::host().is_some_and(|h| h.device.samples(format, count))
        }
        #[cfg(not(windows))]
        {
            let _ = format;
            count <= 1
        }
    }

    fn choice(&self, format: u32) -> Option<Choice> {
        let at = TABLE.iter().position(|e| e.title == format)?;
        Some(choices()[at])
    }

    fn fully_typed_casting(&self) -> bool {
        #[cfg(windows)]
        {
            static CASTING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CASTING.get_or_init(|| {
                let casting = crate::host::host().is_some_and(|h| h.device.fully_typed_casting());
                tracing::info!(
                    casting,
                    "host casts views of typed resources within a family"
                );
                casting
            })
        }
        #[cfg(not(windows))]
        false
    }
}

/// The highest format number asked about.
#[cfg(windows)]
const LAST_FORMAT: u32 = 191;

#[cfg(windows)]
impl Caps for windows::Win32::Graphics::Direct3D12::ID3D12Device {
    fn support1(&self, format: u32) -> u32 {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        let mut data = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
            Format: DXGI_FORMAT(format as i32),
            ..Default::default()
        };
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            self.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_SUPPORT,
                (&raw mut data).cast(),
                size_of_val(&data) as u32,
            )
        };
        if asked.is_ok() {
            data.Support1.0 as u32
        } else {
            0
        }
    }

    fn samples(&self, format: u32, count: u32) -> bool {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        if count <= 1 {
            return true;
        }
        let mut data = D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS {
            Format: DXGI_FORMAT(format as i32),
            SampleCount: count,
            ..Default::default()
        };
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            self.CheckFeatureSupport(
                D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                (&raw mut data).cast(),
                size_of_val(&data) as u32,
            )
        };
        asked.is_ok() && data.NumQualityLevels > 0
    }

    fn fully_typed_casting(&self) -> bool {
        use windows::Win32::Graphics::Direct3D12::*;
        let mut data = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
        // SAFETY: the query writes its own structure of the given size.
        let asked = unsafe {
            self.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS3,
                (&raw mut data).cast(),
                size_of_val(&data) as u32,
            )
        };
        asked.is_ok() && data.CastingFullyTypedFormatSupported.as_bool()
    }
}

/// What a device supports of a resource format: the format's own uses, its
/// family's (its own and those of the formats it can be viewed in), and
/// whether it is typeless.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Support {
    pub(crate) own: u32,
    pub(crate) family: u32,
    pub(crate) typeless: bool,
}

/// Why the host cannot create a texture of a format it supports as
/// `support`, of `dimension` (2 1D, 3 2D, 4 3D) with resource `flags`;
/// `None` if it can. As the D3D12 runtime checks (measured on an RTX 4080
/// and WARP): dimension, render target and depth by the family; unordered
/// access by the format itself, by the family for a typeless one.
pub(crate) fn refusal(support: Support, dimension: u32, flags: u32) -> Option<&'static str> {
    const RENDER_TARGET: u32 = 1;
    const DEPTH: u32 = 2;
    const UNORDERED: u32 = 4;
    let Support {
        own,
        family,
        typeless,
    } = support;
    if family == 0 {
        return Some("the host has no such format");
    }
    let shape = match dimension {
        2 => uses::TEXTURE_1D,
        3 => uses::TEXTURE_2D,
        4 => uses::TEXTURE_3D,
        _ => return Some("not a texture dimension"),
    };
    let needs = [
        (true, family, shape, "no textures of this dimension"),
        (
            flags & RENDER_TARGET != 0,
            family,
            uses::RENDER_TARGET,
            "no render targets",
        ),
        (
            flags & DEPTH != 0,
            family,
            uses::DEPTH_STENCIL,
            "no depth-stencil targets",
        ),
        (
            flags & UNORDERED != 0,
            if typeless { family } else { own },
            uses::TYPED_UNORDERED_ACCESS,
            "no unordered access",
        ),
    ];
    needs
        .iter()
        .find(|&&(needed, has, bit, _)| needed && has & bit == 0)
        .map(|&(_, _, _, why)| why)
}

/// Textures refused before the driver, and made in a stand-in, since the
/// start, by reason; views in another format of their texture's family,
/// and typed buffer views without a host format (for the frame log).
static COUNTS: std::sync::Mutex<Counts> = std::sync::Mutex::new(Counts {
    refused: std::collections::BTreeMap::new(),
    stood_in: std::collections::BTreeMap::new(),
    casts: std::collections::BTreeMap::new(),
    buffers: std::collections::BTreeMap::new(),
    wholes: std::collections::BTreeMap::new(),
});

struct Counts {
    refused: std::collections::BTreeMap<String, u64>,
    stood_in: std::collections::BTreeMap<(u32, u32), u64>,
    /// By use, texture format, view format and whether it kept its own.
    casts: std::collections::BTreeMap<(ViewUse, u32, u32, bool), u64>,
    /// By title format and whether unordered access.
    buffers: std::collections::BTreeMap<(u32, bool), u64>,
    /// Whole views of typeless host copies made without a description, by
    /// the format they read.
    wholes: std::collections::BTreeMap<u32, u64>,
}

fn counts() -> std::sync::MutexGuard<'static, Counts> {
    COUNTS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts a texture refused for `why`; true for the first of that reason
/// (which the caller logs with the description).
pub(crate) fn note_refused(why: String) -> bool {
    let mut counts = counts();
    let n = counts.refused.entry(why).or_default();
    *n += 1;
    *n == 1
}

/// Counts a texture of title format `title` made in `host`.
pub(crate) fn note_stand_in(title: u32, host: u32) {
    *counts().stood_in.entry((title, host)).or_default() += 1;
}

/// Host textures alive: their number and bytes, the bytes stand-in
/// formats take beyond the title's, and the number and bytes of those in
/// the texture pool.
static LIVE: [std::sync::atomic::AtomicI64; 5] =
    [const { std::sync::atomic::AtomicI64::new(0) }; 5];

/// Counts a host texture of `bytes` made (`alive`) or freed: `extra` of
/// them for its stand-in format, `pooled` if placed in the texture pool.
pub(crate) fn note_texture(alive: bool, bytes: u64, extra: u64, pooled: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let sign = if alive { 1 } else { -1 };
    LIVE[0].fetch_add(sign, Relaxed);
    LIVE[1].fetch_add(sign * bytes as i64, Relaxed);
    LIVE[2].fetch_add(sign * extra as i64, Relaxed);
    if pooled {
        LIVE[3].fetch_add(sign, Relaxed);
        LIVE[4].fetch_add(sign * bytes as i64, Relaxed);
    }
}

/// The frame log's texture part: host textures alive (in the pool or
/// committed, and what stand-ins add), the pool's heaps, and textures made
/// in a stand-in and refused since the start (`None` if none of them).
pub(crate) fn report() -> Option<String> {
    use std::sync::atomic::Ordering::Relaxed;
    let [live, bytes, extra, pooled, pooled_bytes] = [0, 1, 2, 3, 4].map(|i| LIVE[i].load(Relaxed));
    let counts = counts();
    if counts.refused.is_empty()
        && counts.stood_in.is_empty()
        && counts.casts.is_empty()
        && counts.buffers.is_empty()
        && counts.wholes.is_empty()
        && live == 0
    {
        return None;
    }
    #[cfg(windows)]
    let pool = {
        let (asked, refused) = super::residency::counts();
        let pool =
            super::pool::global_if_made().map_or_else(String::new, |p| format!("; {}", p.report()));
        format!("; residency asked {asked} times, refused {refused}{pool}")
    };
    #[cfg(not(windows))]
    let pool = String::new();
    let pool = format!(
        " ({pooled} pooled, {} MiB; {} committed, {} MiB; stand-in formats +{} MiB){pool}",
        pooled_bytes >> 20,
        live - pooled,
        (bytes - pooled_bytes) >> 20,
        extra >> 20
    );
    let stood_in: Vec<String> = counts
        .stood_in
        .iter()
        .map(|((title, host), n)| format!("{n} of {title} as {host}"))
        .collect();
    let refused: Vec<String> = counts
        .refused
        .iter()
        .map(|(why, n)| format!("{n} {why}"))
        .collect();
    let casts: Vec<String> = counts
        .casts
        .iter()
        .map(|(&(how, own, view, cast), n)| {
            let what = match (how, cast) {
                (_, true) => "cast",
                (ViewUse::Target | ViewUse::Listed, false) => "not castable",
                (_, false) => "as the texture's",
            };
            let how = match how {
                ViewUse::Shader => "srv",
                ViewUse::Unordered => "uav",
                ViewUse::Target => "rtv",
                ViewUse::Listed => "listed",
            };
            format!("{n} {how} of {own} as {view} {what}")
        })
        .collect();
    let buffers: Vec<String> = counts
        .buffers
        .iter()
        .map(|(&(title, unordered), n)| {
            format!("{n} {} of {title}", if unordered { "uav" } else { "srv" })
        })
        .collect();
    let wholes: Vec<String> = counts
        .wholes
        .iter()
        .map(|(read, n)| format!("{n} as {read}"))
        .collect();
    Some(format!(
        "host textures: {live} alive, {} MiB{pool}; so far: stand-in formats [{}], refused [{}], views in another format of their texture's family [{}], typed buffer views without a host format (null) [{}], typeless copies viewed whole without a description [{}]",
        bytes >> 20,
        stood_in.join(", "),
        refused.join("; "),
        casts.join(", "),
        buffers.join(", "),
        wholes.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint_clears_keep_the_low_bits_of_each_channel() {
        assert_eq!(uint_clear(28, [1, 2, 3, 0x104]), [1, 2, 3, 4]);
        // Signed channels too: the bits, not a clamped value.
        assert_eq!(
            uint_clear(32, [0xffff_ff80, 0x17f, 0x80, u32::MAX]),
            [0x80, 0x7f, 0x80, 0xff]
        );
        assert_eq!(
            uint_clear(25, [0x401, 0x3ff, 0x7ff, 5]),
            [1, 0x3ff, 0x3ff, 1]
        );
        // A channel the format lacks keeps its value (the host ignores it).
        assert_eq!(
            uint_clear(36, [0x1_2345, 0xffff_0001, 9, 9]),
            [0x2345, 1, 9, 9]
        );
        assert_eq!(uint_clear(85, [0x3f; 4]), [0x1f, 0x3f, 0x1f, 0x3f]);
        assert_eq!(uint_clear(42, [u32::MAX; 4]), [u32::MAX; 4]);
        // BC1: no plain channels.
        assert_eq!(uint_clear(71, [0x104; 4]), [0x104; 4]);
    }

    /// Each format's channels fill its element, but for the unused X bits
    /// of B8G8R8X8.
    #[test]
    fn channel_bits_fill_the_element() {
        for format in 1..=191 {
            let (Some(bits), Some((bytes, 1))) =
                (channel_bits(format), crate::resource::element(format))
            else {
                continue;
            };
            let unused = if matches!(format, 88 | 92 | 93) { 8 } else { 0 };
            assert_eq!(
                bits.iter().sum::<u32>() + unused,
                8 * bytes as u32,
                "format {format}"
            );
        }
    }

    /// What an NVIDIA RTX 40 reports for the formats of the first run's
    /// refusals (sampling, targets, blending; no unordered access, and no
    /// B4G4R4A4 targets).
    fn nvidia(format: u32) -> u32 {
        use uses::*;
        match format {
            85 | 86 => TEXTURE_2D | SHADER_SAMPLE | RENDER_TARGET | BLENDABLE,
            115 => TEXTURE_2D | SHADER_SAMPLE,
            28 | 10 | 13 | 49 => CONSOLE_USES,
            _ => 0,
        }
    }

    #[cfg(windows)]
    #[test]
    fn use_bits_are_d3d12s() {
        use windows::Win32::Graphics::Direct3D12::*;
        for (ours, theirs) in [
            (uses::BUFFER, D3D12_FORMAT_SUPPORT1_BUFFER),
            (uses::SHADER_LOAD, D3D12_FORMAT_SUPPORT1_SHADER_LOAD),
            (uses::VERTEX_BUFFER, D3D12_FORMAT_SUPPORT1_IA_VERTEX_BUFFER),
            (uses::TEXTURE_1D, D3D12_FORMAT_SUPPORT1_TEXTURE1D),
            (uses::TEXTURE_2D, D3D12_FORMAT_SUPPORT1_TEXTURE2D),
            (uses::TEXTURE_3D, D3D12_FORMAT_SUPPORT1_TEXTURE3D),
            (uses::SHADER_SAMPLE, D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE),
            (uses::RENDER_TARGET, D3D12_FORMAT_SUPPORT1_RENDER_TARGET),
            (uses::BLENDABLE, D3D12_FORMAT_SUPPORT1_BLENDABLE),
            (uses::DEPTH_STENCIL, D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL),
            (
                uses::TYPED_UNORDERED_ACCESS,
                D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW,
            ),
        ] {
            assert_eq!(ours, theirs.0 as u32);
        }
    }

    #[test]
    fn the_first_runs_refused_formats_stand_in() {
        let choice = |f| choose(entry(f).unwrap(), nvidia);
        assert_eq!(choice(86), Choice::StandIn(28, Some(Convert::Bgr5a1)));
        assert_eq!(choice(115), Choice::StandIn(28, Some(Convert::Bgra4)));
        assert_eq!(choice(85), Choice::StandIn(28, Some(Convert::Bgr565)));
        let typed = |f| Support {
            own: nvidia(f),
            family: nvidia(f),
            typeless: false,
        };
        // The stand-in takes what the first run asked for.
        assert_eq!(refusal(typed(28), 3, 1 | 4), None);
        assert_eq!(refusal(typed(28), 3, 1), None);
        // And the formats themselves would have been refused.
        assert_eq!(refusal(typed(86), 3, 1 | 4), Some("no unordered access"));
        assert_eq!(refusal(typed(115), 3, 1), Some("no render targets"));
    }

    #[test]
    fn a_host_with_every_use_keeps_the_format() {
        let all = |_| CONSOLE_USES;
        for e in TABLE.iter().filter(|e| e.on_pc) {
            assert_eq!(choose(e, all), Choice::Own, "{}", e.name);
        }
        // The console's own numbers mean something else on PC, or nothing.
        assert_eq!(choose(entry(116).unwrap(), all), Choice::StandIn(10, None));
        assert_eq!(
            choose(entry(190).unwrap(), all),
            Choice::StandIn(49, Some(Convert::Rg4))
        );
        assert_eq!(choose(entry(191).unwrap(), all), Choice::Refused);
    }

    #[test]
    fn every_entry_is_sized_as_the_title_has_it() {
        for e in &TABLE {
            let (bytes, block) = crate::resource::element(e.title).unwrap();
            assert_eq!(block, 1, "{}", e.name);
            if let Some((host, Some(convert))) = e.stand_in {
                let (from, to) = convert.bytes();
                assert_eq!(from as u64, bytes, "{}", e.name);
                assert_eq!(Some(to as u64), crate::resource::element(host).map(|x| x.0));
            }
        }
    }

    #[test]
    fn refusals_name_the_missing_use() {
        use uses::*;
        let typed = |bits| Support {
            own: bits,
            family: bits,
            typeless: false,
        };
        assert_eq!(refusal(typed(0), 3, 0), Some("the host has no such format"));
        assert_eq!(
            refusal(typed(TEXTURE_2D), 4, 0),
            Some("no textures of this dimension")
        );
        assert_eq!(
            refusal(typed(TEXTURE_2D), 3, 2),
            Some("no depth-stencil targets")
        );
        assert_eq!(refusal(typed(TEXTURE_2D | DEPTH_STENCIL), 3, 2), None);
        assert_eq!(
            refusal(typed(TEXTURE_2D), 1, 0),
            Some("not a texture dimension")
        );
        // R32_FLOAT as a depth target: its family has depth.
        let r32 = Support {
            own: TEXTURE_2D,
            family: TEXTURE_2D | DEPTH_STENCIL,
            typeless: false,
        };
        assert_eq!(refusal(r32, 3, 2), None);
        // Unordered access: the format's own, its family's for a typeless
        // one.
        assert_eq!(refusal(r32, 3, 4), Some("no unordered access"));
        let typeless = Support {
            typeless: true,
            family: TEXTURE_2D | TYPED_UNORDERED_ACCESS,
            ..r32
        };
        assert_eq!(refusal(typeless, 3, 4), None);
    }

    #[test]
    fn packed_texels_widen_to_the_same_values() {
        let row = |c: Convert, from: &[u8]| {
            let (a, b) = c.bytes();
            let mut to = vec![0; from.len() / a * b];
            c.row(from, &mut to);
            to
        };
        // B5G5R5A1: blue 31, green 0, red 16, alpha 1; then all zero.
        let v: u16 = 31 | 16 << 10 | 1 << 15;
        assert_eq!(
            row(Convert::Bgr5a1, &[v.to_le_bytes(), [0, 0]].concat()),
            [132, 0, 255, 255, 0, 0, 0, 0]
        );
        // B5G6R5: red 31, green 32, blue 1; alpha one.
        let v: u16 = 1 | 32 << 5 | 31 << 11;
        assert_eq!(row(Convert::Bgr565, &v.to_le_bytes()), [255, 130, 8, 255]);
        // B4G4R4A4: blue 1, green 2, red 3, alpha 15.
        let v: u16 = 1 | 2 << 4 | 3 << 8 | 15 << 12;
        assert_eq!(row(Convert::Bgra4, &v.to_le_bytes()), [51, 34, 17, 255]);
        // R4G4: red 15, green 8.
        assert_eq!(row(Convert::Rg4, &[0x8f]), [255, 136]);
        // Every code rounds as the GPU's conversion does.
        for bits in [1, 4, 5, 6] {
            let max = (1u32 << bits) - 1;
            for v in 0..=max {
                let exact = (f64::from(v) / f64::from(max) * 255.0).round() as u8;
                assert_eq!(unorm8(v, bits), exact, "{v} of {bits} bits");
            }
        }
    }

    #[test]
    fn signed_ten_bit_texels_widen_to_the_same_values() {
        let texel = |r: i32, g: i32, b: i32, a: u32| {
            let field = |v: i32| (v as u32) & 0x3ff;
            (field(r) | field(g) << 10 | field(b) << 20 | a << 30).to_le_bytes()
        };
        let mut to = [0u8; 8];
        Convert::Snorm10a2.row(&texel(511, -511, -512, 3), &mut to);
        let words: Vec<i16> = to
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| i16::from_le_bytes(*w))
            .collect();
        assert_eq!(words, [32767, -32767, -32767, 32767]);
        Convert::Snorm10a2.row(&texel(0, 1, -256, 1), &mut to);
        let words: Vec<i16> = to
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| i16::from_le_bytes(*w))
            .collect();
        let near = |v: f64| (v * 32767.0).round() as i16;
        assert_eq!(
            words,
            [0, near(1.0 / 511.0), near(-256.0 / 511.0), near(1.0 / 3.0)]
        );
    }

    /// R9G9B9E5_SHAREDEXP (VIctory #492): where the host neither renders to
    /// it nor writes it (GPUs; WARP does both), RGBA16F stands in for every
    /// texture, view, target and clear value of it.
    #[test]
    fn shared_exponent_textures_stand_in_as_half_floats() {
        use uses::*;
        let pc = |f| match f {
            67 => TEXTURE_2D | SHADER_SAMPLE,
            _ => CONSOLE_USES,
        };
        let wanted = Choice::StandIn(R16G16B16A16_FLOAT, Some(Convert::Rgb9e5));
        assert_eq!(choose(entry(67).unwrap(), pc), wanted);
        assert_eq!(choose(entry(67).unwrap(), nvidia), wanted);
        assert_eq!(choose(entry(67).unwrap(), |_| CONSOLE_USES), Choice::Own);
    }

    /// The host device renders to and writes 67 itself (WARP), or has a
    /// stand-in for it that it renders to and writes (the RTX 4080).
    #[cfg(windows)]
    #[test]
    fn the_host_renders_shared_exponent_targets_in_half_floats() {
        let test =
            "resource::format::tests::the_host_renders_shared_exponent_targets_in_half_floats";
        let Some(_gpu) = crate::testgpu::device(test) else {
            return;
        };
        let own = HostCaps.support1(67);
        if own & CONSOLE_USES == CONSOLE_USES {
            assert_eq!(HostCaps.host_format(67), 67, "{own:#x}");
            eprintln!("{test}: the host has every use of R9G9B9E5 ({own:#x}): kept");
            return;
        }
        assert_eq!(HostCaps.host_format(67), R16G16B16A16_FLOAT, "{own:#x}");
        let support = HostCaps.support1(R16G16B16A16_FLOAT);
        assert_eq!(support & CONSOLE_USES, CONSOLE_USES, "{support:#x}");
    }

    /// Every shared-exponent channel is the same 16-bit float, exactly.
    #[test]
    fn shared_exponent_texels_widen_to_the_same_values() {
        let value = |h: u16| {
            let (e, f) = (i32::from(h >> 10 & 0x1f), f64::from(h & 0x3ff));
            assert!(e < 31, "{h:#x} is not finite");
            if e == 0 {
                f * 2f64.powi(-24)
            } else {
                (1.0 + f / 1024.0) * 2f64.powi(e - 15)
            }
        };
        for exponent in 0..32 {
            for mantissa in 0..512 {
                let exact = f64::from(mantissa) * 2f64.powi(exponent as i32 - 24);
                let h = half_of_9e5(mantissa, exponent);
                assert_eq!(value(h), exact, "{mantissa} * 2^({exponent} - 24)");
            }
        }
        assert_eq!(half_of_9e5(511, 31), 0x7bfc, "65,408, the largest");
        assert_eq!(half_of_9e5(1, 0), 1, "2^-24, the smallest");
        // A texel: red 1 (256 at exponent 16), green 0.5, blue 0.
        let texel: u32 = 256 | 128 << 9 | 16 << 27;
        let mut to = [0u8; 8];
        Convert::Rgb9e5.row(&texel.to_le_bytes(), &mut to);
        let halves: Vec<u16> = to
            .as_chunks::<2>()
            .0
            .iter()
            .map(|w| u16::from_le_bytes(*w))
            .collect();
        assert_eq!(halves, [0x3c00, 0x3800, 0, HALF_ONE]);
    }

    /// Components reading the source alpha read one; the others stay.
    #[test]
    fn views_of_formats_without_alpha_read_one() {
        const DEFAULT: u32 = 0x1688; // R, G, B, A
        assert_eq!(mapping_alpha_one(DEFAULT), 0x1688 & !(7 << 9) | 5 << 9);
        // A, A, A, R: every alpha becomes one, red stays.
        let swizzled = 3 | 3 << 3 | 3 << 6 | 1 << 12;
        assert_eq!(mapping_alpha_one(swizzled), 5 | 5 << 3 | 5 << 6 | 1 << 12);
        // Forced values stay.
        let forced = 4 | 5 << 3 | 2 << 6 | 1 << 9 | 1 << 12;
        assert_eq!(mapping_alpha_one(forced), forced);
        // Formats with an alpha of their own keep it.
        assert!(!alpha_one(28));
        assert!(!alpha_one(86));
    }

    #[test]
    fn the_report_counts_by_reason() {
        note_stand_in(86, 28);
        note_refused("format 191 flags 0x1: console format 191 has no host format".into());
        note_view_cast(28, 29, ViewUse::Shader, true);
        note_view_cast(28, 29, ViewUse::Shader, true);
        note_view_cast(41, 42, ViewUse::Unordered, false);
        note_view_cast(41, 42, ViewUse::Target, false);
        note_buffer_without_format(190, false);
        let report = report().unwrap();
        assert!(report.contains("of 86 as 28"), "{report}");
        assert!(report.contains("console format 191"), "{report}");
        assert!(report.contains("srv of 28 as 29 cast"), "{report}");
        assert!(
            report.contains("uav of 41 as 42 as the texture's"),
            "{report}"
        );
        assert!(report.contains("rtv of 41 as 42 not castable"), "{report}");
        assert!(report.contains("srv of 190"), "{report}");
    }

    /// D3D12's relaxed casting rules: any two formats of a family, but
    /// float with non-float and UNORM with SNORM; depth, BC6H and typeless
    /// formats are left out.
    #[test]
    fn views_cast_within_a_family_as_d3d12_allows() {
        for (own, view) in [
            (28, 29), // RGBA8 UNORM, sRGB
            (29, 28),
            (28, 30), // UNORM as UINT
            (30, 32), // UINT as SINT
            (31, 32), // SNORM as SINT
            (87, 91), // BGRA8 UNORM, sRGB
            (88, 93), // BGRX8 UNORM, sRGB
            (71, 72), // BC1 UNORM, sRGB
            (98, 99), // BC7 UNORM, sRGB
            (24, 25), // RGB10A2 UNORM as UINT
            (11, 12), // RGBA16 UNORM as UINT
            (42, 43), // R32 UINT as SINT
            (56, 57), // R16 UNORM as UINT
            (61, 62), // R8 UNORM as UINT
        ] {
            assert!(castable(own, view), "{own} as {view}");
        }
        for (own, view) in [
            (28, 31), // UNORM as SNORM
            (80, 81), // BC4 UNORM as SNORM
            (41, 42), // R32 FLOAT as UINT
            (42, 41),
            (10, 11), // RGBA16 FLOAT as UNORM
            (54, 56), // R16 FLOAT as UNORM
            (87, 88), // two families
            (28, 87),
            (61, 65), // A8_UNORM appartient à sa propre famille DXGI.
            (65, 61),
            (41, 40), // R32 FLOAT as depth
            (45, 46), // depth with stencil
            (95, 96), // BC6H
            (27, 28), // typeless
        ] {
            assert!(!castable(own, view), "{own} as {view}");
        }
    }

    /// A host that casts, and has the uses asked for: RGBA8 sRGB views of
    /// UNORM keep their format, R32 UINT views of FLOAT do not; a host that
    /// does not cast keeps none.
    #[test]
    fn views_keep_their_format_where_the_host_casts() {
        struct Host(bool);
        impl Caps for Host {
            fn support1(&self, format: u32) -> u32 {
                match format {
                    // No typed unordered access in sRGB.
                    29 => uses::SHADER_LOAD,
                    _ => uses::SHADER_LOAD | uses::TYPED_UNORDERED_ACCESS,
                }
            }
            fn samples(&self, _: u32, count: u32) -> bool {
                count <= 1
            }
            fn fully_typed_casting(&self) -> bool {
                self.0
            }
        }
        let casts = Host(true);
        assert!(view_cast_with(&casts, 28, 29, false));
        assert!(view_cast_with(&casts, 29, 28, false));
        assert!(view_cast_with(&casts, 28, 30, true));
        assert!(!view_cast_with(&casts, 28, 29, true), "no sRGB writes");
        assert!(!view_cast_with(&casts, 41, 42, false));
        assert!(view_cast_with(&casts, 41, 41, true), "its own format");
        let old = Host(false);
        assert!(!view_cast_with(&old, 28, 29, false));
        assert!(view_cast_with(&old, 28, 28, false));
    }

    /// Typed buffer views keep the title's element size: the title's own
    /// format where the host has it for buffers, else a typeless format of
    /// the same size (no view); other formats are as before.
    #[test]
    fn typed_buffer_views_keep_the_titles_element_size() {
        struct Host;
        impl Caps for Host {
            fn support1(&self, format: u32) -> u32 {
                use uses::*;
                match format {
                    85 => BUFFER | TEXTURE_2D | SHADER_SAMPLE,
                    86 => BUFFER | TYPED_UNORDERED_ACCESS | TEXTURE_2D,
                    115 => TEXTURE_2D | SHADER_SAMPLE,
                    _ => CONSOLE_USES | BUFFER,
                }
            }
            fn samples(&self, _: u32, count: u32) -> bool {
                count <= 1
            }
        }
        assert_eq!(buffer_format_with(&Host, 85, false), 85);
        assert_eq!(buffer_format_with(&Host, 85, true), 53, "no typed writes");
        assert_eq!(buffer_format_with(&Host, 86, true), 86);
        assert_eq!(buffer_format_with(&Host, 115, false), 53);
        // The console's own numbers: none on the host, whatever it says.
        assert_eq!(buffer_format_with(&Host, 189, false), 39);
        assert_eq!(buffer_format_with(&Host, 190, false), 60);
        // Formats outside the table, as before.
        assert_eq!(buffer_format_with(&Host, 28, false), 28);
        for title in 118..=120 {
            assert_eq!(buffer_format_with(&Host, title, false), 0);
            assert_eq!(buffer_format_with(&Host, title, true), 0);
        }
        for format in [53, 39, 60] {
            assert_eq!(super::super::family(format), format, "{format} is typeless");
        }
    }
}
