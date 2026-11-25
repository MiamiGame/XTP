//! The ray tracing BTP 6 runs and XTP cannot run yet: which dropped
//! dispatch is which trace, what it would have written, and a stand-in
//! that makes its rays miss.
//!
//! The title's trace programs walk its ray-tracing structure in plain title
//! memory, through addresses the translator does not follow ("an address
//! not from a root parameter", docs/analysis/shadps4.md §3.2), so their
//! dispatches are dropped (no host pipeline) and what they write keeps
//! whatever was there. [`PROGRAMS`] names the thirteen by hash and role;
//! `State::rt_dropped` (`draw.rs`) hands each dropped dispatch of one of
//! them here with the work it would have run and, when the stand-in asks,
//! the views its program's own resource list names for unordered access,
//! as the title's tables (those in the bound heaps) hold them at the
//! dispatch.
//!
//! **Counts** (every build): every [`REPORT_EVERY`] frames, and in
//! diagnostics builds for each frame `XTP_CAPTURE_FRAMES` or
//! `XTP_CAPTURE_WORLD` lists, one line gives the frame's drops by role,
//! whether the stand-in ran and the A/B arm the frame ran in. With the
//! stand-in off that is all a drop costs: an addition under a lock, which
//! the frame's first drop also takes to end the frame before.
//!
//! **Stand-in** (`XTP_RT_STANDIN`, every build): `1` clears what a dropped
//! trace would have written to its "every ray missed" value ([`Fill`]),
//! `0` or unset leaves it, `ab…` alternates the two by A/B phase within one
//! run (`xtp_core::ab`). Which output gets what is [`default_fill`],
//! replaced output by output with `XTP_RT_STANDIN_FILL` ([`parse_fills`]).
//! A clear is recorded where the dispatch would have run, through XTP's
//! own view heap, as the title's UAV clears are (`list::clear_uav`), and
//! only if the dispatch would have run a group ([`Work`]): a direct one
//! with no zero count, an indirect one under a GPU predicate on the
//! count of its commands that run a group, as the host's rewrite of its
//! arguments leaves it; under that predicate a texture whose writes the
//! list tracks (a depth texture's colour twin, a host copy) is not
//! cleared, since the list would mark it written before the GPU knows the
//! count ([`tracked_write`]). A texture the title placed in memory itself is
//! cleared through the resource made for that memory, which its shader
//! views read. With the stand-in on, the first time a program is dropped
//! with an output, the output is logged ("rt: dropped trace"): a texture's
//! size and format, a buffer's range; in diagnostics builds each drop is
//! also an op of the pass dump (`XTP_PASS_DUMP`), so the dump shows where
//! in the frame each trace would have run and what it would have written.
//!
//! shortcut: a stand-in, not ray tracing, until the trace programs
//! translate. A clear writes the whole view, where the trace may write part
//! of it (a checkerboard, tiles, the entries a schedule picked); and what a
//! miss means in a buffer is the program's layout, which nothing here
//! reads, so buffers are left alone unless `XTP_RT_STANDIN_FILL` names
//! them.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use xtp_core::ab;
use xtp_shader::{ResourceType, UsedResource};

use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;

/// What a refused trace program computes, as far as its outputs go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Role {
    /// The sun's shadows traced past the shadow cascades (four variants:
    /// plain, with a leak plug, each with or without a 4× temporal
    /// upsample).
    SunShadow,
    /// Local lights' shadows: the tiled lights' traced shadows and the
    /// checkerboard trace into the shadow mask.
    LocalShadow,
    /// The irradiance cache's shadow rays towards the lights.
    CacheShadow,
    /// Probe gathers through the structure: the primary probes, the
    /// transparent surfaces' probes and the irradiance cache.
    ProbeGather,
    /// The screen probes' rays (indirect diffuse light).
    ScreenProbes,
    /// The screen rays' visibility buffer (reflections, before hit shading).
    Reflections,
    /// GPU particles colliding with the structure.
    Particles,
}

impl Role {
    /// Every role, in report order.
    const ALL: [Role; 7] = [
        Role::SunShadow,
        Role::LocalShadow,
        Role::CacheShadow,
        Role::ProbeGather,
        Role::ScreenProbes,
        Role::Reflections,
        Role::Particles,
    ];

    /// Its name in logs and in `XTP_RT_STANDIN_FILL`.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Role::SunShadow => "sun-shadow",
            Role::LocalShadow => "local-shadow",
            Role::CacheShadow => "cache-shadow",
            Role::ProbeGather => "probe-gather",
            Role::ScreenProbes => "screen-probes",
            Role::Reflections => "reflections",
            Role::Particles => "particles",
        }
    }

    fn index(self) -> usize {
        Role::ALL.iter().position(|&r| r == self).unwrap_or(0)
    }

    fn named(name: &str) -> Option<Role> {
        Role::ALL.into_iter().find(|r| r.name() == name)
    }
}

/// BTP 6's trace programs that do not translate, by the 16-digit hash the
/// logs give (the first 8 bytes of the container's BLAKE3), with their
/// roles (docs/analysis/refusals.md §6.3, docs/analysis/rt-standin.md).
pub(crate) const PROGRAMS: [(&str, Role); 13] = [
    ("a1b20bc47e3c9597", Role::SunShadow),
    ("e8c962358d862e3f", Role::SunShadow),
    ("17829ac71aab4e8d", Role::SunShadow),
    ("afeea48bbb9913ad", Role::SunShadow),
    ("32ed072b9f845048", Role::LocalShadow),
    ("559d9f0184e15fe3", Role::LocalShadow),
    ("f607780aa8e502d0", Role::CacheShadow),
    ("3592426e3e9142d8", Role::ProbeGather),
    ("36f04a4ed0ad1d9b", Role::ProbeGather),
    ("b58eb3ca6ee46427", Role::ProbeGather),
    ("f22a76fcb967ee59", Role::ScreenProbes),
    ("6d7fe7745ab9c6d8", Role::Reflections),
    ("caf96a3229145fa6", Role::Particles),
];

/// The entry of [`PROGRAMS`] of the program with log hash `hash`, if one
/// is.
fn known(hash: &str) -> Option<(&'static str, Role)> {
    PROGRAMS.iter().find(|(h, _)| *h == hash).copied()
}

/// A dropped dispatch's program, when it is one of [`PROGRAMS`]: its hash,
/// role and the resources its console code names.
pub(crate) struct Program<'a> {
    pub(crate) hash: &'static str,
    pub(crate) role: Role,
    used: &'a [UsedResource],
}

impl Program<'_> {
    /// What the program writes through unordered access, in its resource
    /// list's order: output `n` of the logs and of `XTP_RT_STANDIN_FILL`.
    pub(crate) fn writes(&self) -> impl Iterator<Item = &UsedResource> {
        self.used.iter().filter(|u| u.kind == ResourceType::Uav)
    }
}

/// The trace program of title pipeline `pipeline`, if it runs one of
/// [`PROGRAMS`]. Cheap for others: a hash compared with thirteen.
pub(crate) fn program<'a>(pipeline: This) -> Option<Program<'a>> {
    let shader = crate::pipeline::compute_shader(pipeline)?;
    let (hash, role) = known(&shader.hash)?;
    let used = shader.native().map(|n| &n.used[..]).unwrap_or(&[]);
    Some(Program { hash, role, used })
}

/// What an output of a dropped trace is bound to at the dispatch.
#[derive(Debug, Clone)]
pub(crate) enum Bound {
    /// A table entry's view.
    View(View),
    /// A root view's address (a buffer of no stated size).
    Root(u64),
    /// Nothing the title set.
    Unset,
}

/// What "every ray missed" writes into an output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Fill {
    /// Left as it is.
    Keep,
    /// A float clear: the values in the view's format (textures and typed
    /// buffer views).
    Float([f32; 4]),
    /// An integer clear: each value's low bits into each channel (any
    /// unordered view, raw and structured buffers included).
    Uint([u32; 4]),
}

/// The fill a role's output takes without `XTP_RT_STANDIN_FILL`: shadow
/// traces' textures fully lit (visibility 1), everything else left (a
/// gather's radiance, a visibility buffer's hit records and particles'
/// state have no value that means "missed" without their layout).
pub(crate) fn default_fill(role: Role, texture: bool) -> Fill {
    match role {
        Role::SunShadow | Role::LocalShadow | Role::CacheShadow if texture => Fill::Float([1.0; 4]),
        _ => Fill::Keep,
    }
}

/// What `XTP_RT_STANDIN_FILL` names: a program (hash) or a role, and one
/// of its outputs or all.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Rule {
    who: Who,
    output: Option<u32>,
    fill: Fill,
}

#[derive(Debug, Clone, PartialEq)]
enum Who {
    Program(String),
    Role(Role),
}

/// `XTP_RT_STANDIN_FILL`: rules separated by `;`, each
/// `<program hash or role>[.<output>]=<fill>`, the fill `lit` (float 1),
/// `zero` (float 0), `keep`, `f:<v>[,<v>,<v>,<v>]` (floats; one value is
/// taken for all four) or `u:<hex>[,<hex>,<hex>,<hex>]` (integers, the
/// same). The last rule that matches an output wins; an output no rule
/// names takes [`default_fill`]. A bad rule is reported and skipped.
pub(crate) fn parse_fills(text: &str) -> (Vec<Rule>, Vec<String>) {
    let mut rules = Vec::new();
    let mut bad = Vec::new();
    for part in text.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        match parse_rule(part) {
            Some(rule) => rules.push(rule),
            None => bad.push(part.to_owned()),
        }
    }
    (rules, bad)
}

fn parse_rule(text: &str) -> Option<Rule> {
    let (key, fill) = text.split_once('=')?;
    let (who, output) = match key.trim().split_once('.') {
        Some((who, n)) => (who.trim(), Some(n.trim().parse().ok()?)),
        None => (key.trim(), None),
    };
    let who = match Role::named(who) {
        Some(role) => Who::Role(role),
        None if who.len() == 16 && who.bytes().all(|b| b.is_ascii_hexdigit()) => {
            Who::Program(who.to_ascii_lowercase())
        }
        None => return None,
    };
    Some(Rule {
        who,
        output,
        fill: parse_fill(fill.trim())?,
    })
}

fn parse_fill(text: &str) -> Option<Fill> {
    fn four<T: Copy>(values: Vec<T>) -> Option<[T; 4]> {
        match values[..] {
            [v] => Some([v; 4]),
            [a, b, c, d] => Some([a, b, c, d]),
            _ => None,
        }
    }
    match text {
        "lit" => return Some(Fill::Float([1.0; 4])),
        "zero" => return Some(Fill::Float([0.0; 4])),
        "keep" => return Some(Fill::Keep),
        _ => {}
    }
    if let Some(values) = text.strip_prefix("f:") {
        let values: Option<Vec<f32>> = values
            .split(',')
            .map(|v| v.trim().parse().ok().filter(|v: &f32| v.is_finite()))
            .collect();
        return four(values?).map(Fill::Float);
    }
    if let Some(values) = text.strip_prefix("u:") {
        let values: Option<Vec<u32>> = values
            .split(',')
            .map(|v| {
                let v = v.trim();
                u32::from_str_radix(v.strip_prefix("0x").unwrap_or(v), 16).ok()
            })
            .collect();
        return four(values?).map(Fill::Uint);
    }
    None
}

/// The fill output `output` of `hash` (role `role`, a texture or not)
/// takes under `rules`.
pub(crate) fn fill_for(rules: &[Rule], hash: &str, role: Role, output: u32, texture: bool) -> Fill {
    rules
        .iter()
        .rev()
        .find(|r| {
            let who = match &r.who {
                Who::Program(h) => h == hash,
                Who::Role(r) => *r == role,
            };
            who && r.output.is_none_or(|n| n == output)
        })
        .map_or_else(|| default_fill(role, texture), |r| r.fill)
}

/// `XTP_RT_STANDIN`, read once: off unless asked (the A/B of
/// docs/analysis/rt-standin.md decides whether it goes on by default).
static STANDIN: LazyLock<ab::Switch<bool>> = LazyLock::new(|| {
    let switch = ab::switch("XTP_RT_STANDIN", false);
    let mode = match (switch.alternates(), switch.get()) {
        (true, _) => "off in A/B phase A, on in phase B",
        (false, true) => "on",
        (false, false) => "off",
    };
    tracing::info!("ray-tracing stand-in (dropped traces miss): {mode}");
    switch
});

/// `XTP_RT_STANDIN_FILL`, read once.
static RULES: LazyLock<Vec<Rule>> = LazyLock::new(|| {
    let text = std::env::var("XTP_RT_STANDIN_FILL").unwrap_or_default();
    let (rules, bad) = parse_fills(&text);
    for rule in bad {
        tracing::warn!(
            rule,
            "XTP_RT_STANDIN_FILL: not <program hash or role>[.<output>]=lit|zero|keep|f:<v>[,…]|u:<hex>[,…]; skipped"
        );
    }
    if !rules.is_empty() {
        tracing::info!(?rules, "ray-tracing stand-in fills");
    }
    rules
});

/// Frames between two drop reports.
pub(crate) const REPORT_EVERY: u64 = 256;

/// Outputs logged at most (first drops).
const LOGGED_AT_MOST: usize = 256;

/// The frame being counted: its number (`u64::MAX` before the first drop),
/// the A/B arm it ran in (taken at its first drop), its drops by role and
/// its clears.
#[derive(Debug)]
struct Tally {
    frame: u64,
    arm: ab::Arm,
    drops: [u64; 7],
    cleared: u64,
}

impl Tally {
    const fn new(frame: u64, arm: ab::Arm) -> Self {
        Tally {
            frame,
            arm,
            drops: [0; 7],
            cleared: 0,
        }
    }
}

/// The frame being counted, under one lock: lists are recorded on many
/// threads, and the first drop of a frame must end the frame before and
/// start its own in one step, or a drop of the new frame on another thread
/// could be counted in the frame that ends. Then the clears the host
/// refused, over the run.
struct Counts {
    tally: Mutex<Tally>,
    failed: AtomicU64,
}

impl Counts {
    const fn new() -> Self {
        Counts {
            tally: Mutex::new(Tally::new(u64::MAX, ab::Arm::A)),
            failed: AtomicU64::new(0),
        }
    }

    fn tally(&self) -> std::sync::MutexGuard<'_, Tally> {
        self.tally.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A clear recorded, in the frame being counted.
    fn cleared(&self) {
        self.tally().cleared += 1;
    }
}

static COUNTS: Counts = Counts::new();

/// An output logged: (program, output, what it is bound to).
type Logged = (&'static str, u32, String);

/// The outputs logged so far.
static LOGGED: Mutex<Option<HashSet<Logged>>> = Mutex::new(None);

/// How much work a dropped dispatch would have run.
pub(crate) enum Work {
    /// A direct dispatch's thread group counts.
    Groups([u32; 3]),
    /// An indirect one, as the host's rewrite of its arguments left it
    /// (`indirect::rewrite`): the `u32` count of its commands that run a
    /// group, at `offset` of the buffer, in the unordered access state.
    /// Known on the GPU only: the clears run under a predicate on it.
    /// `None` when there is no such count.
    Commands(Option<(ID3D12Resource, u64)>),
}

/// Whether a dispatch that would have run `work` may have written
/// anything: a direct one with no zero group count; an indirect one with
/// a command count (whether it is 0 is the GPU's to say).
fn may_write(work: &Work) -> bool {
    match work {
        Work::Groups(groups) => !groups.contains(&0),
        Work::Commands(count) => count.is_some(),
    }
}

/// Whether the stand-in runs for the frame being prepared. With it off,
/// a dropped trace is only counted ([`dropped`]): its outputs are neither
/// resolved, described, logged nor cleared.
pub(crate) fn active() -> bool {
    STANDIN.get()
}

/// A dispatch of `program` dropped in `list`, which would have run `work`:
/// counted; with the stand-in on ([`active`]), and if it would have run a
/// group, its outputs (`bound` gives what they are bound to, in
/// [`Program::writes`]' order) logged the first time and cleared to their
/// fills. True if the list's descriptor heaps changed (the caller's state
/// must set the title's again).
pub(crate) fn dropped(
    list: &HostList,
    program: &Program,
    work: Work,
    bound: impl FnOnce() -> Vec<Bound>,
) -> bool {
    let frame = crate::queue::presented() + 1;
    count(frame, program.role, ab::current());
    if !active() {
        return false;
    }
    if !may_write(&work) {
        return false;
    }
    let commands = match work {
        Work::Groups(_) | Work::Commands(None) => None,
        Work::Commands(Some(count)) => Some(count),
    };
    // The predicate word holding the command count, made at the first
    // clear.
    let mut predicate = None;
    let mut heaps = false;
    for (n, b) in bound().into_iter().enumerate() {
        let n = n as u32;
        // A placed texture is cleared through its memory's resource; one
        // without a host copy yet stays as it is, and its clear fails.
        let b = match b {
            Bound::View(view) => Bound::View(host_view(view.clone()).unwrap_or(view)),
            other => other,
        };
        let what = describe(&b);
        note_first(program, n, &what);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            format!(
                "rt: dropped {} {} output {n}: {what}",
                program.role.name(),
                program.hash
            )
        });
        let Bound::View(view) = &b else {
            continue;
        };
        let texture = texture_of(view).is_some() || placed_texture(view);
        let fill = fill_for(&RULES, program.hash, program.role, n, texture);
        if fill == Fill::Keep {
            continue;
        }
        let cleared = match &commands {
            None => clear(list, view, fill, None),
            Some((buffer, offset)) => {
                if predicate.is_none() {
                    predicate = Some(predicate_word(list, buffer, *offset));
                }
                match &predicate {
                    Some(Ok(word)) => clear(list, view, fill, Some(*word)),
                    Some(Err(why)) => Err(*why),
                    None => Err("no predicate"),
                }
            }
        };
        match cleared {
            Ok(()) => {
                heaps = true;
                COUNTS.cleared();
            }
            Err(why) => {
                let n_failed = COUNTS.failed.fetch_add(1, Ordering::Relaxed) + 1;
                if n_failed.is_power_of_two() {
                    tracing::warn!(
                        program = program.hash,
                        role = program.role.name(),
                        output = n,
                        what,
                        why,
                        failed = n_failed,
                        "rt stand-in: an output not cleared"
                    );
                }
            }
        }
    }
    if let Some(Ok((words, _))) = predicate {
        end_predicate(list, words);
    }
    heaps
}

/// A frame's counts, once the next frame's first drop ended it.
#[derive(Debug, PartialEq)]
struct Ended {
    frame: u64,
    /// The arm the frame ran in.
    arm: ab::Arm,
    drops: Vec<u64>,
    cleared: u64,
}

impl Counts {
    /// Counts a drop of `role` in `frame`, prepared in A/B arm `arm`. The
    /// first drop of a frame ends the frame before: its counts, with the
    /// arm taken at its own first drop (the arm changes at the present
    /// that ends a phase, before the next frame's first drop).
    fn count(&self, frame: u64, role: Role, arm: ab::Arm) -> Option<Ended> {
        let mut tally = self.tally();
        let mut ended = None;
        if tally.frame != frame {
            let last = std::mem::replace(&mut *tally, Tally::new(frame, arm));
            if last.frame != u64::MAX {
                ended = Some(Ended {
                    frame: last.frame,
                    arm: last.arm,
                    drops: last.drops.to_vec(),
                    cleared: last.cleared,
                });
            }
        }
        tally.drops[role.index()] += 1;
        ended
    }
}

/// Counts a drop of `role` in `frame`, prepared in A/B arm `arm`, and
/// reports the frame it ends when that one is due ([`due`]).
fn count(frame: u64, role: Role, arm: ab::Arm) {
    let Some(ended) = COUNTS.count(frame, role, arm) else {
        return;
    };
    if due(ended.frame) {
        tracing::info!(
            frame = ended.frame,
            drops = %report(&ended.drops),
            cleared = ended.cleared,
            standin = STANDIN.at(ended.arm),
            arm = %ended.arm,
            "rt: dropped traces in a frame"
        );
    }
}

/// Whether frame `frame`'s drops are reported: every [`REPORT_EVERY`]
/// frames, and (diagnostics) the frames captures list.
fn due(frame: u64) -> bool {
    #[cfg(feature = "diagnostics")]
    if crate::capture::listed(frame) {
        return true;
    }
    frame.is_multiple_of(REPORT_EVERY)
}

/// A frame's drops by role, as reported: `sun-shadow 4, probe-gather 3`.
fn report(drops: &[u64]) -> String {
    let parts: Vec<String> = Role::ALL
        .iter()
        .zip(drops)
        .filter(|(_, n)| **n > 0)
        .map(|(r, n)| format!("{} {n}", r.name()))
        .collect();
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(", ")
    }
}

/// Logs output `n` of `program` the first time it is seen bound so.
fn note_first(program: &Program, n: u32, what: &str) {
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    let logged = logged.get_or_insert_with(HashSet::new);
    if logged.len() >= LOGGED_AT_MOST {
        return;
    }
    if logged.insert((program.hash, n, what.to_owned())) {
        tracing::info!(
            program = program.hash,
            role = program.role.name(),
            output = n,
            what,
            "rt: dropped trace, output"
        );
    }
}

/// The view a clear of `view` writes through: a texture the title placed
/// in memory itself is viewed through the resource made for that memory
/// (`resource::texture_at`, which shader views of it read too, as
/// `descriptor::host::placed_texture` makes them); other views as they
/// are. `None` for a placed texture without a host copy yet.
fn host_view(view: View) -> Option<View> {
    const BUFFER: u32 = 1;
    match view {
        View::Placed(true, resource, desc) if resource[0] != BUFFER => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let texture = crate::resource::texture_at(address, &resource)?;
            Some(View::Unordered(
                texture as usize,
                0,
                desc[..10].try_into().ok(),
            ))
        }
        other => Some(other),
    }
}

/// Whether `view` is unordered access to a texture the title placed in
/// memory itself (one [`host_view`] found no host copy for).
fn placed_texture(view: &View) -> bool {
    const BUFFER: u32 = 1;
    matches!(view, View::Placed(true, resource, _) if resource[0] != BUFFER)
}

/// A texture view's host texture: (title resource, host resource).
fn texture_of(view: &View) -> Option<(This, &'static ID3D12Resource)> {
    let View::Unordered(resource, _, _) = view else {
        return None;
    };
    let resource = *resource as This;
    crate::resource::buffer_bytes(resource)
        .is_none()
        .then_some(())?;
    Some((resource, crate::resource::host_resource(resource)?))
}

/// What an output is bound to, in a few words: a texture's resource, size
/// and host format, or a buffer's title range (`buffer Some((<address>,
/// <bytes>))`, which the pass dump samples when small).
fn describe(bound: &Bound) -> String {
    match bound {
        Bound::Unset => "unset".into(),
        Bound::Root(address) => format!("root view at {address:#x}"),
        Bound::View(view) => {
            if let Some((resource, texture)) = texture_of(view) {
                // SAFETY: a live resource.
                let d = unsafe { texture.GetDesc() };
                return format!(
                    "texture {resource:?} {}x{}x{} f{}",
                    d.Width, d.Height, d.DepthOrArraySize, d.Format.0
                );
            }
            match crate::descriptor::host::buffer_range(view) {
                Some(range) => format!("buffer {:?}", Some(range)),
                None => match view {
                    View::Placed(true, r, _) => {
                        format!("placed texture dim {} {}x{}", r[0], r[4], r[6])
                    }
                    View::Unordered(r, ..) => format!("unordered {r:#x} without a host copy"),
                    other => format!("{other:?}").chars().take(60).collect(),
                },
            }
        }
    }
}

/// CPU-only view descriptors for the clears' second handle, taken in turn
/// (as `list::ClearSlots`).
struct Slots {
    /// Held: the descriptors live in it.
    _heap: ID3D12DescriptorHeap,
    start: usize,
    step: usize,
    next: usize,
}

// SAFETY: D3D12 descriptor heaps are free-threaded; the rest is plain data
// behind the mutex.
unsafe impl Send for Slots {}

const SLOTS: usize = 64;

impl Slots {
    fn take(&mut self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let at = self.start + self.next * self.step;
        self.next = (self.next + 1) % SLOTS;
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at }
    }
}

/// The clears' CPU-only descriptors, made on first use; `None` without a
/// host GPU.
fn slots() -> Option<&'static Mutex<Slots>> {
    static MADE: std::sync::OnceLock<Option<Mutex<Slots>>> = std::sync::OnceLock::new();
    MADE.get_or_init(|| {
        let host = crate::host::host()?;
        // SAFETY: plain heap creation and queries of a live heap.
        unsafe {
            let heap = host
                .device
                .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: SLOTS as u32,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .inspect_err(|e| tracing::warn!(error = %e, "rt stand-in: no CPU-only heap"))
                .ok()?;
            let start = heap.GetCPUDescriptorHandleForHeapStart().ptr;
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            Some(Mutex::new(Slots {
                _heap: heap,
                start,
                step,
                next: 0,
            }))
        }
    })
    .as_ref()
}

/// Predicate words (8 bytes each, the upper four always 0) that hold an
/// indirect dispatch's command count while its clears are recorded:
/// handed out in turn, a word is written again only [`WORDS`] dropped
/// indirect traces later, long after the list that read it ran.
struct Words {
    buffer: ID3D12Resource,
    next: AtomicU64,
}

const WORDS: u64 = 4096;

/// The predicate words, made on first use (zeroed, in the common state,
/// to which each use returns them); `None` without a host GPU.
fn words() -> Option<&'static Words> {
    static MADE: std::sync::OnceLock<Option<Words>> = std::sync::OnceLock::new();
    MADE.get_or_init(|| {
        let host = crate::host::host()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: WORDS * 8,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a buffer; a default heap's committed
        // memory starts zeroed.
        unsafe {
            host.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        }
        .inspect_err(|e| tracing::warn!(error = %e, "rt stand-in: no predicate words"))
        .ok()?;
        let buffer = buffer?;
        crate::host::name(&buffer, || "rt stand-in predicates".into());
        Some(Words {
            buffer,
            next: AtomicU64::new(0),
        })
    })
    .as_ref()
}

/// Records in `list` the copy of the `u32` command count at `offset` of
/// `count` (in the unordered access state, which it is left in) into a
/// predicate word, left in the predication state: the clears that follow
/// run under it, so they run only if the count is not 0. Ends with
/// [`end_predicate`].
fn predicate_word(
    list: &HostList,
    count: &ID3D12Resource,
    offset: u64,
) -> Result<(&'static ID3D12Resource, u64), &'static str> {
    use crate::host::transition;
    if !offset.is_multiple_of(4) {
        return Err("a command count not 4-byte aligned");
    }
    let words = words().ok_or("no predicate words")?;
    let at = words.next.fetch_add(1, Ordering::Relaxed) % WORDS * 8;
    list.keep(&words.buffer);
    list.keep(count);
    // SAFETY: an open list; both buffers are live and held by the list, in
    // the states the barriers name (the count as the rewrite left it, the
    // words as every use leaves them).
    unsafe {
        list.list.ResourceBarrier(&[
            transition(
                count,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            ),
            transition(
                &words.buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            ),
        ]);
        list.list
            .CopyBufferRegion(&words.buffer, at, count, offset, 4);
        list.list.ResourceBarrier(&[
            transition(
                count,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
            transition(
                &words.buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PREDICATION,
            ),
        ]);
    }
    Ok((&words.buffer, at))
}

/// Returns the predicate words to the common state after the clears that
/// [`predicate_word`] prepared.
fn end_predicate(list: &HostList, words: &ID3D12Resource) {
    // SAFETY: an open list; the words are in the predication state since
    // `predicate_word`.
    unsafe {
        list.list.ResourceBarrier(&[crate::host::transition(
            words,
            D3D12_RESOURCE_STATE_PREDICATION,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Whether a clear of `view` is refused because the list would track its
/// write before the GPU says whether it runs: under a GPU count
/// (`predicated`), an unordered access view of a resource (a texture, or a
/// buffer range without a shadow, `shadowed`). Recording such a clear
/// marks a depth texture's colour twin ahead of the depth texture
/// (`HostList::twins_written`) and the resource's host copy written on the
/// GPU (`texture::gpu_write`), so a dispatch whose count is 0 would leave
/// it marked written by nothing. A shadowed buffer range is refreshed just
/// before its clear, so what goes back to title memory is right whether
/// the clear runs or not; a placed texture tracks no write.
fn tracked_write(view: &View, shadowed: bool, predicated: bool) -> bool {
    predicated && !shadowed && matches!(view, View::Unordered(..))
}

/// Clears `view` to `fill` in `list`, where the dropped dispatch would have
/// run: a texture's host copy, or a buffer range's host copy (which goes
/// back to title memory after the submission, as shader writes do). Under
/// `predicate` (a word and its offset, [`predicate_word`]), the clear
/// itself runs only if the word is not 0, and a texture whose writes the
/// list tracks is refused ([`tracked_write`]). Sets XTP's own view heap on
/// the list. Why not if it cannot.
fn clear(
    list: &HostList,
    view: &View,
    fill: Fill,
    predicate: Option<(&ID3D12Resource, u64)>,
) -> Result<(), &'static str> {
    const UNORDERED_ACCESS: u32 = 0x8;
    let words: Option<&[u32]> = match view {
        View::Unordered(_, _, desc) => desc.as_ref().map(|d| &d[..]),
        View::Placed(true, _, desc) => Some(&desc[..10]),
        _ => return Err("not an unordered access view"),
    };
    let range = crate::descriptor::host::buffer_range(view);
    let shadow = range
        .filter(|&(_, len)| len > 0)
        .and_then(|(address, len)| Some((crate::shadow::find(address, len)?, address, len)));
    if tracked_write(view, shadow.is_some(), predicate.is_some()) {
        return Err("under a GPU count, a view whose write the list would track before the count");
    }
    // A raw or structured buffer view (a stride, or the raw flag) has no
    // format to clear in floats.
    let untyped = words.is_some_and(|w| w[5] != 0 || w[8] & 1 != 0);
    if matches!(fill, Fill::Float(_)) && range.is_some() && untyped {
        return Err("a float fill of a raw or structured buffer");
    }
    let heap = crate::descriptor::own_view_heap().ok_or("no view heap of XTP's own")?;
    let slots = slots().ok_or("no CPU-only heap")?;
    let slot = heap
        .dynamic(1)
        .ok_or("no dynamic slot in XTP's view heap")?;
    list.keep_dynamic(crate::descriptor::OWN_VIEW_HEAP, slot, 1);
    let mut slots = slots.lock().unwrap_or_else(|e| e.into_inner());
    let at = slots.take();
    let target =
        crate::descriptor::host::make_view(view, at).ok_or("the host cannot make the view")?;
    if !heap.make_at(slot, view) {
        return Err("the host cannot place the view in its heap");
    }
    if let Some((shadow, address, len)) = shadow {
        shadow.refresh(list, address, len);
        shadow.gpu_writes(list, address, len);
    } else if let View::Unordered(texture, _, _) = view {
        let texture = *texture as This;
        list.twin_use(texture, true, true);
        list.twins_written();
        crate::list::ensure_state(list, texture, UNORDERED_ACCESS);
        // A view clears one subresource, not every texel of the texture.
        crate::texture::gpu_write(list, texture, false);
    }
    #[cfg(feature = "diagnostics")]
    if let View::Unordered(resource, _, _) = view {
        // The pass dump samples what a "clear UAV … of <resource>" writes.
        list.note_op(|| format!("clear UAV {fill:?} of {:#x} (rt stand-in)", *resource));
    }
    list.keep(&heap.heap);
    list.keep(&target);
    let gpu = heap.gpu_handle(slot);
    // SAFETY: an open list; XTP's shader-visible heap, bound here, holds
    // the view at `gpu`, the CPU-only one at `at`, both of the live
    // `target`; the barriers name no resource.
    unsafe {
        list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        // The clear is an unordered write: after the work before it that
        // wrote the same memory, and before the work after it.
        list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
        // Only the clear is predicated: the copies that bring a buffer's
        // host copy up to date above run whatever the count.
        if let Some((words, at)) = predicate {
            list.list
                .SetPredication(words, at, D3D12_PREDICATION_OP_EQUAL_ZERO);
        }
        match fill {
            Fill::Uint(v) => {
                // Drivers differ on values wider than their channels: the
                // low bits only, in the format the clear writes (the
                // texture's, a typed buffer view's, none for raw ones).
                let own = target.GetDesc().Format.0 as u32;
                let format = match (own, words) {
                    (0, Some(w)) if !untyped => crate::resource::pc_format(w[0]),
                    _ => own,
                };
                let v = crate::resource::format::uint_clear(format, v);
                list.list
                    .ClearUnorderedAccessViewUint(gpu, at, &target, &v, &[]);
            }
            Fill::Float(v) => list
                .list
                .ClearUnorderedAccessViewFloat(gpu, at, &target, &v, &[]),
            Fill::Keep => {}
        }
        if predicate.is_some() {
            list.list
                .SetPredication(None::<&ID3D12Resource>, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
        }
        list.list.ResourceBarrier(&[crate::host::uav_barrier()]);
    }
    drop(slots);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_standin_programs_are_distinct_log_hashes() {
        let mut seen = HashSet::new();
        for (hash, its) in PROGRAMS {
            assert_eq!(hash.len(), 16, "{hash}");
            assert!(
                hash.bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            );
            assert!(seen.insert(hash), "{hash} twice");
            assert_eq!(known(hash), Some((hash, its)), "{hash}");
        }
        assert_eq!(known("0000000000000000"), None);
        // The thirteen consumers of shadps4.md §3.2, by role.
        let n = |r| PROGRAMS.iter().filter(|(_, x)| *x == r).count();
        assert_eq!(n(Role::SunShadow), 4);
        assert_eq!(n(Role::LocalShadow), 2);
        assert_eq!(n(Role::CacheShadow) + n(Role::ProbeGather), 4);
        assert_eq!(n(Role::ScreenProbes) + n(Role::Reflections), 2);
        assert_eq!(n(Role::Particles), 1);
    }

    #[test]
    fn rt_standin_roles_round_trip_by_name() {
        for role in Role::ALL {
            assert_eq!(Role::named(role.name()), Some(role));
            assert_eq!(Role::ALL[role.index()], role);
        }
        assert_eq!(Role::named("sun"), None);
    }

    #[test]
    fn rt_standin_fills_parse() {
        let (rules, bad) = parse_fills(
            " sun-shadow=lit; 559D9F0184E15FE3.1=u:ffffffff ;probe-gather.0=f:0.25;\
             screen-probes=f:0.1,0.2,0.3,1; reflections=keep; particles.2=zero",
        );
        assert!(bad.is_empty(), "{bad:?}");
        assert_eq!(rules.len(), 6);
        assert_eq!(rules[0].who, Who::Role(Role::SunShadow));
        assert_eq!(rules[0].fill, Fill::Float([1.0; 4]));
        assert_eq!(rules[1].who, Who::Program("559d9f0184e15fe3".into()));
        assert_eq!(rules[1].output, Some(1));
        assert_eq!(rules[1].fill, Fill::Uint([0xffff_ffff; 4]));
        assert_eq!(rules[2].fill, Fill::Float([0.25; 4]));
        assert_eq!(rules[3].fill, Fill::Float([0.1, 0.2, 0.3, 1.0]));
        assert_eq!(rules[4].fill, Fill::Keep);
        assert_eq!(rules[5].fill, Fill::Float([0.0; 4]));
        let (rules, bad) = parse_fills(
            "sun=lit;1234=lit;sun-shadow.x=lit;sun-shadow=f:1,2;=;local-shadow=u:zz;cache-shadow=f:nan",
        );
        assert!(rules.is_empty(), "{rules:?}");
        assert_eq!(bad.len(), 7);
        assert_eq!(parse_fills("").0, Vec::new());
    }

    #[test]
    fn rt_standin_fill_defaults_and_overrides() {
        // Without rules: shadow traces' textures lit, the rest kept.
        for role in [Role::SunShadow, Role::LocalShadow, Role::CacheShadow] {
            assert_eq!(fill_for(&[], "x", role, 0, true), Fill::Float([1.0; 4]));
            assert_eq!(fill_for(&[], "x", role, 0, false), Fill::Keep);
        }
        for role in [
            Role::ProbeGather,
            Role::ScreenProbes,
            Role::Reflections,
            Role::Particles,
        ] {
            assert_eq!(fill_for(&[], "x", role, 0, true), Fill::Keep);
        }
        let (rules, _) = parse_fills(
            "local-shadow=u:1; 559d9f0184e15fe3.2=keep; probe-gather=f:0.5; 3592426e3e9142d8=zero",
        );
        let shadow_mask = "559d9f0184e15fe3";
        assert_eq!(
            fill_for(&rules, shadow_mask, Role::LocalShadow, 0, false),
            Fill::Uint([1; 4])
        );
        assert_eq!(
            fill_for(&rules, shadow_mask, Role::LocalShadow, 2, true),
            Fill::Keep
        );
        // The last rule that matches wins.
        assert_eq!(
            fill_for(&rules, "3592426e3e9142d8", Role::ProbeGather, 1, true),
            Fill::Float([0.0; 4])
        );
        assert_eq!(
            fill_for(&rules, "36f04a4ed0ad1d9b", Role::ProbeGather, 1, true),
            Fill::Float([0.5; 4])
        );
        // Rules of other roles leave the defaults.
        assert_eq!(
            fill_for(&rules, "a1b20bc47e3c9597", Role::SunShadow, 0, true),
            Fill::Float([1.0; 4])
        );
    }

    #[test]
    fn rt_standin_ends_each_frame_with_its_own_arm() {
        use ab::Arm;
        let counts = Counts::new();
        // The first drop ever ends no frame.
        assert_eq!(counts.count(10, Role::ProbeGather, Arm::A), None);
        assert_eq!(counts.count(10, Role::ProbeGather, Arm::A), None);
        counts.cleared();
        // The phase changed at the present after frame 10: frame 10 is
        // reported in A, the arm it ran in, not B.
        let ended = counts.count(11, Role::Particles, Arm::B).unwrap();
        assert_eq!(ended.frame, 10);
        assert_eq!(ended.arm, Arm::A);
        assert_eq!(ended.drops[Role::ProbeGather.index()], 2);
        assert_eq!(ended.drops.iter().sum::<u64>(), 2);
        assert_eq!(ended.cleared, 1);
        let ended = counts.count(12, Role::ProbeGather, Arm::B).unwrap();
        assert_eq!((ended.frame, ended.arm), (11, Arm::B));
        assert_eq!(ended.drops[Role::Particles.index()], 1);
        assert_eq!(ended.cleared, 0);
    }

    #[test]
    fn rt_standin_counts_each_drop_of_a_frame_in_that_frame_across_threads() {
        use ab::Arm;
        const THREADS: u64 = 8;
        const DROPS: u64 = 2000;
        let counts = Counts::new();
        assert_eq!(counts.count(1, Role::Particles, Arm::A), None);
        // Threads recording frame 2 at once: one of them ends frame 1, and
        // every drop of frame 2 is counted in frame 2, none in frame 1.
        let ended: Vec<Ended> = std::thread::scope(|s| {
            let threads: Vec<_> = (0..THREADS)
                .map(|_| {
                    s.spawn(|| {
                        (0..DROPS)
                            .filter_map(|_| counts.count(2, Role::ProbeGather, Arm::B))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            threads
                .into_iter()
                .flat_map(|t| t.join().unwrap())
                .collect()
        });
        assert_eq!(ended.len(), 1);
        assert_eq!((ended[0].frame, ended[0].arm), (1, Arm::A));
        assert_eq!(ended[0].drops.iter().sum::<u64>(), 1);
        let ended = counts.count(3, Role::Particles, Arm::B).unwrap();
        assert_eq!((ended.frame, ended.arm), (2, Arm::B));
        assert_eq!(ended.drops[Role::ProbeGather.index()], THREADS * DROPS);
        assert_eq!(ended.drops.iter().sum::<u64>(), THREADS * DROPS);
    }

    #[test]
    fn rt_standin_refuses_a_tracked_texture_write_under_a_gpu_count() {
        let texture = View::Unordered(0, 0, None);
        // Under a GPU count, a texture's view would be marked written
        // before the count is known: refused.
        assert!(tracked_write(&texture, false, true));
        // Without one the clear runs whenever it is recorded.
        assert!(!tracked_write(&texture, false, false));
        // A buffer range with a shadow tracks no texture write.
        assert!(!tracked_write(&texture, true, true));
        // Nor does a placed view.
        let mut resource = [0u32; 14];
        resource[0] = 3;
        assert!(!tracked_write(
            &View::Placed(true, resource, [0; 16]),
            false,
            true
        ));
    }

    #[test]
    fn rt_standin_clears_only_for_work_that_may_write() {
        assert!(may_write(&Work::Groups([1, 1, 1])));
        assert!(may_write(&Work::Groups([160, 90, 1])));
        for empty in [[0, 1, 1], [4, 0, 1], [4, 4, 0]] {
            assert!(!may_write(&Work::Groups(empty)), "{empty:?}");
        }
        assert!(!may_write(&Work::Commands(None)));
    }

    #[test]
    fn rt_standin_knows_placed_textures_from_placed_buffers() {
        let resource = |dimension| {
            let mut r = [0u32; 14];
            r[0] = dimension;
            r
        };
        assert!(placed_texture(&View::Placed(true, resource(3), [0; 16])));
        assert!(!placed_texture(&View::Placed(true, resource(1), [0; 16])));
        assert!(!placed_texture(&View::Placed(false, resource(3), [0; 16])));
        assert!(!placed_texture(&View::Unordered(0, 0, None)));
        // Buffers and other views go through as they are.
        let buffer = View::Placed(true, resource(1), [0; 16]);
        assert!(matches!(host_view(buffer), Some(View::Placed(true, r, _)) if r[0] == 1));
    }

    #[test]
    fn rt_standin_reports_drops_by_role() {
        let mut drops = [0u64; 7];
        assert_eq!(report(&drops), "none");
        drops[Role::ProbeGather.index()] = 3;
        drops[Role::Particles.index()] = 30;
        assert_eq!(report(&drops), "probe-gather 3, particles 30");
        assert!(due(REPORT_EVERY * 3));
    }
}
