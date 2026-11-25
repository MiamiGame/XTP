//! In-run A/B: a switch alternates between its two values within one
//! session, by phases timed at presents, and the frame line says which
//! phase it measured.
//!
//! A before/after across two sessions costs two runs on the PC, and the
//! two differ in scene and in what the caches hold. A switch read through
//! [`switch`] takes, besides `0` and `1`, `ab`, `ab:<phase s>` or
//! `ab:<phase s>:<settle s>`: phase A runs with the value `0` gives, phase
//! B with the value `1` gives, in turn, 20 s each unless told (2 to
//! 3600 s), from the first present after the switch was read. The first
//! 3 s of each phase settle (pipelines made and caches filled for the new
//! value) and are not measured. Every switch set to `ab` in a run follows
//! the same clock, so two set together alternate together (A: both `0`, B:
//! both `1`), timed as the first one read says.
//!
//! The clock moves at presents ([`present`]): a switch changes between two
//! frames, at the present that ends a phase, and code that needs a value
//! to hold for a whole frame reads it once per frame. Each phase logs when
//! it begins, with the values it runs, and when it ends its settled
//! presents, frame rate, longest frame and frames of 50 ms or more, with
//! its arm's totals so far. The frame line carries the [`Mark`] of the
//! frames it measured (`[A/B B3]`, `[A/B A0 settling]`), and [`summary`]
//! gives medians per arm of the frame lines measured past settling.
//!
//! Switches that change how shaders are translated or pipelines made
//! ([`FIXED`]) cannot alternate: a translation made in one phase would serve
//! the other, and the shader cache keys on them once per run. A run that
//! asks `ab` of one of them, or of a switch that does not alternate
//! ([`SWITCHES`]), is refused at start ([`check_environment`]).
//!
//! Ported from simnJS/AnyPS5 (commit 2eadda9b42, `KYTY_AB` in
//! `core/kyty/common/runtimeSwitches.cpp`: listed settings alternate at
//! flips within one session, with a settling time and per-phase frame
//! counts).

use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// The switches that may be set to `ab`: those read through [`switch`],
/// and those that still run an A/B of their own until they move onto this
/// module (`XTP_SNAPSHOT_MIRRORS`, by copystat lines). A switch read
/// through [`switch`] is added here.
pub const SWITCHES: &[&str] = &[
    "XTP_FRAMES_IN_FLIGHT",
    "XTP_SNAPSHOT_MIRRORS",
    "XTP_RT_STANDIN",
    "XTP_RT_PATH",
];

/// Switches read once, when shaders are translated or pipelines made,
/// which cannot alternate within a run: what a phase translated or made
/// would serve the other phase, the shader cache keys the run's
/// translations on the first three once ([`refusal`]), and keeping both
/// sets would translate and make every program twice, the second set in
/// the middle of the run (docs/dev/START_HERE.md, "A/B a switch").
pub const FIXED: &[&str] = &[
    "XTP_DXIL_OPT",
    "XTP_SHADER_LOOP_LIMIT",
    "XTP_SHADER_LOOP_BUDGET",
    "XTP_STRIDE_TABLE",
    "XTP_HOST_LANES",
    "XTP_WAVE64_MESH",
    "XTP_ROOT_SRV_DIRECT",
];

/// One arm of an A/B run: A runs a switch's `0`, B its `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arm {
    A,
    B,
}

impl Arm {
    /// The arm of phase `index`: A, B, A, …
    pub fn of(index: u64) -> Arm {
        if index.is_multiple_of(2) {
            Arm::A
        } else {
            Arm::B
        }
    }

    /// The switch value it runs: `false` for `0`, `true` for `1`.
    pub fn on(self) -> bool {
        self == Arm::B
    }

    fn index(self) -> usize {
        usize::from(self.on())
    }
}

impl fmt::Display for Arm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(if self.on() { "B" } else { "A" })
    }
}

/// How an A/B run is timed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    /// How long a phase lasts (to the first present past it).
    pub phase: Duration,
    /// How long a phase settles first, not measured.
    pub settle: Duration,
}

impl Plan {
    /// `ab` alone: phases of 20 s, the first 3 s of each settling.
    pub const DEFAULT: Plan = Plan {
        phase: Duration::from_secs(20),
        settle: Duration::from_secs(3),
    };
}

/// What a switch's value asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Asked {
    /// Unset or empty: the switch's default.
    Unset,
    /// `0`.
    Off,
    /// `1`.
    On,
    /// `ab`, `ab:<phase s>` or `ab:<phase s>:<settle s>`.
    Ab(Plan),
}

/// `name`'s value read: unset or empty, `0`, `1`, `ab`, `ab:<phase s>` or
/// `ab:<phase s>:<settle s>` (whole seconds, phases of 2 to 3600 s,
/// settling shorter than the phase).
pub fn parse(name: &str, value: Option<&str>) -> Result<Asked, String> {
    let Some(text) = value.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(Asked::Unset);
    };
    match text {
        "0" => return Ok(Asked::Off),
        "1" => return Ok(Asked::On),
        _ => {}
    }
    let bad = || format!("{name}={text}: 0, 1 or ab[:<phase s>[:<settle s>]]");
    let mut fields = text.split(':');
    if fields.next() != Some("ab") {
        return Err(bad());
    }
    let mut seconds = |default: Duration| match fields.next() {
        None => Ok(default),
        Some(f) => f
            .trim()
            .parse::<u64>()
            .map(Duration::from_secs)
            .map_err(|_| bad()),
    };
    let phase = seconds(Plan::DEFAULT.phase)?;
    let settle = seconds(Plan::DEFAULT.settle)?;
    let phases = Duration::from_secs(2)..=Duration::from_secs(3600);
    if fields.next().is_some() || !phases.contains(&phase) || settle >= phase {
        return Err(bad());
    }
    Ok(Asked::Ab(Plan { phase, settle }))
}

/// Why `name=value` is refused, when `value` asks for an A/B (`ab` or
/// `ab:…`) and `name` is a translation switch ([`FIXED`]) or one that does
/// not alternate ([`SWITCHES`]).
pub fn refusal(name: &str, value: &str) -> Option<String> {
    let value = value.trim();
    if value != "ab" && !value.starts_with("ab:") {
        return None;
    }
    if FIXED.contains(&name) {
        return Some(format!(
            "{name}={value}: {name} changes how shaders are translated or pipelines made, \
             which happens once per program and run, so it cannot alternate within one; \
             compare two runs instead (docs/dev/START_HERE.md, \"A/B a switch\")"
        ));
    }
    if name.starts_with("XTP_") && !SWITCHES.contains(&name) {
        return Some(format!(
            "{name}={value}: {name} does not alternate within a run; the switches that do: {}",
            SWITCHES.join(", ")
        ));
    }
    None
}

/// The first [`refusal`] the environment calls for: a binary stops on it
/// at start, rather than run an A/B that alternates nothing.
pub fn check_environment() -> Result<(), String> {
    check(
        std::env::vars_os().filter_map(|(name, value)| {
            Some((name.into_string().ok()?, value.into_string().ok()?))
        }),
    )
}

/// [`check_environment`] over `vars`. Pure, for testing.
fn check(vars: impl IntoIterator<Item = (String, String)>) -> Result<(), String> {
    for (name, value) in vars {
        if let Some(refused) = refusal(&name, &value) {
            return Err(refused);
        }
    }
    Ok(())
}

/// A switch read once ([`switch`]): its value, or its value in each arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Switch<T> {
    name: &'static str,
    value: Value<T>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Value<T> {
    Fixed(T),
    Ab { a: T, b: T },
}

impl<T: Copy> Switch<T> {
    /// Its value for the frame being prepared: its own, or as the phase
    /// says.
    #[inline]
    pub fn get(&self) -> T {
        self.at(current())
    }

    /// Its value in `arm` (a switch that does not alternate has one).
    pub fn at(&self, arm: Arm) -> T {
        match self.value {
            Value::Fixed(v) => v,
            Value::Ab { a, b } => {
                if arm.on() {
                    b
                } else {
                    a
                }
            }
        }
    }

    /// Whether it alternates in this run.
    pub fn alternates(&self) -> bool {
        matches!(self.value, Value::Ab { .. })
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

/// Reads the switch `name` from the environment: `0` gives
/// `T::from(false)`, `1` gives `T::from(true)`, unset gives `default`, and
/// `ab…` alternates the two from the next present ([`parse`]). A bad value
/// is logged and taken as unset. It reads the environment each call: keep
/// the switch in a static (`LazyLock`) and list its name in [`SWITCHES`].
pub fn switch<T: From<bool> + Copy>(name: &'static str, default: T) -> Switch<T> {
    let value = std::env::var(name).ok();
    let (switch, plan) = read(name, value.as_deref(), default);
    if let Some(plan) = plan {
        join(name, plan);
    }
    switch
}

/// [`switch`] for `value`, without the clock: the switch, and the plan it
/// asks for if it alternates. Pure, for testing.
fn read<T: From<bool> + Copy>(
    name: &'static str,
    value: Option<&str>,
    default: T,
) -> (Switch<T>, Option<Plan>) {
    let asked = parse(name, value).and_then(|asked| match (asked, value) {
        (Asked::Ab(_), Some(value)) => refusal(name, value).map_or(Ok(asked), Err),
        _ => Ok(asked),
    });
    let asked = asked.unwrap_or_else(|e| {
        tracing::warn!("{e}; taken as unset");
        Asked::Unset
    });
    let fixed = |value| Switch {
        name,
        value: Value::Fixed(value),
    };
    match asked {
        Asked::Unset => (fixed(default), None),
        Asked::Off => (fixed(T::from(false)), None),
        Asked::On => (fixed(T::from(true)), None),
        Asked::Ab(plan) => {
            let value = Value::Ab {
                a: T::from(false),
                b: T::from(true),
            };
            (Switch { name, value }, Some(plan))
        }
    }
}

/// The arm the frame being prepared runs in (0: A, 1: B).
static ARM: AtomicU8 = AtomicU8::new(0);
/// Whether a switch alternates: presents then move the clock.
static ACTIVE: AtomicBool = AtomicBool::new(false);
/// The run's clock, made by the first switch that alternates.
static CLOCK: Mutex<Option<Clock>> = Mutex::new(None);

/// The arm the frame being prepared runs in (A without an A/B).
#[inline]
pub fn current() -> Arm {
    if ARM.load(Ordering::Relaxed) == 1 {
        Arm::B
    } else {
        Arm::A
    }
}

/// Puts `name` on the run's clock, which the first switch to alternate
/// times.
fn join(name: &'static str, plan: Plan) {
    let mut clock = CLOCK.lock().unwrap_or_else(|e| e.into_inner());
    let line = match &mut *clock {
        None => {
            *clock = Some(Clock::new(plan, name));
            format!(
                "{name} alternates: A {name}=0, B {name}=1, from the next present ({})",
                plan_text(plan)
            )
        }
        Some(clock) => clock.join(name, plan),
    };
    drop(clock);
    tracing::info!("A/B: {line}");
    ACTIVE.store(true, Ordering::Release);
}

/// Moves the clock at a present, and gives the mark of the frames presented
/// since `since` (the frame line's window): the phase they ran in, and
/// whether part of the window came before the phase settled. Called once
/// per present, by the presenting thread; without an A/B, nothing.
pub fn present(since: Instant) -> Mark {
    if !ACTIVE.load(Ordering::Acquire) {
        return Mark(None);
    }
    let now = Instant::now();
    let (tag, lines) = {
        let mut clock = CLOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(clock) = clock.as_mut() else {
            return Mark(None);
        };
        let moved = clock.present(now, since);
        ARM.store(clock.arm() as u8, Ordering::Relaxed);
        moved
    };
    for line in lines {
        tracing::info!("A/B: {line}");
    }
    Mark(Some(tag))
}

/// How a plan reads in the log.
fn plan_text(plan: Plan) -> String {
    format!(
        "phases of {} s, the first {} s of each settling",
        plan.phase.as_secs(),
        plan.settle.as_secs()
    )
}

/// A frame of this length or longer is counted as long (AnyPS5's 50 ms).
const LONG_FRAME: Duration = Duration::from_millis(50);

/// The phase clock: phases timed at presents, A, B, A, …, each one's
/// presents past its settling counted. [`present`] runs the run's own.
#[derive(Debug)]
pub struct Clock {
    plan: Plan,
    /// The switches it alternates, in the order they joined.
    switches: Vec<&'static str>,
    /// Presents seen.
    presents: u64,
    /// When the first phase began.
    started: Option<Instant>,
    /// The phase running, from the first present on.
    phase: Option<Phase>,
    /// Per arm, what its phases measured past settling.
    totals: [Total; 2],
}

#[derive(Debug, Clone, Copy)]
struct Phase {
    index: u64,
    began: Instant,
    /// The last present.
    last: Instant,
    /// From the first present past its settling.
    settled: Option<Settled>,
}

#[derive(Debug, Clone, Copy)]
struct Settled {
    from: Instant,
    /// Presents since, the longest frame between two and frames of
    /// [`LONG_FRAME`] or more.
    presents: u64,
    longest: Duration,
    long: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Total {
    phases: u64,
    presents: u64,
    seconds: f64,
}

impl Clock {
    pub fn new(plan: Plan, name: &'static str) -> Clock {
        Clock {
            plan,
            switches: vec![name],
            presents: 0,
            started: None,
            phase: None,
            totals: [Total::default(); 2],
        }
    }

    pub fn plan(&self) -> Plan {
        self.plan
    }

    /// Adds `name`, which asked for `plan`: it follows the clock's own.
    /// Returns the line saying so.
    pub fn join(&mut self, name: &'static str, plan: Plan) -> String {
        if !self.switches.contains(&name) {
            self.switches.push(name);
        }
        let when = match self.phase {
            Some(p) => format!("from phase {}{}", Arm::of(p.index), p.index),
            None => "from the next present".into(),
        };
        let timing = if plan == self.plan {
            String::new()
        } else {
            format!(
                "; it asked for {}, the run has {}",
                plan_text(plan),
                plan_text(self.plan)
            )
        };
        format!("{name} alternates with the others {when}{timing}")
    }

    /// The arm the frames after the last present run in.
    pub fn arm(&self) -> Arm {
        self.phase.map_or(Arm::A, |p| Arm::of(p.index))
    }

    /// The switches' values in `arm`, as the log gives them.
    fn values(&self, arm: Arm) -> String {
        let value = u8::from(arm.on());
        self.switches
            .iter()
            .map(|name| format!("{name}={value}"))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// A present at `now`: the tag of the frames presented since `since`,
    /// and the lines to log (a phase that ended, one that began). A phase
    /// ends at the first present past its length; the frames after that
    /// present run in the next.
    pub fn present(&mut self, now: Instant, since: Instant) -> (Tag, Vec<String>) {
        self.presents += 1;
        let present = self.presents;
        let mut lines = Vec::new();
        let Some(mut phase) = self.phase else {
            // Phase 0 begins; the frames before it ran before the A/B.
            lines.push(format!(
                "{}; A: {}; B: {}",
                plan_text(self.plan),
                self.values(Arm::A),
                self.values(Arm::B)
            ));
            lines.push(format!(
                "phase A0 begins at present {present}: {}",
                self.values(Arm::A)
            ));
            self.started = Some(now);
            self.phase = Some(Phase::new(0, now));
            let tag = Tag {
                arm: Arm::A,
                phase: 0,
                settling: true,
            };
            return (tag, lines);
        };
        let frame = now.saturating_duration_since(phase.last);
        phase.last = now;
        let settled_at = phase.began + self.plan.settle;
        match &mut phase.settled {
            Some(s) => {
                s.presents += 1;
                s.longest = s.longest.max(frame);
                s.long += u64::from(frame >= LONG_FRAME);
            }
            None if now >= settled_at => {
                phase.settled = Some(Settled {
                    from: now,
                    presents: 0,
                    longest: Duration::ZERO,
                    long: 0,
                });
            }
            None => {}
        }
        let arm = Arm::of(phase.index);
        let tag = Tag {
            arm,
            phase: phase.index,
            settling: since < settled_at,
        };
        if now.saturating_duration_since(phase.began) < self.plan.phase {
            self.phase = Some(phase);
            return (tag, lines);
        }
        lines.push(self.close(&phase, present, now));
        let next = Phase::new(phase.index + 1, now);
        let in_run = self
            .started
            .map_or(0.0, |s| now.saturating_duration_since(s).as_secs_f64());
        let arm = Arm::of(next.index);
        lines.push(format!(
            "phase {arm}{} begins at present {present}, {in_run:.1} s in: {}",
            next.index,
            self.values(arm)
        ));
        self.phase = Some(next);
        (tag, lines)
    }

    /// The line closing `phase` at present `present` (`now`), its settled
    /// presents added to its arm's totals.
    fn close(&mut self, phase: &Phase, present: u64, now: Instant) -> String {
        let arm = Arm::of(phase.index);
        let length = now.saturating_duration_since(phase.began).as_secs_f64();
        let head = format!(
            "phase {arm}{} ends at present {present} after {length:.1} s",
            phase.index
        );
        let Some(s) = phase.settled.filter(|s| s.presents > 0) else {
            return format!(
                "{head}: nothing measured (no frame wholly past its first {} s)",
                self.plan.settle.as_secs()
            );
        };
        let seconds = now.saturating_duration_since(s.from).as_secs_f64();
        let total = &mut self.totals[arm.index()];
        total.phases += 1;
        total.presents += s.presents;
        total.seconds += seconds;
        format!(
            "{head}: {} presents in {seconds:.1} s settled, {:.2} fps, longest frame {:.0} ms, \
             {} of {} ms or more; {arm} so far: {} phases, {} presents in {:.1} s, {:.2} fps",
            s.presents,
            rate(s.presents, seconds),
            s.longest.as_secs_f64() * 1000.0,
            s.long,
            LONG_FRAME.as_millis(),
            total.phases,
            total.presents,
            total.seconds,
            rate(total.presents, total.seconds),
        )
    }
}

impl Phase {
    fn new(index: u64, now: Instant) -> Phase {
        Phase {
            index,
            began: now,
            last: now,
            settled: None,
        }
    }
}

/// Presents per second.
fn rate(presents: u64, seconds: f64) -> f64 {
    if seconds > 0.0 {
        presents as f64 / seconds
    } else {
        0.0
    }
}

/// Which phase a window of frames ran in, and whether part of it came
/// before the phase settled (a window across a flip is settling too).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tag {
    pub arm: Arm,
    /// The phase's index from the run's first: A0, B1, A2, …
    pub phase: u64,
    pub settling: bool,
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let settling = if self.settling { " settling" } else { "" };
        write!(f, "[A/B {}{}{settling}]", self.arm, self.phase)
    }
}

impl Tag {
    /// The tag a log line carries, as [`Tag`]'s `Display` writes it.
    pub fn find(line: &str) -> Option<Tag> {
        let rest = &line[line.find("[A/B ")? + 5..];
        let inner = &rest[..rest.find(']')?];
        let (phase, settling) = match inner.strip_suffix(" settling") {
            Some(phase) => (phase, true),
            None => (inner, false),
        };
        let arm = match phase.as_bytes().first()? {
            b'A' => Arm::A,
            b'B' => Arm::B,
            _ => return None,
        };
        let phase: u64 = phase[1..].parse().ok()?;
        (Arm::of(phase) == arm).then_some(Tag {
            arm,
            phase,
            settling,
        })
    }
}

/// The frame line's mark ([`present`]): a [`Tag`] and a space in an A/B
/// run, nothing otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mark(pub Option<Tag>);

impl fmt::Display for Mark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(tag) => write!(f, "{tag} "),
            None => Ok(()),
        }
    }
}

/// Reads one figure of a frame line.
pub type Figure = fn(&str) -> Option<f64>;

/// For the end of a session: the median per arm of each figure of the
/// frame lines that carry a settled [`Tag`], and B's change from A.
/// Nothing when no line carries a tag.
pub fn summary<S: AsRef<str>>(lines: &[S], figures: &[(&str, Figure)]) -> Vec<String> {
    let mut arms: [Vec<&str>; 2] = [Vec::new(), Vec::new()];
    let mut settling = 0;
    for line in lines.iter().map(AsRef::as_ref) {
        match Tag::find(line) {
            Some(tag) if !tag.settling => arms[tag.arm.index()].push(line),
            Some(_) => settling += 1,
            None => {}
        }
    }
    let [a, b] = &arms;
    if a.is_empty() && b.is_empty() && settling == 0 {
        return Vec::new();
    }
    let mut out = vec![
        format!(
            "A/B, median per arm of the frame lines past settling ({} A, {} B; {settling} \
             settling left out):",
            a.len(),
            b.len()
        ),
        format!("  {:<16} {:>8} {:>8} {:>8}", "", "A", "B", "B/A"),
    ];
    for (name, read) in figures {
        let [ma, mb] = [a, b].map(|lines| median(lines.iter().filter_map(|l| read(l))));
        if ma.is_none() && mb.is_none() {
            continue;
        }
        let shown = |m: Option<f64>| m.map_or_else(|| "-".into(), |v| format!("{v:.1}"));
        let change = match (ma, mb) {
            (Some(a), Some(b)) if a != 0.0 => format!("{:+.1}%", (b / a - 1.0) * 100.0),
            _ => "-".into(),
        };
        out.push(format!(
            "  {name:<16} {:>8} {:>8} {change:>8}",
            shown(ma),
            shown(mb)
        ));
    }
    out
}

/// The median of `values` (the upper one of an even count), as `xtask
/// play`'s table takes it.
fn median(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: fn(u64) -> Duration = Duration::from_secs;

    #[test]
    fn a_switch_takes_0_1_and_ab() {
        let p = |v| parse("XTP_X", v);
        assert_eq!(p(None), Ok(Asked::Unset));
        assert_eq!(p(Some(" ")), Ok(Asked::Unset));
        assert_eq!(p(Some("0")), Ok(Asked::Off));
        assert_eq!(p(Some(" 1 ")), Ok(Asked::On));
        assert_eq!(p(Some("ab")), Ok(Asked::Ab(Plan::DEFAULT)));
        let plan = |phase, settle| {
            Ok(Asked::Ab(Plan {
                phase: S(phase),
                settle: S(settle),
            }))
        };
        assert_eq!(p(Some("ab:30")), plan(30, 3));
        assert_eq!(p(Some("ab:30:5")), plan(30, 5));
        assert_eq!(p(Some("ab:2:0")), plan(2, 0));
        assert_eq!(p(Some("ab:3600:1")), plan(3600, 1));
        for bad in [
            "yes",
            "2",
            "ab:1",
            "ab:3601",
            "ab:10:10",
            "ab:x",
            "ab:10:x",
            "ab:10:2:1",
            "ba",
            "ab:",
            "AB",
            "abc",
        ] {
            let e = p(Some(bad)).expect_err(bad);
            assert!(e.starts_with(&format!("XTP_X={bad}: 0, 1 or ab")), "{e}");
        }
    }

    #[test]
    fn a_switch_gives_its_value_or_each_arms() {
        let (off, plan) = read("XTP_X", Some("0"), true);
        assert_eq!((off.at(Arm::A), off.at(Arm::B), plan), (false, false, None));
        let (on, _) = read("XTP_X", Some("1"), false);
        assert_eq!((on.at(Arm::A), on.at(Arm::B)), (true, true));
        let (unset, _) = read("XTP_X", None, true);
        assert!(unset.at(Arm::A) && !unset.alternates());
        // A bad value is taken as unset.
        let (bad, plan) = read("XTP_X", Some("maybe"), true);
        assert!(bad.at(Arm::B) && plan.is_none());
        let (ab, plan) = read("XTP_FRAMES_IN_FLIGHT", Some("ab:10:2"), true);
        assert!(ab.alternates());
        assert_eq!((ab.at(Arm::A), ab.at(Arm::B)), (false, true));
        assert_eq!(
            plan,
            Some(Plan {
                phase: S(10),
                settle: S(2)
            })
        );
        assert_eq!(ab.name(), "XTP_FRAMES_IN_FLIGHT");
        // A switch read here but not listed does not alternate.
        let (unlisted, plan) = read("XTP_UNLISTED", Some("ab"), true);
        assert!(!unlisted.alternates() && plan.is_none() && unlisted.at(Arm::B));
    }

    #[test]
    fn translation_switches_and_unlisted_ones_refuse_ab() {
        let e = refusal("XTP_DXIL_OPT", "ab").unwrap();
        assert!(e.starts_with("XTP_DXIL_OPT=ab: XTP_DXIL_OPT changes how shaders are translated"));
        assert!(refusal("XTP_SHADER_LOOP_BUDGET", " ab:30:5 ").is_some());
        assert!(refusal("XTP_DXIL_OPT", "dead,fold").is_none());
        assert!(refusal("XTP_DXIL_OPT", "1").is_none());
        let e = refusal("XTP_SYNC_SUBMIT", "ab").unwrap();
        assert!(e.contains("does not alternate"), "{e}");
        assert!(e.ends_with(&SWITCHES.join(", ")), "{e}");
        assert!(refusal("XTP_FRAMES_IN_FLIGHT", "ab:20:3").is_none());
        assert!(refusal("XTP_SNAPSHOT_MIRRORS", "ab").is_none());
        // Not ours, or not asking for an A/B.
        assert!(refusal("PATH", "ab").is_none());
        assert!(refusal("XTP_SYNC_SUBMIT", "abc").is_none());
        let vars = |list: &[(&str, &str)]| {
            list.iter()
                .map(|(n, v)| ((*n).to_string(), (*v).to_string()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            check(vars(&[("XTP_FRAMES_IN_FLIGHT", "ab"), ("XTP_LOG", "info")])),
            Ok(())
        );
        let e = check(vars(&[
            ("XTP_FRAMES_IN_FLIGHT", "ab"),
            ("XTP_STRIDE_TABLE", "ab"),
        ]));
        assert!(e.unwrap_err().starts_with("XTP_STRIDE_TABLE=ab: "));
        // Every translation switch is refused, and none is listed as
        // alternating.
        for name in FIXED {
            assert!(!SWITCHES.contains(name), "{name}");
            assert!(
                refusal(name, "ab").unwrap().contains("translated"),
                "{name}"
            );
        }
    }

    /// A plan of 4 s phases settling 1 s, a present every half second.
    fn clock() -> (Clock, impl Fn(u64) -> Instant) {
        let t0 = Instant::now();
        let at = move |half_seconds: u64| t0 + Duration::from_millis(500 * half_seconds);
        let plan = Plan {
            phase: S(4),
            settle: S(1),
        };
        (Clock::new(plan, "XTP_X"), at)
    }

    #[test]
    fn the_clock_alternates_at_presents_past_a_phase() {
        let (mut clock, at) = clock();
        assert_eq!(clock.arm(), Arm::A);
        // The first present begins phase A0.
        let (tag, lines) = clock.present(at(0), at(0));
        assert_eq!((tag.arm, tag.phase, tag.settling), (Arm::A, 0, true));
        assert_eq!(
            lines,
            [
                "phases of 4 s, the first 1 s of each settling; A: XTP_X=0; B: XTP_X=1",
                "phase A0 begins at present 1: XTP_X=0"
            ]
        );
        for t in 1..8 {
            let (tag, lines) = clock.present(at(t), at(t - 1));
            assert_eq!((tag.arm, tag.phase), (Arm::A, 0), "{t}");
            assert!(lines.is_empty(), "{t}");
            assert_eq!(clock.arm(), Arm::A);
        }
        // At 4 s phase A0 ends: the frames up to it ran in A, those after
        // run in B.
        let (tag, lines) = clock.present(at(8), at(7));
        assert_eq!((tag.arm, tag.phase), (Arm::A, 0));
        assert_eq!(clock.arm(), Arm::B);
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].starts_with("phase A0 ends at present 9 after 4.0 s: "),
            "{}",
            lines[0]
        );
        assert_eq!(lines[1], "phase B1 begins at present 9, 4.0 s in: XTP_X=1");
        for t in 9..16 {
            let (tag, _) = clock.present(at(t), at(t - 1));
            assert_eq!((tag.arm, tag.phase), (Arm::B, 1), "{t}");
        }
        let (tag, lines) = clock.present(at(16), at(15));
        assert_eq!((tag.arm, tag.phase), (Arm::B, 1));
        assert_eq!(clock.arm(), Arm::A);
        assert_eq!(lines[1], "phase A2 begins at present 17, 8.0 s in: XTP_X=0");
    }

    #[test]
    fn the_clock_flags_windows_that_began_while_settling() {
        let (mut clock, at) = clock();
        clock.present(at(0), at(0));
        // A window from 0 to 0.5 s and one from 0.5 to 1 s began before the
        // phase settled (1 s); one from 1 s on did not.
        assert!(clock.present(at(1), at(0)).0.settling);
        assert!(clock.present(at(2), at(1)).0.settling);
        assert!(!clock.present(at(3), at(2)).0.settling);
        // A window of 2 s ending at 3 s began at 1 s: settled.
        clock.present(at(4), at(3));
        assert!(!clock.present(at(6), at(2)).0.settling);
        clock.present(at(7), at(6));
        clock.present(at(8), at(7));
        // B1 began at 4 s: a window across the flip, from 3.5 to 4.5 s, is
        // B1's and settling.
        let tag = clock.present(at(9), at(7)).0;
        assert_eq!((tag.arm, tag.phase, tag.settling), (Arm::B, 1, true));
        let tag = clock.present(at(11), at(10)).0;
        assert_eq!((tag.arm, tag.phase, tag.settling), (Arm::B, 1, false));
    }

    #[test]
    fn the_clock_counts_each_phases_settled_presents() {
        let (mut clock, at) = clock();
        clock.present(at(0), at(0));
        // Phase A0: presents at 0.5, 1 (settled from here), 1.5, 2, 3 (a
        // frame of 1 s), 3.5, then 4 (the end).
        for t in [1, 2, 3, 4, 6, 7] {
            clock.present(at(t), at(t - 1));
        }
        let (_, lines) = clock.present(at(8), at(7));
        assert_eq!(
            lines[0],
            "phase A0 ends at present 8 after 4.0 s: 5 presents in 3.0 s settled, 1.67 fps, \
             longest frame 1000 ms, 5 of 50 ms or more; A so far: 1 phases, 5 presents in 3.0 \
             s, 1.67 fps"
        );
        // Phase B1 (4 to 8 s): a present every half second.
        for t in 9..16 {
            clock.present(at(t), at(t - 1));
        }
        let (_, lines) = clock.present(at(16), at(15));
        assert!(
            lines[0].starts_with(
                "phase B1 ends at present 16 after 4.0 s: 6 presents in 3.0 s settled, 2.00 fps, \
                 longest frame 500 ms"
            ),
            "{}",
            lines[0]
        );
        assert!(lines[0].ends_with("B so far: 1 phases, 6 presents in 3.0 s, 2.00 fps"));
        // Phase A2 (8 to 12 s): one frame of 4 s, nothing past settling.
        let (_, lines) = clock.present(at(24), at(16));
        assert_eq!(
            lines[0],
            "phase A2 ends at present 17 after 4.0 s: nothing measured (no frame wholly past \
             its first 1 s)"
        );
        // Phase B3 adds to B's totals.
        for t in 25..32 {
            clock.present(at(t), at(t - 1));
        }
        let (_, lines) = clock.present(at(32), at(31));
        assert!(lines[0].ends_with("B so far: 2 phases, 12 presents in 6.0 s, 2.00 fps"));
    }

    #[test]
    fn switches_joining_follow_the_first_ones_plan() {
        let (mut clock, at) = clock();
        let line = clock.join("XTP_Y", Plan::DEFAULT);
        assert_eq!(
            line,
            "XTP_Y alternates with the others from the next present; it asked for phases of 20 \
             s, the first 3 s of each settling, the run has phases of 4 s, the first 1 s of \
             each settling"
        );
        assert_eq!(clock.plan().phase, S(4));
        let (_, lines) = clock.present(at(0), at(0));
        assert_eq!(lines[1], "phase A0 begins at present 1: XTP_X=0, XTP_Y=0");
        assert_eq!(
            clock.join("XTP_Z", clock.plan()),
            "XTP_Z alternates with the others from phase A0"
        );
        // Joining twice lists it once.
        clock.join("XTP_Z", clock.plan());
        assert_eq!(clock.values(Arm::B), "XTP_X=1, XTP_Y=1, XTP_Z=1");
    }

    #[test]
    fn tags_read_back_from_a_line() {
        for tag in [
            Tag {
                arm: Arm::A,
                phase: 0,
                settling: true,
            },
            Tag {
                arm: Arm::B,
                phase: 13,
                settling: false,
            },
        ] {
            let line = format!(
                "2026-11-24T13:52:27Z  INFO xtp_gpu::host: {} 1.4 fps",
                Mark(Some(tag))
            );
            assert_eq!(Tag::find(&line), Some(tag), "{line}");
        }
        assert_eq!(Mark(None).to_string(), "");
        assert_eq!(
            Mark(Some(Tag {
                arm: Arm::B,
                phase: 3,
                settling: false
            }))
            .to_string(),
            "[A/B B3] "
        );
        for bad in [
            "1.4 fps",
            "[A/B ] x",
            "[A/B B2] x",
            "[A/B C1]",
            "[A/B A0 settled]",
            "[A/B A0",
        ] {
            assert_eq!(Tag::find(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_summary_gives_medians_per_arm_past_settling() {
        let figures: [(&str, Figure); 2] = [
            ("fps", |l| figure(l, " fps,")),
            ("ms on the GPU", |l| figure(l, " on the GPU")),
        ];
        let line = |tag: &str, fps: f64, gpu: f64| {
            format!("INFO xtp_gpu::host: {tag} {fps:.1} fps, 33.3 ms/frame ({gpu:.1} on the GPU)")
        };
        let lines = [
            line("[A/B A0 settling]", 99.0, 99.0),
            line("[A/B A0]", 20.0, 40.0),
            line("[A/B A0]", 22.0, 30.0),
            line("[A/B A0]", 21.0, 35.0),
            line("[A/B B1 settling]", 1.0, 1.0),
            line("[A/B B1]", 30.0, 20.0),
            line("[A/B B1]", 32.0, 10.0),
            line("", 5.0, 5.0),
        ];
        let report = summary(&lines, &figures);
        assert_eq!(
            report,
            [
                "A/B, median per arm of the frame lines past settling (3 A, 2 B; 2 settling left \
                 out):",
                "                          A        B      B/A",
                "  fps                  21.0     32.0   +52.4%",
                "  ms on the GPU        35.0     20.0   -42.9%",
            ]
        );
        // Only one arm measured yet.
        let report = summary(&lines[..4], &figures);
        assert_eq!(report[2], "  fps                  21.0        -        -");
        // No A/B in the run: nothing.
        assert!(summary(&[line("", 5.0, 5.0)], &figures).is_empty());
    }

    /// The number written just before `label`.
    fn figure(line: &str, label: &str) -> Option<f64> {
        let head = &line[..line.find(label)?];
        head.rsplit(|c: char| c.is_whitespace() || c == '(')
            .next()?
            .parse()
            .ok()
    }
}
