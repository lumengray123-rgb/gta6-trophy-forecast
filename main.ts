// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01): a structural
// cohort-maturity model.
//
// A trophy's earn rate is earners / players-tracked, and in launch week both
// move fast for different reasons: players-tracked grows as new players start
// (and Exophase picks them up), earners grow as players progress through the
// game. Persistence (the house baseline) misses both forces; a pure trend fit
// extrapolates the recent dilution and misses the catch-up as the day-0 cohort
// matures into late-game trophies. This program separates the two forces:
//
//   1. Players: log players-tracked is fit linear in log(1 + hours/scale)
//      (a power law) with the scale fit too — the scale sets where the curve
//      bends, and the wrong scale badly mis-extrapolates. The fit gives the
//      player count at the target and the hourly arrival cohorts behind it.
//   2. Earners: for each trophy, the earners above the launch-night baseline
//      are fit to a lognormal maturity kernel over those cohorts: a player who
//      started s hours ago has earned the trophy with probability
//      F * Phi((ln(age) - ln(delta)) / sigma). (delta, sigma, F) is only
//      weakly identified from 2.5 days of data, so instead of the single best
//      grid cell the forecast averages the cells' forecasts, weighted by fit,
//      with guards: cells need enough matured players to identify F, F is
//      capped by tier, slow maturation pays a prior penalty, and high-rate
//      trophies are scored on their non-earners (near 100%, the lag lives in
//      the few percent who have not earned it yet). A trophy that has not
//      started maturing gets a late-bloomer prior instead of a fit.
//   3. Two simpler opinions hedge the structural one: a log-log trend of the
//      earner count (slope shrunk toward the player-growth slope) and a
//      log-linear trend of the logit rate (slope halved). The median blends
//      the three; the spread widens with their disagreement, the spread
//      across plausible kernel cells, the fit residuals, the horizon, and
//      how much of a trophy's earners are still just the launch-night
//      baseline.
//
// Exophase refreshes on its own schedule, so consecutive identical reads are
// collapsed to distinct snapshots before any fitting. Everything is pure
// arithmetic on the case file: deterministic, no network, standard library
// only (Node 22 type-strippable TypeScript).
//
// The design was validated on synthetic final-like cases (a separate
// simulator, not part of this entry): across five scenarios it scores a
// pinball loss of roughly 0.21-0.41, against 0.69-1.18 for the house baseline
// and 0.43-1.30 for the house curve baseline.
//
// Run locally: mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Random-walk spread, in logit units per square-root day. */
const SIGMA_RW = 0.2;
/** Weight of model disagreement in the spread (multiplies the opinion range). */
const W_DISAGREE = 0.25;
/** Extra spread for Exophase refresh staleness (the answer read may reflect a page state hours old). */
const SIGMA_REFRESH = 0.15;
/** Floor / cap on the total logit spread. */
const SPREAD_MIN = 0.3;
const SPREAD_MAX = 2.2;
/** How much of an earner trend's excess over player growth survives (catch-up decays). */
const SLOPE_SHRINK = 0.4;
/** How much of a rate trend's slope survives (recent trends over-extrapolate). */
const TREND_SHRINK = 0.5;
/** Cap on earner growth over the horizon, as a multiple of player growth. */
const GROWTH_CAP = 2.5;
/** Blend weights for the three median opinions: kernel, earner trend, rate trend. */
const W_KERNEL = 0.55;
const W_EARN = 0.3;
const W_TREND = 0.15;
/** Candidate scales (hours) for the launch clock log(1 + h / s). */
const SCALES = [0.5, 1, 2, 3, 5, 8, 13, 21];
/** A kernel cell is fittable only if at least this many players have matured by the last snapshot. */
const KERNEL_MIN_MATURED = 10;
/** Prior penalty weight on slow maturation: score += W_DELTA_PRIOR * ln(delta/DELTA_PRIOR_AT)^2 for delta > DELTA_PRIOR_AT. */
const W_DELTA_PRIOR = 0.05;
const DELTA_PRIOR_AT = 72;
/** Temperature for averaging over kernel grid cells: weight = exp(-(score - best) / CELL_TEMP). */
const CELL_TEMP = 0.05;
/** Per-tier cap on F (the eventual fraction of mature players earning the trophy). */
const F_CAP: Record<string, number> = { platinum: 0.05, gold: 0.35, silver: 0.5, bronze: 1 };
/** Per-tier prior F for trophies that have not started maturing by the checkpoint. */
const F_PRIOR: Record<string, number> = { platinum: 0.01, gold: 0.02, silver: 0.06, bronze: 0.12 };
/** Prior median time-to-earn (hours) for not-yet-maturing trophies. */
const LATE_DELTA = 96;
/** Extra spread for trophies whose earners are still mostly the launch-night baseline. */
const IMMATURE_SPREAD = 0.7;

const DEBUG = process.env.MAKEFUN_DEBUG === '1';

/** Candidate median time-to-earn values (hours) for the maturity kernel. */
const DELTAS = [1, 2, 3, 5, 8, 12, 18, 24, 36, 48, 72, 96, 120, 160, 220];
/** Candidate log-spreads of the time-to-earn distribution. */
const SIGMAS = [0.35, 0.5, 0.7];

interface TrophyMeta {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: TrophyMeta[];
  history: Read[];
  notes?: string;
}

interface Snapshot {
  /** Hours since the release anchor. */
  h: number;
  /** Players tracked at this snapshot. */
  p: number;
  rates: Array<number | null>;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** erf, Abramowitz-Stegun 7.1.26 (|error| <= 1.5e-7). Node has no Math.erf. */
function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly =
    t *
    (0.254829592 +
      t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return sign * (1 - poly * Math.exp(-ax * ax));
}
const normCdf = (x: number): number => 0.5 * (1 + erf(x / Math.SQRT2));

/** Ordinary least squares for y ~ a + b*x; null when degenerate. */
function ols(xs: number[], ys: number[]): { a: number; b: number; rmse: number } | null {
  const n = xs.length;
  if (n < 2) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx < 1e-12) return null;
  const b = sxy / sxx;
  const a = my - b * mx;
  let sse = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    sse += r * r;
  }
  return { a, b, rmse: Math.sqrt(sse / n) };
}

/** log(1 + h / s), floored so pre-release reads (h slightly negative) stay valid. */
const logTScale = (h: number, s: number): number => Math.log(1 + Math.max(h / s, -0.9));

function forecastCase(c: CaseFile): {
  rows: Array<{ id: number; q: number[] }>;
  debug: { pTarget: number; growth: number; betaP: number; scaleP: number; nSnaps: number; trophies: Array<Record<string, number | string | null>> };
} {
  const anchorMs =
    Date.parse(`${c.game.release_date}T00:00:00Z`) || Date.parse('2026-11-19T00:00:00Z');
  const hoursSince = (iso: string): number => (Date.parse(iso) - anchorMs) / 3_600_000;
  const atH = hoursSince(c.at);
  const targetH = atH + c.horizon_minutes / 60;
  const days = c.horizon_minutes / 1440;

  // Distinct snapshots: collapse consecutive identical reads (Exophase frozen
  // between refreshes) so frozen stretches do not overweight the fits. Each
  // snapshot keeps its first-seen time.
  const snaps: Snapshot[] = [];
  for (const r of c.history) {
    const prev = snaps[snaps.length - 1];
    if (
      prev !== undefined &&
      prev.p === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    snaps.push({ h: hoursSince(r.at), p: r.players_tracked, rates: r.rates });
  }
  if (snaps.length === 0) throw new Error(`${c.id} has no history`);
  const lastSnap = snaps[snaps.length - 1];
  const pLast = Math.max(lastSnap.p, 1);

  // ---- Player growth: log P linear in log(1 + h / s) (power law). ----
  // The scale s is fit too: it sets where the power law bends, and the wrong
  // scale badly mis-extrapolates (a scale of 1 under-forecasts the target's
  // players by a third on launch-week-shaped data). Pick the scale with the
  // lowest fit residual; its clock is then shared by every trend in the case.
  let betaP = 0;
  let alphaP = Math.log(pLast);
  let scaleP = 1;
  if (snaps.length >= 3) {
    let bestRmse = Infinity;
    for (const s of SCALES) {
      const fit = ols(
        snaps.map((x) => logTScale(x.h, s)),
        snaps.map((x) => Math.log(Math.max(x.p, 1))),
      );
      if (fit !== null && fit.rmse < bestRmse) {
        bestRmse = fit.rmse;
        betaP = Math.min(3, Math.max(0, fit.b));
        alphaP = fit.a;
        scaleP = s;
      }
    }
  }
  const logT = (h: number): number => logTScale(h, scaleP);
  const pModel = (h: number): number => Math.exp(alphaP + betaP * logT(h));
  const pTarget = Math.max(pModel(targetH), pLast);
  const growth = Math.min(50, Math.max(1, pTarget / pLast));

  // ---- Hourly arrival cohorts from the player model. ----
  // Cohort s holds the players who arrived during hour s (midpoint s + 0.5).
  // The release population itself is NOT a cohort here: its earners are
  // already counted in each trophy's launch-night baseline (see below), and
  // counting them again would double-count.
  const cohorts: Array<{ mid: number; n: number }> = [];
  const nHours = Math.max(1, Math.ceil(targetH));
  let prevP = pModel(0);
  for (let s = 0; s < nHours; s++) {
    const pNext = pModel(s + 1);
    const n = Math.max(0, pNext - prevP);
    if (n > 0) cohorts.push({ mid: s + 0.5, n });
    prevP = pNext;
  }

  // ---- Maturity kernel, precomputed per snapshot / target and grid cell. ----
  // K(t; delta, sigma): players who have had enough time to earn a trophy
  // whose time-to-earn is lognormal with median delta and log-spread sigma.
  const kernelAt = (t: number, delta: number, sigma: number): number => {
    const ld = Math.log(delta);
    let sum = 0;
    for (const co of cohorts) {
      const age = t - co.mid;
      if (age <= 0) continue;
      sum += co.n * normCdf((Math.log(age) - ld) / sigma);
    }
    return sum;
  };
  const kSnap: number[][][] = snaps.map(() =>
    DELTAS.map(() => SIGMAS.map(() => 0)),
  );
  for (let j = 0; j < snaps.length; j++) {
    for (let di = 0; di < DELTAS.length; di++) {
      for (let si = 0; si < SIGMAS.length; si++) {
        kSnap[j][di][si] = kernelAt(snaps[j].h, DELTAS[di], SIGMAS[si]);
      }
    }
  }
  const kTarget: number[][] = DELTAS.map((d, di) =>
    SIGMAS.map((s, si) => kernelAt(targetH, DELTAS[di], SIGMAS[si])),
  );

  const rows: Array<{ id: number; q: number[] }> = [];
  const debugRows: Array<Record<string, number | string | null>> = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const trophy = c.trophies[i];
    // Earner series for this trophy, with snapshot indices for the kernel table.
    const series: Array<{ snap: number; h: number; e: number; r: number }> = [];
    for (let j = 0; j < snaps.length; j++) {
      const rate = snaps[j].rates[i];
      if (rate === null || rate === undefined) continue;
      series.push({ snap: j, h: snaps[j].h, e: (rate * snaps[j].p) / 100, r: rate });
    }
    const lastRate = series.length > 0 ? series[series.length - 1].r : 0;
    const lastE = series.length > 0 ? series[series.length - 1].e : 0;
    const lastH = series.length > 0 ? series[series.length - 1].h : atH;

    // No earners yet: a handful of players will earn it by the target, but
    // the rate stays tiny; say so with a wide spread.
    if (lastE <= 0) {
      const m = logit(0.05);
      rows.push({ id: trophy.id, q: quantiles(m, 1.5) });
      if (DEBUG) {
        debugRows.push({
          id: trophy.id, tier: trophy.tier, name: trophy.name, lastRate, lastE,
          mKernel: null, mEarn: null, mTrend: null, median: m, spread: 1.5,
          fitDelta: null, fitSigma: null, fitF: null, residKernel: 0,
        });
      }
      continue;
    }

    const opinions: Array<{ m: number; w: number }> = [];
    let residKernel = 0;
    let mKernel: number | null = null;
    let mEarn: number | null = null;
    let mTrend: number | null = null;
    let fitDelta: number | null = null;
    let fitSigma: number | null = null;
    let fitF: number | null = null;

    // Opinion 1: cohort-maturity kernel. Earners above the launch-night
    // baseline (reviewers / early players, a constant offset) follow
    // F * K(t; delta, sigma). Grid-search (delta, sigma); F by least squares
    // in levels, scored by mean squared log residual with a +0.5 continuity
    // correction so the flat pre-rise stretch counts too.
    //
    // Three guards keep the grid honest: a cell is fittable only once enough
    // players have matured by the last snapshot (otherwise F is 0/0 and the
    // fit invents any value); F is capped by tier (a platinum is 100%
    // completion, so F near 1 is absurd); and slow maturation pays a prior
    // penalty, because over a 2.5-day window a large delta with a large F
    // mimics a small delta with a small F, yet they diverge wildly at the
    // target. For high-rate trophies the cells are scored on the NON-earners
    // instead: near 100%, the lag structure lives in the few percent who
    // have not earned it yet, and the earner curve cannot see it.
    //
    // (delta, sigma, F) is only weakly identified from 2.5 days of data: the
    // fit score is nearly flat along a ridge, and the forecasts along that
    // ridge differ by a lot. So instead of the single best cell, the kernel
    // opinion averages the cells' forecasts, weighted by exp(-score / temp),
    // and the weighted spread across cells measures that identification
    // uncertainty.
    //
    // A trophy whose earners never grew past the launch-night baseline has no
    // fittable kernel at all: it has not started maturing yet. For those, a
    // late-bloomer prior (median time-to-earn ~4 days, tier F) stands in for
    // the kernel opinion — the day-0 cohort will reach it during the horizon,
    // just later than the observed window shows.
    const fCap = F_CAP[trophy.tier] ?? 1;
    let cellStd = 0;
    if (series.length >= 4) {
      const eRev = series[0].e;
      const pFirst = snaps[0].p;
      const useNonEarners = lastRate > 50;
      const lastSnapIdx = snaps.length - 1;
      const cells: Array<{ score: number; f: number; eTgt: number; di: number; si: number }> = [];
      for (let di = 0; di < DELTAS.length; di++) {
        for (let si = 0; si < SIGMAS.length; si++) {
          if (kSnap[lastSnapIdx][di][si] < KERNEL_MIN_MATURED) continue;
          let num = 0;
          let den = 0;
          for (const pt of series) {
            const pub = pt.e - eRev;
            if (pub < 1) continue;
            const k = kSnap[pt.snap][di][si];
            num += pub * k;
            den += k * k;
          }
          if (den <= 0) continue;
          const f = Math.min(fCap, num / den);
          let sse = 0;
          for (const pt of series) {
            const k = kSnap[pt.snap][di][si];
            let obs: number;
            let pred: number;
            if (useNonEarners) {
              const pPub = Math.max(snaps[pt.snap].p - pFirst, 0);
              obs = Math.max(pPub - (pt.e - eRev) + 0.5, 0.1);
              pred = Math.max(pPub - f * k + 0.5, 0.1);
            } else {
              obs = Math.max(pt.e - eRev + 0.5, 0.1);
              pred = Math.max(f * k + 0.5, 0.1);
            }
            const d = Math.log(obs) - Math.log(pred);
            sse += d * d;
          }
          let score = sse / series.length;
          if (DELTAS[di] > DELTA_PRIOR_AT) {
            const lp = Math.log(DELTAS[di] / DELTA_PRIOR_AT);
            score += W_DELTA_PRIOR * lp * lp;
          }
          cells.push({ score, f, eTgt: eRev + f * kTarget[di][si], di, si });
        }
      }
      if (cells.length > 0) {
        const minScore = Math.min(...cells.map((x) => x.score));
        const weights = cells.map((x) => Math.exp(-(x.score - minScore) / CELL_TEMP));
        const wSum = weights.reduce((s, w) => s + w, 0);
        const mCells = cells.map((x) => logit((100 * x.eTgt) / pTarget));
        let mMean = 0;
        let fMean = 0;
        for (let k = 0; k < cells.length; k++) {
          mMean += (weights[k] / wSum) * mCells[k];
          fMean += (weights[k] / wSum) * cells[k].f;
        }
        let vSum = 0;
        for (let k = 0; k < cells.length; k++) {
          vSum += (weights[k] / wSum) * (mCells[k] - mMean) * (mCells[k] - mMean);
        }
        cellStd = Math.sqrt(vSum);
        const bestCell = cells.reduce((a, b) => (a.score <= b.score ? a : b));
        if (fMean >= 0.01) {
          mKernel = mMean;
          opinions.push({ m: mKernel, w: W_KERNEL });
          residKernel = Math.sqrt(Math.max(minScore, 0));
          fitDelta = DELTAS[bestCell.di];
          fitSigma = SIGMAS[bestCell.si];
          fitF = bestCell.f;
        } else {
          // Fitted F is effectively zero: the trophy has not started
          // maturing in any cell's reading. Late-bloomer prior.
          const fPrior = F_PRIOR[trophy.tier] ?? 0.12;
          const diLate = DELTAS.indexOf(LATE_DELTA);
          const siLate = SIGMAS.indexOf(0.5);
          const eTgt = eRev + fPrior * kTarget[diLate][siLate];
          mKernel = logit((100 * eTgt) / pTarget);
          opinions.push({ m: mKernel, w: W_KERNEL });
          fitDelta = LATE_DELTA;
          fitSigma = 0.5;
          fitF = fPrior;
        }
      } else {
        // No fittable cell at all: late-bloomer prior.
        const fPrior = F_PRIOR[trophy.tier] ?? 0.12;
        const diLate = DELTAS.indexOf(LATE_DELTA);
        const siLate = SIGMAS.indexOf(0.5);
        const eTgt = eRev + fPrior * kTarget[diLate][siLate];
        mKernel = logit((100 * eTgt) / pTarget);
        opinions.push({ m: mKernel, w: W_KERNEL });
        fitDelta = LATE_DELTA;
        fitSigma = 0.5;
        fitF = fPrior;
      }
    }

    // Opinion 2: log-log earner trend, anchored at the last read, slope
    // shrunk toward the player-growth slope (a trophy's catch-up over the
    // player base decays). Earner growth is capped relative to player growth.
    {
      const use = series.filter((s) => s.e > 0);
      if (use.length >= 3) {
        const fit = ols(
          use.map((s) => logT(s.h)),
          use.map((s) => Math.log(s.e)),
        );
        if (fit !== null) {
          const bShrunk = betaP + SLOPE_SHRINK * (fit.b - betaP);
          const logETgt = Math.log(lastE) + bShrunk * (logT(targetH) - logT(lastH));
          let g = Math.exp(logETgt) / lastE;
          g = Math.min(Math.max(g, 1), GROWTH_CAP * growth);
          mEarn = logit((lastRate * g) / growth);
          opinions.push({ m: mEarn, w: W_EARN });
        }
      }
    }

    // Opinion 3: logit-rate trend over the last 24h (the house curve
    // baseline's shape), anchored at the last read with its slope halved:
    // a straight line fit to a concave curve over-extrapolates.
    {
      const recent = series.filter((s) => s.h >= atH - 24);
      if (recent.length >= 3) {
        const fit = ols(
          recent.map((s) => logT(s.h)),
          recent.map((s) => logit(s.r)),
        );
        if (fit !== null) {
          const base = logit(lastRate);
          const m = base + TREND_SHRINK * fit.b * (logT(targetH) - logT(lastH));
          mTrend = Math.min(Math.max(m, base - 3), base + 3);
          opinions.push({ m: mTrend, w: W_TREND });
        }
      }
    }

    // Blend; with no opinion at all (frozen history), fall back to persistence.
    let median: number;
    let disagree = 0;
    if (opinions.length === 0) {
      median = logit(lastRate);
    } else {
      const wsum = opinions.reduce((s, o) => s + o.w, 0);
      median = opinions.reduce((s, o) => s + o.w * o.m, 0) / wsum;
      const ms = opinions.map((o) => o.m);
      disagree = Math.max(...ms) - Math.min(...ms);
    }

    // Spread: random-walk horizon term, model disagreement, kernel fit
    // residual, the spread across plausible kernel cells (identification
    // uncertainty), refresh staleness, a bump when the history is frozen (no
    // trend information at all: a refresh is overdue), and a bump for
    // trophies whose earners are still mostly the launch-night baseline
    // (their maturation is genuinely unobserved, so the forecast is a prior).
    const maturity = series.length > 0 ? (lastE - series[0].e) / lastE : 0;
    const immatureBump = IMMATURE_SPREAD * Math.min(Math.max(1 - maturity, 0), 1);
    const frozenBump = snaps.length <= 2 ? 0.3 : 0;
    let spread = Math.sqrt(
      (SIGMA_RW * Math.sqrt(days)) ** 2 +
        (W_DISAGREE * disagree) ** 2 +
        Math.min(residKernel, 1) ** 2 +
        Math.min(cellStd, 1.5) ** 2 +
        SIGMA_REFRESH ** 2 +
        frozenBump ** 2 +
        immatureBump ** 2,
    );
    spread = Math.min(SPREAD_MAX, Math.max(SPREAD_MIN, spread));
    rows.push({ id: trophy.id, q: quantiles(median, spread) });
    if (DEBUG) {
      debugRows.push({
        id: trophy.id,
        tier: trophy.tier,
        name: trophy.name,
        lastRate,
        lastE,
        mKernel,
        mEarn,
        mTrend,
        median,
        spread,
        fitDelta,
        fitSigma,
        fitF,
        residKernel,
      });
    }
  }
  return {
    rows,
    debug: { pTarget, growth, betaP, scaleP, nSnaps: snaps.length, trophies: debugRows },
  };
}

/** The five quantiles around median logit m with spread s, as percents. */
function quantiles(m: number, s: number): number[] {
  const qs = Z.map((z) => round4(clampPct(sigmoidPct(m + z * s))));
  for (let j = 1; j < qs.length; j++) {
    if (qs[j] < qs[j - 1]) qs[j] = qs[j - 1];
  }
  return qs;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(
      `${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`,
    );
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const { rows, debug } = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
  if (DEBUG) {
    writeFileSync(join(OUTPUT_DIR, `${id}.debug.json`), `${JSON.stringify(debug, null, 1)}\n`);
  }
}
