"""Possession model v2: logistic evidence + Viterbi (a hybrid HMM) + boundary snapping.

numpy only. Every function takes plain arrays in metres / m/s with one row per video frame, so
the same code runs on the competition match and on the validation games.

Pipeline
    features_dead(...)   per-frame signals that the ball is out of play
    features_team(...)   per-frame signals of which team has the ball
    Logistic.fit/predict two logistic regressions -> P(D), P(H | alive)
    decode(...)          Viterbi over H/A/D with switch and dead costs
    snap(...)            move each H<->A change to the frame the new team reaches the ball

Design rule: ball *visibility* is never used as evidence. In the validation games the ball
vanishes mostly when it is out of play; in the competition tracking it vanishes in play. Any
model that learned "missing ball means dead ball" would transfer badly, so ball-based features
are only computed where the ball is seen, and a separate player-only model covers the rest.
"""

import warnings

import numpy as np

H, A, D, X = 0, 1, 2, 3
FPS = 25


# -- helpers -----------------------------------------------------------------------------------

def nanmean_window(x, before, after):
    """Mean of x over [t - before, t + after], ignoring NaN. x is 1-D or 2-D (rows = time)."""
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    n = len(x)
    valid = ~np.isnan(x)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.where(valid, x, 0.0), axis=0)])
    cc = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(valid, axis=0)])
    i = np.arange(n)
    lo = np.clip(i - before, 0, n)
    hi = np.clip(i + after + 1, 0, n)
    count = cc[hi] - cc[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, (cs[hi] - cs[lo]) / np.maximum(count, 1), np.nan)
    return out[:, 0] if squeeze else out


def robust_z(x, mask):
    """Standardise by the match's own median and IQR inside mask (per-match normalisation)."""
    ref = x[mask & np.isfinite(x)]
    if len(ref) < 10:
        return np.zeros_like(x)
    med = np.median(ref)
    iqr = np.subtract(*np.percentile(ref, [75, 25])) or 1.0
    return (x - med) / iqr


def in_periods(n, periods):
    mask = np.zeros(n, dtype=bool)
    for first, last in periods:
        mask[first:last + 1] = True
    return mask


def ball_kinematics(ball):
    """Ball speed (m/s) from positions over +/-2 frames; NaN where not seen."""
    n = len(ball)
    speed = np.full(n, np.nan)
    k = 2
    if n > 2 * k:
        d = np.hypot(ball[2 * k:, 0] - ball[:-2 * k, 0], ball[2 * k:, 1] - ball[:-2 * k, 1])
        speed[k:-k] = d / (2 * k) * FPS
    return speed


def attack_signs(pos, slot_team, periods, length):
    """+1 if a team attacks towards x = length in that period, else -1. From mean positions."""
    signs = []
    for first, last in periods:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hx = np.nanmean(pos[first:last + 1][:, slot_team == H, 0])
            ax = np.nanmean(pos[first:last + 1][:, slot_team == A, 0])
        # the team whose players sit nearer x = 0 on average defends that goal
        s_home = 1.0 if hx < ax else -1.0
        signs.append((s_home, -s_home))
    return signs


# -- features ----------------------------------------------------------------------------------

def common(ball, pos, speed, slot_team, periods, pitch):
    """Arrays shared by both feature sets."""
    n = len(ball)
    length, width = pitch
    graded = in_periods(n, periods)
    seen = ~np.isnan(ball[:, 0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med_speed = np.nanmedian(speed, axis=1)
        mean_speed = np.nanmean(speed, axis=1)
        sorted_speed = np.sort(np.where(np.isnan(speed), -np.inf, speed), axis=1)[:, ::-1]
        top3 = np.where(np.isfinite(sorted_speed[:, :3]), sorted_speed[:, :3], np.nan)
        top3_speed = np.nanmean(top3, axis=1)
        slow_share = np.nanmean(np.where(np.isnan(speed), np.nan, speed < 1.0), axis=1)

    diff = pos - ball[:, None, :]
    dist = np.hypot(diff[..., 0], diff[..., 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, slot_team == H].min(axis=1)
    d_away = dist[:, slot_team == A].min(axis=1)

    bspeed = ball_kinematics(ball)
    signs = attack_signs(pos, slot_team, periods, length)
    sign_home = np.zeros(n)
    for (first, last), (sh, _) in zip(periods, signs):
        sign_home[first:last + 1] = sh

    return dict(n=n, graded=graded, seen=seen, _ball=ball, med_speed=med_speed, mean_speed=mean_speed,
                top3_speed=top3_speed, slow_share=slow_share, d_home=d_home, d_away=d_away,
                bspeed=bspeed, sign_home=sign_home, length=length, width=width)


def features_dead(c):
    """(player_only (n, k1), with_ball (n, k2)) features for the dead-ball model."""
    g = c["graded"]
    cols = []
    for name in ("med_speed", "mean_speed", "top3_speed"):
        z = robust_z(c[name], g)
        for before, after in ((12, 12), (50, 50), (125, 0), (0, 125)):
            cols.append(nanmean_window(z, before, after))
    slow = c["slow_share"]
    for before, after in ((25, 25), (75, 75)):
        cols.append(nanmean_window(slow, before, after))
    # acceleration of the group: speed now vs 2 s ago (a restart is a sudden burst)
    m = robust_z(c["med_speed"], g)
    cols.append(nanmean_window(m, 0, 25) - nanmean_window(m, 50, -25))
    players = np.nan_to_num(np.stack(cols, axis=1))

    length, width = c["length"], c["width"]
    bx, by = c["_ball"][:, 0], c["_ball"][:, 1]
    with np.errstate(invalid="ignore"):
        outside = np.maximum.reduce([-bx, bx - length, -by, by - width])
        out = (outside > 0.5).astype(float)
        edge = np.clip(np.minimum.reduce([bx, length - bx, by, width - by]), 0, 15)
    bs = np.log(np.clip(c["bspeed"], 0.1, 40))
    still_1s = nanmean_window(np.where(np.isnan(c["bspeed"]), np.nan, c["bspeed"] < 1.0), 25, 25)
    nearest = np.minimum(c["d_home"], c["d_away"])
    ball_cols = [out, nanmean_window(out, 25, 25), edge, bs, nanmean_window(bs, 12, 12),
                 still_1s, np.log(np.clip(nearest, 0.2, 40))]
    with_ball = np.nan_to_num(np.hstack([players, np.stack(ball_cols, axis=1)]))
    return players, with_ball


def features_team(c, pos, slot_team, ball):
    """(no_ball (n, k1), with_ball (n, k2)) features, oriented so that positive favours H."""
    n = c["n"]
    length = c["length"]
    sh = c["sign_home"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hx = pos[:, slot_team == H, 0]
        ax = pos[:, slot_team == A, 0]
        hy = pos[:, slot_team == H, 1]
        ay = pos[:, slot_team == A, 1]
        # how far each team has advanced towards the opponent goal (mean and last line)
        h_adv = np.nanmean(hx, axis=1) * sh
        a_adv = np.nanmean(ax, axis=1) * -sh
        h_line = np.nanpercentile(hx * sh[:, None], 15, axis=1)
        a_line = np.nanpercentile(ax * -sh[:, None], 15, axis=1)
        h_spread = np.nanstd(hx, axis=1) + np.nanstd(hy, axis=1)
        a_spread = np.nanstd(ax, axis=1) + np.nanstd(ay, axis=1)
    h_vel = np.gradient(nanmean_window(h_adv, 12, 12)) * FPS
    a_vel = np.gradient(nanmean_window(a_adv, 12, 12)) * FPS

    raw = [h_adv - a_adv, h_line - a_line, h_spread - a_spread, h_vel - a_vel]
    cols = []
    for r in raw:
        for before, after in ((25, 25), (100, 100)):
            cols.append(nanmean_window(r, before, after))
    no_ball = np.nan_to_num(np.stack(cols, axis=1))

    with np.errstate(invalid="ignore"):
        adv = np.clip(c["d_away"] - c["d_home"], -8, 8)
    adv = np.where(np.isfinite(adv) & c["seen"], adv, np.nan)
    near = np.minimum(c["d_home"], c["d_away"])
    close = np.where(c["seen"], np.clip((4 - near) / 3, 0, 1), np.nan)
    ball_or = np.where(c["seen"], ball[:, 0] * sh - (length / 2) * sh, np.nan)  # >0: H's attacking half
    bcols = [adv, nanmean_window(adv, 6, 6), nanmean_window(adv, 25, 25),
             adv * close, nanmean_window(adv * close, 12, 12), ball_or]
    with_ball = np.nan_to_num(np.hstack([no_ball, np.stack(bcols, axis=1)]))
    return no_ball, with_ball


# -- logistic regression -----------------------------------------------------------------------

class Logistic:
    """L2-regularised logistic regression fitted by Newton's method. Balanced class weights."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, Xm, y, sample_weight=None):
        self.mu = Xm.mean(axis=0)
        self.sd = Xm.std(axis=0) + 1e-9
        Z = np.hstack([np.ones((len(Xm), 1)), (Xm - self.mu) / self.sd])
        w = np.ones(len(y)) if sample_weight is None else sample_weight.astype(float)
        pos = y == 1
        w = w * np.where(pos, 0.5 / max(pos.mean(), 1e-9), 0.5 / max((~pos).mean(), 1e-9))
        beta = np.zeros(Z.shape[1])
        reg = self.l2 * np.eye(Z.shape[1])
        reg[0, 0] = 0
        for _ in range(30):
            p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
            grad = Z.T @ (w * (p - y)) + reg @ beta
            hess = (Z * (w * p * (1 - p))[:, None]).T @ Z + reg
            step = np.linalg.solve(hess, grad)
            beta -= step
            if np.abs(step).max() < 1e-6:
                break
        self.beta = beta
        return self

    def logit(self, Xm):
        Z = np.hstack([np.ones((len(Xm), 1)), (Xm - self.mu) / self.sd])
        return Z @ self.beta

    def to_dict(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.mu, m.sd, m.beta = np.array(d["mu"]), np.array(d["sd"]), np.array(d["beta"])
        return m


# -- decoding ----------------------------------------------------------------------------------

def viterbi(emis, switch_cost, dead_cost):
    """emis: (n, 3) log-scores for H, A, D. Returns the best path with change penalties."""
    n = len(emis)
    trans = np.array([[0.0, -switch_cost, -dead_cost],
                      [-switch_cost, 0.0, -dead_cost],
                      [-dead_cost, -dead_cost, 0.0]])
    back = np.zeros((n, 3), dtype=np.int8)
    score = emis[0].copy()
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emis[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def emissions(logit_dead, logit_home, temp_dead=1.0, temp_team=1.0):
    """Log P(H), log P(A), log P(D) from the two logits, each scaled by a temperature."""
    ld = logit_dead * temp_dead
    lt = logit_home * temp_team
    log_d = -np.logaddexp(0, -ld)
    log_alive = -np.logaddexp(0, ld)
    log_h = -np.logaddexp(0, -lt)
    log_a = -np.logaddexp(0, lt)
    return np.stack([log_alive + log_h, log_alive + log_a, log_d], axis=1)


def snap(codes, c, window=25, reach_m=1.5):
    """Move each H<->A change to the first frame the new team is within reach of the ball."""
    out = codes.copy()
    n = len(codes)
    change = np.flatnonzero((codes[1:] != codes[:-1]) & (codes[1:] < D) & (codes[:-1] < D)) + 1
    for t in change:
        new = codes[t]
        d_new = c["d_home"] if new == H else c["d_away"]
        d_old = c["d_away"] if new == H else c["d_home"]
        lo, hi = max(0, t - window), min(n, t + window)
        ok = c["seen"][lo:hi] & (d_new[lo:hi] < reach_m) & (d_new[lo:hi] < d_old[lo:hi])
        hits = np.flatnonzero(ok)
        if not len(hits):
            continue
        target = lo + hits[0]
        if target < t and np.all(out[target:t] == codes[t - 1]):
            out[target:t] = new
        elif target > t and np.all(out[t:target] == new):
            out[t:target] = codes[t - 1]
    return out
