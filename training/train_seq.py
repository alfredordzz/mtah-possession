"""Train the sequence model (bidirectional GRU) on the external validation games.

    python -m pip install "jax[cpu]" optax
    python training/train_seq.py VALIDATION_ROOT                                   # v10's network: model_v10_seq.npz
    python training/train_seq.py VALIDATION_ROOT --seed 1 --epochs 8 --out cache/seq_s1.npz

v11 averages three networks trained exactly like this, with seeds 0, 1 and 2: seed 0 with 12 epochs
(the v10 network) and seeds 1 and 2 with 8 epochs, as they were validated. training/build_v11_seq.py
packs the three into model_v11_seq.npz.

Input per frame: v9.1's features (dead-ball judge, team judge without and with the ball, ball seen),
read every 5th frame (5 Hz) per half. Standardised with the training mean and spread; columns that
never vary are ignored and values clipped to +/-6 spreads. Model: dense 64 + ReLU, two bidirectional
GRU layers of 64 units, softmax over H, A, D. Training: the hard variants of the 8 games, windows of
256 steps, class-balanced cross-entropy, x3 weight within 75 frames of a change of possession (as
v9.1), Adam 2e-3, 12 epochs and seed 0 by default. Only external data is used. Inference in the submission is plain
numpy (possession_v9_standalone.seq_emissions), so JAX is needed for training only.
"""

import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_final as TF  # noqa: E402

TF.M = TF.MODELS["v9"]
M9 = TF.M
STEP, WIN, EPOCHS, HID, SEED = 5, 256, 12, 64, 0


def feats(p):
    s = p["seen"].astype(np.float32)[:, None]
    return np.hstack([p["dall"], p["tn"], p["tb"] * s, s]).astype(np.float32)


def halves(p, X):
    gt = p["gt"]
    y = gt.astype(np.int64).copy()
    y[y == 3] = -100
    old = (M9.TRANSITION_WEIGHT, M9.TRANSITION_WINDOW)
    M9.TRANSITION_WEIGHT, M9.TRANSITION_WINDOW = 3.0, 75
    w = M9.transition_weights(gt, (0, 1)).astype(np.float32)
    M9.TRANSITION_WEIGHT, M9.TRANSITION_WINDOW = old
    return [(X[np.arange(a, b + 1, STEP)], y[np.arange(a, b + 1, STEP)], w[np.arange(a, b + 1, STEP)])
            for a, b in p["periods"]]


def stdz(x, mu, sd):
    z = np.clip((x - mu) / sd, -6.0, 6.0)
    z[:, sd < 1e-3] = 0.0
    return z.astype(np.float32)


def init(key, d):
    def dense(k, a, b):
        return jax.random.normal(k, (a, b)) * (1.0 / np.sqrt(a))
    ks = jax.random.split(key, 12)
    P = {"Win": dense(ks[0], d, HID), "bin": jnp.zeros(HID)}
    i = 1
    for layer, din in ((0, HID), (1, 2 * HID)):
        for dr in ("f", "b"):
            P["W%d%s" % (layer, dr)] = dense(ks[i], din, 3 * HID)
            P["U%d%s" % (layer, dr)] = dense(ks[i + 1], HID, 3 * HID) * 0.5
            P["b%d%s" % (layer, dr)] = jnp.zeros(3 * HID)
            i += 2
    P["Wout"] = dense(ks[i], 2 * HID, 3)
    P["bout"] = jnp.zeros(3)
    return P


def gru(x, W, U, b):
    xw = x @ W + b

    def step(h, xt):
        z_r = jax.nn.sigmoid(xt[:, :2 * HID] + h @ U[:, :2 * HID])
        z, r = z_r[:, :HID], z_r[:, HID:]
        hh = jnp.tanh(xt[:, 2 * HID:] + (r * h) @ U[:, 2 * HID:])
        h = (1 - z) * h + z * hh
        return h, h
    _, hs = jax.lax.scan(step, jnp.zeros((x.shape[1], HID)), xw)
    return hs


def forward(P, x, key=None, train=False):
    h = jax.nn.relu(x @ P["Win"] + P["bin"])
    if train:
        h = h * jax.random.bernoulli(key, 0.9, h.shape) / 0.9
    h = jnp.swapaxes(h, 0, 1)
    for layer in (0, 1):
        f = gru(h, P["W%df" % layer], P["U%df" % layer], P["b%df" % layer])
        b = gru(h[::-1], P["W%db" % layer], P["U%db" % layer], P["b%db" % layer])[::-1]
        h = jnp.concatenate([f, b], -1)
    return jnp.swapaxes(h, 0, 1) @ P["Wout"] + P["bout"]


def train(seqs, seed=SEED, epochs=EPOCHS):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    allx = np.vstack([s[0] for s in seqs])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    ally = np.concatenate([s[1] for s in seqs])
    cnt = np.bincount(ally[ally >= 0], minlength=3).astype(float)
    cw = jnp.array(cnt.sum() / (3 * cnt), dtype=jnp.float32)
    P = init(key, allx.shape[1])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(2e-3, weight_decay=1e-5))
    st = opt.init(P)

    def loss_fn(P, xb, yb, wb, k):
        lp = jax.nn.log_softmax(forward(P, xb, k, True))
        valid = yb >= 0
        yc = jnp.where(valid, yb, 0)
        nll = -jnp.take_along_axis(lp, yc[..., None], -1)[..., 0] * cw[yc]
        ww = wb * valid
        return (nll * ww).sum() / ww.sum()

    @jax.jit
    def step(P, st, xb, yb, wb, k):
        loss, g = jax.value_and_grad(loss_fn)(P, xb, yb, wb, k)
        up, st = opt.update(g, st, P)
        return optax.apply_updates(P, up), st, loss

    windows = [(i, s) for i, (x, _, _) in enumerate(seqs) for s in range(0, max(1, len(x) - WIN), WIN // 2)]
    for ep in range(epochs):
        rng.shuffle(windows)
        tot, nb = 0.0, 0
        for b in range(0, len(windows) - 15, 16):
            xs, ys, ws = [], [], []
            for i, s in windows[b:b + 16]:
                x, y, w = seqs[i]
                s = min(s + int(rng.integers(0, WIN // 2)), max(0, len(x) - WIN))
                xs.append(stdz(x[s:s + WIN], mu, sd))
                ys.append(y[s:s + WIN])
                ws.append(w[s:s + WIN])
            key, k = jax.random.split(key)
            P, st, loss = step(P, st, jnp.array(np.stack(xs)), jnp.array(np.stack(ys)), jnp.array(np.stack(ws)), k)
            tot += float(loss)
            nb += 1
        print("epoch {} loss {:.4f}".format(ep + 1, tot / max(nb, 1)), flush=True)
    return P, mu, sd


def main():
    import argparse
    ap = argparse.ArgumentParser(description="train the sequence model on the external games")
    ap.add_argument("root", help="validation root with the external games (see training/README)")
    ap.add_argument("out_pos", nargs="?", help=argparse.SUPPRESS)
    ap.add_argument("--out", help="output .npz (default model_v10_seq.npz in the repository root)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    root = a.root
    out = a.out or a.out_pos or os.path.join(os.path.dirname(HERE), "model_v10_seq.npz")
    paths = TF.game_paths(root)
    seqs = []
    for v in TF.TRAIN:
        for name, path in paths[v]:
            p = TF.prepare(TF.load_npz(path))
            seqs += halves(p, feats(p))
            print("prepared", v, name, flush=True)
    P, mu, sd = train(seqs, a.seed, a.epochs)
    np.savez(out, mu=mu, sd=sd, **{k: np.asarray(v) for k, v in P.items()})
    print("wrote", out)


if __name__ == "__main__":
    main()
