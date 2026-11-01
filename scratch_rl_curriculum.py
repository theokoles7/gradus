"""# RL-Controlled Stochastic Curriculum Harness

Trains <network> on <dataset> using all training examples, while a lightweight policy-gradient
controller chooses how strongly each epoch should deviate from ordinary random shuffling.

The curriculum score defines a ranked sampling preference. The controller chooses alpha in [0, 1]:
    alpha = 0  -> exact uniform/random sampling
    alpha > 0  -> increasingly strong score-biased sampling

Sampling probabilities are a convex mixture:
    p_t(i) = (1 - alpha_t) / N + alpha_t * q_t(i)
where q_t is a softmax distribution over standardized curriculum scores. Sampling is without
replacement, so every epoch still sees every training sample exactly once; only ORDER changes.
This keeps DSI/compute identical to ordinary shuffled training.

The RL controller is intentionally small: a contextual REINFORCE policy over a discrete alpha grid.
It acts once per epoch using optimization-state features from the previous epoch. Reward is the
subsequent validation-loss improvement, optionally penalized for curriculum intervention.

## Policies
    --controller random      ordinary shuffled training (alpha = 0)
    --controller fixed       fixed curriculum strength (--fixed-alpha)
    --controller random-alpha random alpha from the action grid each epoch (non-learning control)
    --controller reinforce   learned contextual policy-gradient controller

## Curriculum score
    --score loss | el2n | aum
    --direction easy | hard
    --temperature T          softness of q_t; larger T approaches uniform
    --score-alpha A          EMA weight on newest per-sample score observation

## RL
    --actions 0,0.25,0.5,0.75,1.0
    --policy-lr 0.01
    --entropy-beta 0.01
    --intervention-cost 0.0
    --warmup 10              epochs forced to random before the controller acts

## Usage
    python scratch_rl_curriculum.py --controller random --seed 1
    python scratch_rl_curriculum.py --controller fixed --fixed-alpha 0.5 --score loss --direction hard --seed 1
    python scratch_rl_curriculum.py --controller random-alpha --score loss --direction hard --seed 1
    python scratch_rl_curriculum.py --controller reinforce --score loss --direction hard --seed 1
    python scratch_rl_curriculum.py --summarize

## Outputs (under --output/<dataset>/<network>/)
    runs/<run_id>.json      config, per-epoch curves, actions, probabilities, rewards
    results.csv             one row per completed run
    summary.txt             seed-averaged results and paired deltas vs random shuffle
"""

from argparse import ArgumentParser, Namespace
from hashlib import md5
from json import dump, dumps
from pathlib import Path
from time import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

POLICY_KEYS: List[str] = [
    "network", "dataset", "epochs", "batch_size", "controller", "score", "direction",
    "temperature", "score_alpha", "warmup", "actions", "fixed_alpha", "policy_lr",
    "entropy_beta", "intervention_cost", "reward_scale"
]


def parse_arguments() -> Namespace:
    p = ArgumentParser(description="RL-controlled stochastic curriculum harness.")
    p.add_argument("--network", type=str, default="resnet-18")
    p.add_argument("--dataset", type=str, default="cifar-100")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--root", type=str, default=".cache/data")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output", type=str, default="analysis/rl_curriculum")

    p.add_argument("--controller", type=str, default="reinforce",
                   choices=["random", "fixed", "random-alpha", "reinforce"])
    p.add_argument("--score", type=str, default="loss", choices=["loss", "el2n", "aum"])
    p.add_argument("--direction", type=str, default="hard", choices=["easy", "hard"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--score-alpha", type=float, default=0.5,
                   help="EMA weight on newest per-sample score observation.")
    p.add_argument("--warmup", type=int, default=10,
                   help="Initial epochs forced to alpha=0 while score/state estimates accumulate.")

    p.add_argument("--actions", type=str, default="0,0.25,0.5,0.75,1.0",
                   help="Comma-separated alpha actions available to the controller.")
    p.add_argument("--fixed-alpha", type=float, default=0.5)
    p.add_argument("--policy-lr", type=float, default=0.01)
    p.add_argument("--entropy-beta", type=float, default=0.01)
    p.add_argument("--intervention-cost", type=float, default=0.0,
                   help="Reward penalty multiplied by alpha; tests whether departures from random earn their cost.")
    p.add_argument("--reward-scale", type=float, default=100.0,
                   help="Scale validation-loss improvement before policy update.")

    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()

    actions = [float(x) for x in args.actions.split(",")]
    if not actions or any(a < 0 or a > 1 for a in actions):
        p.error("--actions must contain values in [0, 1].")
    if 0.0 not in actions:
        p.error("--actions must include 0 so random shuffling is an available/null action.")
    if args.temperature <= 0:
        p.error("--temperature must be > 0.")
    if not 0 <= args.fixed_alpha <= 1:
        p.error("--fixed-alpha must be in [0, 1].")
    return args


def action_values(args: Namespace) -> np.ndarray:
    return np.asarray([float(x) for x in args.actions.split(",")], dtype=np.float64)


class ScoreState:
    """EMA per-sample curriculum scores observed during ordinary training passes."""

    def __init__(self, n: int, alpha: float):
        self.alpha = alpha
        self.values = {k: np.full(n, np.nan, dtype=np.float64) for k in ("loss", "el2n", "aum")}

    def update(self, idx: np.ndarray, values: Dict[str, np.ndarray]) -> None:
        for key, value in values.items():
            prev = self.values[key][idx]
            self.values[key][idx] = np.where(np.isnan(prev), value, self.alpha * value + (1 - self.alpha) * prev)

    def get(self, key: str) -> np.ndarray:
        return self.values[key]


def curriculum_distribution(scores: np.ndarray, direction: str, temperature: float) -> np.ndarray:
    """Soft curriculum distribution q(i); high probability means earlier in the epoch."""
    s = scores.astype(np.float64).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.full(len(s), 1.0 / len(s))
    fill = np.nanmedian(s[finite])
    s[~finite] = fill
    z = (s - s.mean()) / (s.std() + 1e-12)

    # loss/el2n: larger = harder. AUM: smaller = harder.
    if direction == "hard":
        oriented = -z if False else z
    else:
        oriented = -z
    # AUM has the opposite semantic orientation.
    return oriented


def score_logits(scores: np.ndarray, score_name: str, direction: str, temperature: float) -> np.ndarray:
    s = scores.astype(np.float64).copy()
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros(len(s), dtype=np.float64)
    s[~finite] = np.nanmedian(s[finite])
    z = (s - s.mean()) / (s.std() + 1e-12)
    hard = z if score_name in ("loss", "el2n") else -z
    oriented = hard if direction == "hard" else -hard
    return oriented / temperature


def mixed_order(scores: np.ndarray, score_name: str, direction: str, temperature: float,
                alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Weighted random permutation. alpha=0 is exactly a uniform random permutation."""
    n = len(scores)
    if alpha <= 0:
        return rng.permutation(n)

    logits = score_logits(scores, score_name, direction, temperature)
    logits -= logits.max()
    q = np.exp(logits)
    q /= q.sum()
    p = (1.0 - alpha) / n + alpha * q

    # Gumbel-top-k produces a weighted permutation without replacement.
    keys = np.log(p + 1e-300) + rng.gumbel(size=n)
    return np.argsort(-keys, kind="stable")


class ReinforceController:
    """Tiny contextual softmax policy with online REINFORCE updates."""

    def __init__(self, n_features: int, actions: np.ndarray, lr: float, entropy_beta: float,
                 seed: int):
        self.actions = actions
        self.lr = lr
        self.entropy_beta = entropy_beta
        self.rng = np.random.default_rng(seed + 50_000)
        self.W = np.zeros((len(actions), n_features + 1), dtype=np.float64)  # + bias
        self.baseline = 0.0
        self.baseline_beta = 0.1

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max()
        p = np.exp(z)
        return p / p.sum()

    def choose(self, state: np.ndarray):
        x = np.r_[state, 1.0]
        probs = self._softmax(self.W @ x)
        a_idx = int(self.rng.choice(len(self.actions), p=probs))
        return a_idx, float(self.actions[a_idx]), probs, x

    def update(self, x: np.ndarray, a_idx: int, probs: np.ndarray, reward: float) -> float:
        advantage = reward - self.baseline
        self.baseline = (1 - self.baseline_beta) * self.baseline + self.baseline_beta * reward

        onehot = np.zeros_like(probs); onehot[a_idx] = 1.0
        grad_logp = (onehot - probs)[:, None] * x[None, :]

        # Entropy gradient for softmax logits, then chain through linear policy.
        logp = np.log(probs + 1e-12)
        entropy = -float(np.sum(probs * logp))
        g_logits = -probs * (logp + entropy)
        grad_entropy = g_logits[:, None] * x[None, :]

        self.W += self.lr * (advantage * grad_logp + self.entropy_beta * grad_entropy)
        return advantage


def make_state(history: List[Dict]) -> np.ndarray:
    """Compact optimization state, standardized by bounded/simple transforms."""
    if not history:
        return np.zeros(7, dtype=np.float64)
    h = history[-1]
    prev = history[-2] if len(history) > 1 else h
    return np.asarray([
        h["train_loss"],
        h["val_loss"],
        h["val_loss"] - h["train_loss"],
        h["train_loss"] - prev["train_loss"],
        h["val_loss"] - prev["val_loss"],
        h["val_acc"] - prev["val_acc"],
        np.log1p(h["grad_norm"]),
    ], dtype=np.float64)


def run(args: Namespace, run_id: str, run_dir: Path) -> Dict:
    from torch import Generator, no_grad
    from torch.nn.functional import cross_entropy, one_hot, softmax
    from torch.optim import SGD
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader, Dataset, Sampler
    from tqdm import tqdm

    from gradus.registration import DATASET_REGISTRY, NETWORK_REGISTRY
    from gradus.utilities import determine_device, set_seed

    class Indexed(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i):
            x, y = self.data[i]
            return x, y, i

    class OrderSampler(Sampler):
        def __init__(self, order): self.order = np.asarray(order)
        def __len__(self): return len(self.order)
        def __iter__(self): return iter(self.order.tolist())

    set_seed(args.seed)
    device = determine_device(args.device)
    rng = np.random.default_rng(args.seed)

    dataset = DATASET_REGISTRY.load_dataset(
        dataset_id=args.dataset, epochs=args.epochs, seed=args.seed, root=args.root,
        batch_size=args.batch_size, max_workers=args.workers
    )
    train = dataset.train_data
    N, E = len(train), args.epochs

    sampler = OrderSampler(np.arange(N))
    loader = DataLoader(Indexed(train), batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

    network = NETWORK_REGISTRY.load_network(
        network_id=args.network, input_shape=dataset.input_shape, num_classes=dataset.num_classes
    ).to(device)
    optimizer = SGD(network.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=E)

    scores = ScoreState(N, args.score_alpha)
    actions = action_values(args)
    controller = ReinforceController(7, actions, args.policy_lr, args.entropy_beta, args.seed)

    history: List[Dict] = []
    started = time()

    for epoch in range(1, E + 1):
        state = make_state(history)
        probs = np.zeros(len(actions), dtype=np.float64)
        action_idx = int(np.where(actions == 0.0)[0][0])

        if epoch <= args.warmup or args.controller == "random":
            alpha = 0.0
            probs[action_idx] = 1.0
            policy_x = None
        elif args.controller == "fixed":
            alpha = args.fixed_alpha
            policy_x = None
        elif args.controller == "random-alpha":
            action_idx = int(rng.integers(len(actions)))
            alpha = float(actions[action_idx])
            probs[:] = 1.0 / len(actions)
            policy_x = None
        else:
            action_idx, alpha, probs, policy_x = controller.choose(state)

        # Use scores accumulated only through the previous epoch to choose the current order.
        sampler.order = mixed_order(scores.get(args.score), args.score, args.direction,
                                    args.temperature, alpha, rng)

        network.train()
        correct = seen = 0
        train_loss_sum = 0.0
        grad_norm_sum = 0.0
        batches = 0

        for x, y, idx in tqdm(loader, desc=f"[{run_id}] epoch {epoch}/{E}", leave=False):
            x, y = x.to(device), y.to(device)
            logits = network(x)
            per_sample = cross_entropy(logits, y, reduction="none")
            loss = per_sample.mean()

            optimizer.zero_grad()
            loss.backward()

            sq = 0.0
            for param in network.parameters():
                if param.grad is not None:
                    sq += float(param.grad.detach().norm(2).item() ** 2)
            grad_norm = sq ** 0.5
            optimizer.step()

            with no_grad():
                z = logits.detach()
                onehot = one_hot(y, num_classes=z.shape[1]).float()
                margin = z.gather(1, y.unsqueeze(1)).squeeze(1) - \
                         z.masked_fill(onehot.bool(), float("-inf")).max(dim=1).values
                values = {
                    "loss": per_sample.detach().cpu().numpy(),
                    "el2n": (softmax(z, dim=1) - onehot).norm(dim=1).cpu().numpy(),
                    "aum": margin.cpu().numpy(),
                }
                scores.update(idx.numpy(), values)
                correct += z.argmax(dim=1).eq(y).sum().item()
                seen += y.size(0)
                train_loss_sum += float(per_sample.sum().item())
                grad_norm_sum += grad_norm
                batches += 1

        scheduler.step()

        network.eval()
        hits = total = 0
        val_loss_sum = 0.0
        with no_grad():
            for x, y in dataset.test_loader:
                x, y = x.to(device), y.to(device)
                z = network(x)
                val_loss_sum += float(cross_entropy(z, y, reduction="sum").item())
                hits += z.argmax(dim=1).eq(y).sum().item()
                total += y.size(0)

        train_loss = train_loss_sum / max(1, seen)
        val_loss = val_loss_sum / max(1, total)
        val_acc = hits / max(1, total)

        # Reward THIS epoch's action by improvement from the pre-action validation state.
        reward = advantage = None
        if history:
            raw_improvement = history[-1]["val_loss"] - val_loss
            reward = args.reward_scale * raw_improvement - args.intervention_cost * alpha
            if args.controller == "reinforce" and policy_x is not None:
                advantage = controller.update(policy_x, action_idx, probs, reward)

        row = {
            "epoch": epoch,
            "alpha": float(alpha),
            "action_idx": int(action_idx),
            "action_probs": probs.tolist(),
            "train_loss": train_loss,
            "train_acc": correct / max(1, seen),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "generalization_gap": val_loss - train_loss,
            "grad_norm": grad_norm_sum / max(1, batches),
            "reward": None if reward is None else float(reward),
            "advantage": None if advantage is None else float(advantage),
        }
        history.append(row)

        print(f"[{run_id}] epoch {epoch:3}/{E}: alpha={alpha:.2f}; "
              f"train loss={train_loss:.4f}; val loss={val_loss:.4f}; val acc={val_acc:.4f}")

    val = [h["val_acc"] for h in history]
    alphas = [h["alpha"] for h in history[args.warmup:]]
    rewards = [h["reward"] for h in history if h["reward"] is not None]
    result = {
        "run_id": run_id,
        "seed": args.seed,
        **{k: getattr(args, k) for k in POLICY_KEYS},
        "final_acc": val[-1],
        "best_acc": max(val),
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0,
        "random_action_frac": float(np.mean(np.asarray(alphas) == 0)) if alphas else 1.0,
        "mean_reward": float(np.mean(rewards)) if rewards else np.nan,
        "wall_minutes": (time() - started) / 60,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / f"{run_id}.json", "w") as f:
        dump({**result, "history": history,
              "policy_weights": controller.W.tolist() if args.controller == "reinforce" else None}, f, indent=2)
    return result


def load_results(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], dtype={"run_id": str})


def append_result(path: Path, result: Dict) -> None:
    row = pd.DataFrame([result])
    if not path.exists():
        row.to_csv(path, index=False); return
    header = list(pd.read_csv(path, nrows=0).columns)
    if set(header) == set(row.columns):
        row[header].to_csv(path, mode="a", header=False, index=False)
    else:
        pd.concat([load_results(path), row], ignore_index=True).to_csv(path, index=False)


def run_config(args: Namespace) -> Dict:
    return {k: getattr(args, k) for k in POLICY_KEYS} | {"seed": args.seed}


def summarize(out_dir: Path) -> str:
    df = load_results(out_dir / "results.csv")

    # Same-seed random-shuffle comparison. Random controller's curriculum-specific knobs are ignored.
    random_rows = df[df["controller"] == "random"].groupby("seed")["final_acc"].mean()
    df["paired"] = [row.final_acc - random_rows.get(row.seed, np.nan) for _, row in df.iterrows()]

    keys = [k for k in POLICY_KEYS if k not in ("network", "dataset", "batch_size", "policy_lr",
                                                  "entropy_beta", "intervention_cost", "reward_scale")]
    agg = df.groupby(keys, dropna=False).agg(
        n=("seed", "count"), final_acc=("final_acc", "mean"), final_std=("final_acc", "std"),
        best_acc=("best_acc", "mean"), paired=("paired", "mean"), paired_sd=("paired", "std"),
        n_pairs=("paired", "count"), mean_alpha=("mean_alpha", "mean"),
        random_action_frac=("random_action_frac", "mean"), minutes=("wall_minutes", "mean")
    ).reset_index()
    agg["paired_se"] = agg["paired_sd"] / np.sqrt(agg["n_pairs"])

    cols = ["controller", "score", "direction", "temperature", "warmup", "actions", "fixed_alpha",
            "n", "final_acc", "final_std", "best_acc", "paired", "paired_se", "n_pairs",
            "mean_alpha", "random_action_frac", "minutes"]
    table = agg[cols].sort_values(["paired", "final_acc"], ascending=False)
    for c in ("final_acc", "final_std", "best_acc", "paired", "paired_se", "mean_alpha", "random_action_frac"):
        table[c] = (100 * table[c]).round(2)
    table["minutes"] = table["minutes"].round(1)

    text = "\n".join([
        f"{len(df)} runs; accuracy/deltas/alpha frequencies shown in percentage points.",
        "paired: same-seed final-accuracy difference from ordinary random shuffling.",
        "mean_alpha: average curriculum pressure after warmup; random_action_frac: fraction of post-warmup epochs choosing alpha=0.",
        "", table.to_string(index=False), ""
    ])
    (out_dir / "summary.txt").write_text(text)
    return text


def main() -> None:
    args = parse_arguments()
    out_dir = Path(args.output) / args.dataset / args.network
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    if args.summarize:
        print(summarize(out_dir)); return

    config = run_config(args)
    run_id = md5(dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    results = out_dir / "results.csv"

    if results.exists() and run_id in set(load_results(results)["run_id"]):
        print(f"[{run_id}] Already recorded; skipping. ({config})"); return

    print(f"[{run_id}] {config}")
    result = run(args, run_id, out_dir / "runs")
    append_result(results, result)
    print(f"[{run_id}] final acc={result['final_acc']:.4f}; best={result['best_acc']:.4f}; "
          f"mean alpha={result['mean_alpha']:.3f}")


if __name__ == "__main__":
    main()
