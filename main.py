# =========================
# GLB-OMD vs GLM-UCB vs GLOC
# Logistic & Poisson Bandits (S=3,5), 10 trials
# Regret curves + Runtime bars (log10)
# =========================
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import matplotlib.pyplot as plt

# -------------------------
# Link functions (mean, derivative, sampling)
# -------------------------
class Link:
    def __init__(self, kind="logistic"):
        assert kind in ("logistic", "poisson")
        self.kind = kind

    def mu(self, z):
        if self.kind == "logistic":
            return 1.0 / (1.0 + np.exp(-z))
        else:  # poisson
            return np.exp(z)

    def mu_prime(self, z):
        if self.kind == "logistic":
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        else:  # poisson
            return np.exp(z)

    def sample_reward(self, z, rng):
        if self.kind == "logistic":
            p = 1.0 / (1.0 + np.exp(-z))
            return 1.0 if rng.random() < p else 0.0
        else:  # poisson
            rate = np.exp(z)
            return float(rng.poisson(rate))

# -------------------------
# Utilities
# -------------------------
def l2_project(theta, S):
    n = np.linalg.norm(theta)
    return theta if n <= S or n == 0 else theta * (S / n)

def beta_radius(t, d, S, delta=0.05, c=2.0):
    # 실용적 κ-free 반경 (살짝 보수적)
    return c * S * np.sqrt(d * (1.0 + np.log((t + 1.0) / delta)))

def make_arm_pool(rng, K, d):
    X = rng.normal(size=(K, d))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    return X

# -------------------------
# Algorithms
# -------------------------
class GLB_OMD:
    """
    O(1) per-round OMD-UCB (H and H^{-1} 유지, Sherman–Morrison)
    패치 반영: step = solve(A, g) (H@g 아님)
    """
    def __init__(self, d, S, link: Link, delta=0.05, lam=0.05, c_beta=2.0, mup_clip=1e-6, warmup=100):
        self.d, self.S = d, float(S)
        self.link = link
        self.delta, self.c_beta = float(delta), float(c_beta)
        self.mup_clip = float(mup_clip)
        self.eta = 1.0 + self.S  # R=1 가정
        self.lam = float(lam)
        self.theta = np.zeros(d)
        self.H = np.eye(d) * self.lam
        self.Hinv = np.eye(d) / self.lam
        self.t = 0
        self.warmup = int(warmup)

    def _beta(self):
        return beta_radius(self.t + 1, self.d, self.S, self.delta, self.c_beta)

    def select(self, X, rng):
        if self.t < self.warmup:
            return rng.integers(0, len(X))
        quad = np.einsum('ij,jk,ik->i', X, self.Hinv, X)
        score = X @ self.theta + self._beta() * np.sqrt(np.maximum(quad, 1e-12))
        return int(np.argmax(score))

    def update(self, x, r):
        # gradient & local Hessian coeff at theta_t
        z = float(x @ self.theta)
        mu_val = float(self.link.mu(z))
        mup = max(float(self.link.mu_prime(z)), self.mup_clip)  # 안정화
        g = (mu_val - r) * x                                   # ∇ℓ_t(θ_t)

        # A = H + η * L_t  with L_t = mup * x x^T
        A = self.H + self.eta * mup * np.outer(x, x)
        step = np.linalg.solve(A, g)                           # 핵심 패치: g 그대로
        theta_new = self.theta - self.eta * step               # θ_{t+1}
        theta_new = l2_project(theta_new, self.S)

        # 누적 곡률 H 갱신은 θ_{t+1} 기준
        z_next = float(x @ theta_new)
        alpha = max(float(self.link.mu_prime(z_next)), self.mup_clip)
        # Sherman–Morrison for Hinv
        v = self.Hinv @ x
        denom = 1.0 + alpha * float(x @ v)
        self.Hinv = self.Hinv - (alpha / denom) * np.outer(v, v)
        self.H = self.H + alpha * np.outer(x, x)

        self.theta = theta_new
        self.t += 1


class GLM_UCB:
    """
    간이 GLM-UCB: 매 라운드 일부 MLE GD/뉴턴 반복 → UCB 선택
    느리지만 후회는 낮게 나올 수 있음. 실행시간 비교 용도.
    """
    def __init__(self, d, S, link: Link, lam=0.05, iters=3, lr=0.5, delta=0.05, c_beta=2.0):
        self.d, self.S = d, float(S)
        self.link = link
        self.lam = float(lam)
        self.iters, self.lr = int(iters), float(lr)
        self.delta, self.c_beta = float(delta), float(c_beta)
        self.theta = np.zeros(d)
        self.Xhist, self.rhist = [], []

    def _beta(self, t):
        return beta_radius(t, self.d, self.S, self.delta, self.c_beta)

    def _fit_mle_once(self):
        # 뉴턴-스텝 하나 또는 준-뉴턴
        th = self.theta.copy()
        H = np.eye(self.d) * self.lam
        g = np.zeros(self.d)
        for x, r in zip(self.Xhist, self.rhist):
            z = float(x @ th)
            mu = float(self.link.mu(z))
            mup = float(self.link.mu_prime(z))
            g += (mu - r) * x
            H += mup * np.outer(x, x)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ g
        th = th - self.lr * step
        self.theta = l2_project(th, self.S)
        return H

    def select(self, X, t, H_last):
        if len(self.Xhist) < 5:
            return np.random.randint(len(X))
        Hinv = np.linalg.pinv(H_last)
        quad = np.einsum('ij,jk,ik->i', X, Hinv, X)
        score = X @ self.theta + self._beta(t) * np.sqrt(np.maximum(quad, 1e-12))
        return int(np.argmax(score))

    def update(self, x, r):
        self.Xhist.append(x)
        self.rhist.append(r)

    def fit_and_H(self):
        H = np.eye(self.d) * self.lam
        for _ in range(self.iters):
            H = self._fit_mle_once()
        return H


class GLOC:
    """
    단순 선형 근사 UCB (가벼운 기준선)
    """
    def __init__(self, d, S, lam=0.05, delta=0.05, c_beta=2.0):
        self.d, self.S = d, float(S)
        self.lam = float(lam)
        self.delta, self.c_beta = float(delta), float(c_beta)
        self.V = np.eye(d) * self.lam
        self.b = np.zeros(d)
        self.theta = np.zeros(d)
        self.t = 0

    def _beta(self):
        return beta_radius(self.t + 1, self.d, self.S, self.delta, self.c_beta)

    def select(self, X):
        Vinv = np.linalg.pinv(self.V)
        quad = np.einsum('ij,jk,ik->i', X, Vinv, X)
        score = X @ self.theta + self._beta() * np.sqrt(np.maximum(quad, 1e-12))
        return int(np.argmax(score))

    def update(self, x, r):
        self.V += np.outer(x, x)
        self.b += r * x
        Vinv = np.linalg.pinv(self.V)
        self.theta = Vinv @ self.b
        self.theta = l2_project(self.theta, self.S)
        self.t += 1

# -------------------------
# One experiment run (one trial)
# -------------------------
def run_trial(rng, kind, S, d=20, K=20, T=3000, show_progress=True):
    link = Link(kind)
    theta_star = rng.normal(size=d)
    theta_star = theta_star / np.linalg.norm(theta_star) * S
    X = make_arm_pool(rng, K, d)

    omd = GLB_OMD(d, S, link, delta=0.05, lam=0.05, c_beta=2.0, mup_clip=1e-6, warmup=100)
    ucb = GLM_UCB(d, S, link, lam=0.05, iters=3, lr=0.7, delta=0.05, c_beta=2.0)
    gloc = GLOC(d, S, lam=0.05, delta=0.05, c_beta=2.0)
    algos = {"GLB-OMD": omd, "GLM-UCB": ucb, "GLOC": gloc}

    regrets = {k: [] for k in algos}
    runtimes = {}

    def best_mu(): return np.max(link.mu(X @ theta_star))

    # 실시간 그래프 초기화
    if show_progress:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6,4))
        lines = {k: ax.plot([], [], label=k)[0] for k in algos}
        ax.legend()
        ax.set_title(f"{kind.capitalize()} Bandit (S={S})")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Regret")
        plt.show()

    # 실험 루프
    for name, algo in algos.items():
        cum, regrets[name] = 0.0, []
        t0 = time.time()
        for t in trange(T, desc=f"{kind}-{name}-S{S}", leave=False):
            if isinstance(algo, GLB_OMD):
                idx = algo.select(X, rng)
            elif isinstance(algo, GLM_UCB):
                H_last = algo.fit_and_H()
                idx = algo.select(X, t+1, H_last)
            else:
                idx = algo.select(X)

            x = X[idx]
            r = link.sample_reward(float(x @ theta_star), rng)
            mu_sel = float(link.mu(float(x @ theta_star)))
            cum += best_mu() - mu_sel
            regrets[name].append(cum)

            algo.update(x, r)

            # 100회마다 그래프 갱신
            if show_progress and (t+1) % 100 == 0:
                for k in algos:
                    lines[k].set_data(np.arange(len(regrets[k])), regrets[k])
                ax.relim(); ax.autoscale_view()
                plt.pause(0.001)
        runtimes[name] = time.time() - t0

    if show_progress:
        # plt.ion()
        # plt.show()
        plt.savefig("figure.png", dpi=200)
        plt.close() 
    return regrets, runtimes

# -------------------------
# Multi-trial runner
# -------------------------
def run_all(kind="logistic", S_list=(3, 5), d=20, K=20, T=3000, n_trials=10, seed=0):
    rng = np.random.default_rng(seed)
    all_means = {}
    all_stds = {}
    all_times = {}
    algo_names = ["GLB-OMD", "GLM-UCB", "GLOC"]

    for S in S_list:
        agg = {name: [] for name in algo_names}
        times_agg = {name: [] for name in algo_names}
        for _ in range(n_trials):
            regrets, times = run_trial(rng, kind, S, d, K, T)
            for name in algo_names:
                agg[name].append(np.array(regrets[name]))
                times_agg[name].append(times[name])
        # align & aggregate
        for name in algo_names:
            arr = np.stack(agg[name], axis=0)   # (n_trials, T)
            all_means[(S, name)] = arr.mean(axis=0)
            all_stds[(S, name)]  = arr.std(axis=0)
            all_times[(S, name)] = np.mean(times_agg[name])
    return all_means, all_stds, all_times

# -------------------------
# Plotting like the paper
# -------------------------
def plot_figures(kind, means, stds, times, S_list=(3,5), T=3000, savepath=None):
    algo_order = ["GLB-OMD", "GLM-UCB", "GLOC"]
    colors = {"GLB-OMD": "#1f77b4", "GLM-UCB": "#9467bd", "GLOC": "#2ca02c"}

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for i, S in enumerate(S_list):
        for name in algo_order:
            m = means[(S, name)]
            s = stds[(S, name)]
            axs[i].plot(m, label=name, color=colors[name])
            axs[i].fill_between(np.arange(len(m)), m - s, m + s, color=colors[name], alpha=0.15, linewidth=0)
        axs[i].set_title(f"{kind.capitalize()} Bandit (S={S})")
        axs[i].set_xlabel("Iterations")
        axs[i].set_ylabel("Cumulative Regret")
        axs[i].legend()

    # runtime bars (log10 seconds)
    labels = []
    vals = []
    for S in S_list:
        for name in algo_order:
            labels.append(f"{name} S={S}")
            vals.append(times[(S, name)])
    axs[2].bar(labels, np.log10(vals))
    axs[2].set_ylabel("log10 Runtime (s)")
    axs[2].set_title("Running Time")

    fig.suptitle(f"Regret and Running Time – {kind.capitalize()} Bandit")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=220)
    plt.show()
# -------------------------
# Numeric comparison (summary table)
# -------------------------
def summarize_results(kind, means, times, S_list=(3,5), T=3000):
    print(f"\n===== Numeric Summary for {kind.capitalize()} Bandit =====")
    print(f"{'Algorithm':<12} {'S':<3} {'FinalRegret':>12} {'Regret/1kStep':>15} {'Runtime(s)':>12}")
    print("-"*60)
    algo_order = ["GLB-OMD", "GLM-UCB", "GLOC"]
    for S in S_list:
        for name in algo_order:
            final_regret = float(means[(S, name)][-1])
            slope = final_regret / (T / 1000.0)  # 1000-step 당 증가량
            runtime = float(times[(S, name)])
            print(f"{name:<12} {S:<3d} {final_regret:12.1f} {slope:15.2f} {runtime:12.2f}")
    print("-"*60)
    print("Lower Regret and Runtime indicate better performance.\n")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Figure 1: Logistic
    # means, stds, times = run_all(kind="logistic", S_list=(3,5), d=20, K=20, T=1500, n_trials=10, seed=0)
    # plot_figures("logistic", means, stds, times, S_list=(3,5), T=1500, savepath="figure1_logistic.png")
    means, stds, times = run_all(kind="logistic", S_list=(3,5), d=20, K=20, T=1000, n_trials=5, seed=0)
    plot_figures("logistic", means, stds, times, S_list=(3,5), T=1500, savepath="figure1_logistic.png")

    # # Figure 2: Poisson
    # means, stds, times = run_all(kind="poisson", S_list=(3,5), d=20, K=20, T=1500, n_trials=10, seed=1)
    # plot_figures("poisson", means, stds, times, S_list=(3,5), T=1500, savepath="figure2_poisson.png")
    summarize_results("poisson", means, times, S_list=(3,5), T=1000)    