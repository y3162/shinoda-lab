## 2. Preliminaries: Mixture-of-Experts for Transformers

Transformer の第 $l$ 層への入力を $h_{1:T}^{l-1}$ とする。
自己注意機構による出力 $u_{1:T}^{l}$ を用いると、第 $l$ 層の出力 $h_{1:T}^{l}$ は以下のように計算される。
$$
\begin{align*}
    u_{1:T}^{l} &= \text{Self-Att}\left(h_{1:T}^{l-1}\right) + h_{1:T}^{l-1} \\
    h_{t}^{l} &= \text{FFN}\left(u_{t}^{l}\right) + u_{t}^{l}
\end{align*}
$$

MoE 言語モデルは、特定の（周期の）FFN を MoE 化する。
$$
\begin{align*}
    h_{t}^{l} &= \sum_{i=1}^{N} \left(g_{i,t} \text{FFN}_{i}\left(u_{t}^{l}\right)\right) + u_{t}^{l} \\
    g_{i, t} &= \begin{cases}
        s_{i,t} & s_{i,t} \in \text{Topk}\left(\{s_{j,t}\}_{j=1}^{N}, K\right)\\
        0 & \text{otherwise}
    \end{cases} \\
    s_{i,t} &= \text{Softmax}_{i}\left({u_{t}^{l}}^{\top} e_{i}^{l}\right)
\end{align*}
$$

---

## 3. DeepSeekMoE Architecture

### 3.1. Fine-Grained Expert Segmentation

通常の MoE では、専門家1つが多様なタスクを担当することになっていた。
DeepSeekMoE では通常の MoE と比べて、専門家の数を $m$ 倍に増やし、 $m$ 倍の専門家が選択される。
各専門家のパラメータは $1/m$ に縮小されており、各 FFN の総パラメータ数は一定となっている。
$$
\begin{align*}
    h_{t}^{l} &= \sum_{i=1}^{mN} \left(g_{i,t} \text{FFN}_{i}\left(u_{t}^{l}\right)\right) + u_{t}^{l} \\
    g_{i, t} &= \begin{cases}
        s_{i,t} & s_{i,t} \in \text{Topk}\left(\{s_{j,t}\}_{j=1}^{mN}, mK\right)\\
        0 & \text{otherwise}
    \end{cases} \\
    s_{i,t} &= \text{Softmax}_{i}\left({u_{t}^{l}}^{\top} e_{i}^{l}\right)
\end{align*}
$$
これによって、専門家選択の組み合わせ数は ${}_NC_K$ から ${}_{mN}C_{mK}$ に増え、$O(m^{m})$ 倍に増える。

### 3.2. Shared Expert Isolation

異なるタスク間でも共通の知識は必要であるはずであり、共通の知識を持つ専門家を設計し強制的にルーティングすることで専門家の冗長性を減らす。
このような専門家は $K_s$ 個設計されるが、選択される専門家の数は $mK$ のままである。

$$
\begin{align*}
    h_{t}^{l} &= \sum_{i=1}^{K_s} \text{FFN}_{i}\left(u_{t}^{l}\right) + \sum_{i=K_s+1}^{mN} \left(g_{i,t} \text{FFN}_{i}\left(u_{t}^{l}\right)\right) + u_{t}^{l} \\
    g_{i, t} &= \begin{cases}
        s_{i,t} & s_{i,t} \in \text{Topk}\left(\{s_{j,t}\}_{j=1}^{mN}, mK - K_s\right)\\
        0 & \text{otherwise}
    \end{cases} \\
    s_{i,t} &= \text{Softmax}_{i}\left({u_{t}^{l}}^{\top} e_{i}^{l}\right)
\end{align*}
$$

### 3.3. Load Balance Consideration

- [ ] **Expert-Level Balance Loss.**

$$
\begin{align*}
    \mathcal{L}_{\text{ExpBal}} &= \alpha_1 \sum_{i=1}^{mN - K_s} f_i P_i \\
    f_i &= \frac{mN -K_s}{(mK - K_s)T} \sum_{t=1}^{T} 1\left[\text{Token } t \text{ selects Expert } i\right] \\
    P_i &= \frac{1}{T} \sum_{t=1}^{T} s_{i,t}
\end{align*}
$$

---

## 4. Validation Experiments

### 4.2. Evaluations

GShard($K=2,m=1,K_s=0$) と DeepSeekMoE($K=2,m=4,K_s=1$) を比較した。
これら2つのモデルは、同じ総パラメータ数(2.0B)・活性化パラメータ数(0.3B)を持つ。
また、同じ訓練データ量(100B)で事前学習した。

| Metric | GShard | DeepSeekMoE |
| :----- | :----: | :---------: |
| Pile (Loss)           | 1.867 | **1.808** |
| TriviaQA (EM)         |  10.2 |  **16.6** |
| NaturalQuestions (EM) |   3.2 |   **5.7** |

### 4.3. DeepSeekMoE Aligns Closely with the upper bound of MoE Models

DeepSeekMoE と同等の層パラメータ数を持つ Dense $\times16$ と比較して、同程度の性能を達成している。
Dense モデルは、同程度の総パラメータ数を持つ MoE モデルの上限として解釈できる。
また、GShard が同程度の性能となるには専門家のパラメータ数を 1.5 倍にする必要があり、DeepSeek の MoE 化が重要であることを示している。

| Metric | Dense $\times4$ | Dense $\times16$ | GShard $\times1.5$ | DeepSeekMoE |
| :----- | :-------------: | :--------------: | :----------------: | :---------: |
| Relative Expert Size      |     1 |         1 |      1.5 |     0.25 |
| # Experts                 |   0+4 |      0+16 |     0+16 |     1+63 |
| # Activated Experts       |   0+4 |      0+16 |      0+2 |      1+7 |
| # Total Expert Params     | 0.47B |     1.89B |    2.83B |    1.89B |
| # Activated Expert Params | 0.47B |     1.89B |    0.35B |    0.24B |
| Pile (Loss)               | 1.908 | **1.806** |    1.808 |    1.808 |
| TriviaQA (EM)             |   9.9 |      16.5 |     15.7 | **16.6** |
| NaturalQuestions (EM)     |   3.0 |   **6.3** |      4.7 |      5.7 |

### 4.4. Ablation Studies

- [ ] **Shared Expert Isolation.**

2 つの専門家を選ぶよりも、1 つの専門家を共有した上でもう 1 つの専門家を選ぶ方が言語モデルの性能を向上させる。

- [ ] **Fine-Grained Expert Segmentation.**

専門家の粒度 $m$ を大きくすればするほど、言語モデルのの性能が向上する。

- [ ] **Ratios Between Shared and Routed Experts.**

共有専門家の数 $K_s$ は、$K_s : (mK - K_s) = 1 : 3$ が最適であったが、性能には大きな影響を与えない。

### 4.5. Analysis on Expert Specialization

GShard$\times1.5$ と DeepSeekMoE において、上位から特定の割合だけ専門家を除外した場合の性能を比較した。
どちらとも、$1/16$ の除外で性能が大幅に向上するものの、DeepSeekMoE の方が性能低下が顕著で、割合が増加しても性能低下が継続する。
DeepSeekMoE では各専門家が特定のタスク（・ドメイン）を担当していることが示唆される。
