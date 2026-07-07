## 3 Methods and experimental details

### 3.1 High-level methodology

- [ ] **Step 1: Collect demonstration data, and train a supervised policy.**
過去に OpenAI API を介して送信されたプロンプトに対して、理想的な回答を人間が作成する。
このデータを用いて、事前学習済みモデルを教師ありファインチューニングする。

- [ ] **Step 2: Collect comparison data, and train a reward model.**
言語モデルの応答とそれに対する評価（どちらの回答の方が最も好ましいか）からなるデータセットを作成する。
人間の評価を予測するような報酬モデルを学習させる。

- [ ] **Step 3: Optimize a policy against the reward model using PPO.**
上で学習させた報酬モデルを用いた PPO(Proximal Policy Optimization) によって、報酬を最大化するようにモデルを更新する。

Step 2 と Step 3 は繰り返すことができる。
実際に与えられた選択肢の大部分からなるものであり、 PPO によって最適化途中のモデルからの応答は多くは含んでいない。

### 3.2 Dataset

OpenAI API を介して得られたプロンプトをユーザー一人当たり最大 200 個使用した。
また、各プロンプトにおいて、共通するテキストを多く含むものについては、ヒューリスティックに重複削除した。

人間のラベラーに対しては、以下の 3 種類のプロンプトとその回答を作成するように依頼した。

- [ ] **Plain:**
多様なタスクからなる任意のプロンプト

- [ ] **Few-shot:**
指示文とそれに対応する複数の回答

- [ ] **User-based:**
API の利用申請フォームをもとに作成された、想定される実際の使用例を参考にしたもの

このようにして集められたデータを、目的別に以下 3 種類のデータセットへ分割した。

| Dataset | # samples | API samples | Human samples |
|---------|:---------:|:-----------:|:-------------:|
| SFT     |       13k |         yes |           yes |
| RM      |       33k |         yes |           yes |
| PPO     |       31k |         yes |            no |

### 3.5 Models

- [ ] **Supervised fine-tuning (SFT)**
コサイン学習率減衰と 0.2 の残差ドロップアウトを使用し、16 エポック学習させた。
先行研究と同様に、1 エポックで過剰適合が確認された。
一方で、より多くのエポックでの学習が RM スコアや人間による評価を向上させることがわかった。

- [ ] **Reward model (RM)**
SFT モデルを初期値として、unembedding 層を排除したモデルを使用する。

プロンプト $x$ に対する応答 $y_w, y_l$ で、より報酬の高かった方を $y_w$ とする。
モデルの出力する報酬ロジットをそれぞれ $r_w, r_l$ とすると、$y_l$ が選ばれる確率は
$$
\frac{\sigma(r_w)(1 - \sigma(r_l))}{\sigma(r_w)(1 - \sigma(r_l)) + (1 - \sigma(r_w))\sigma(r_l)} = \sigma(r_w - r_l)
$$
とかける。

また、 $k$ 個の応答から選ばれた 2 つの応答に対する評価が人間によって行われており、過学習を避けるために、 ${}_kC_2$ 個全てを平均したものを 1 要素とした。
$$
\text{loss}(\theta) = - \frac{1}{{}_kC_2} \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma(r_w - r_l)\right]
$$

- [ ] **Reinforcement learning (RL)**
強化学習によって、以下の目的関数を最大化するようにモデルを更新する。
$$
\text{objective}(\phi) = 
\mathbb{E}_{(x, y) \sim \mathcal{D}_{\pi_{\phi}^{\text{RL}}}} \left[
    r_{\theta}(x, y) -
    \beta \log \left(\frac{\pi_{\phi}^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}\right)
\right] +
\gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[
    \log \pi_{\phi}^{\text{RL}}(x)
\right]
$$

単に強化学習させただけのモデル $(\gamma=0)$ を "PPO" と呼ぶ。
また、事前学習の枠組みを継続学習させたモデル $(\gamma>0)$ を "PPO-ptx" と呼ぶ。

---

## 4 Results

1.3B のパラメータを持つ InstructGPT モデルが、175B パラメータを持つ SFT モデルよりも、人間に好まれるような回答を生成することができた。

GPT-3 < GPT-3 (Few-shot) < SFT < PPO という順序が確認された。
また、PPO-ptx の性能は PPO とほぼ同であったものの、ハルシネーションの低減が見られた。
