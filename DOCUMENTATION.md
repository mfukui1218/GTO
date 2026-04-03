# GTO Solver — プロジェクトドキュメント

## 概要

テキサスホールデムを中心としたポーカーのGTO（Game Theory Optimal）ソルバー。
CFR（Counterfactual Regret Minimization）アルゴリズムによるナッシュ均衡計算、ハンド評価、エクイティ計算、レンジ分析、ポストフロップ復習機能を備える。

- **言語**: Rust (Edition 2021)
- **コード規模**: 約6,300行（Rust + HTML）
- **テスト**: 36テスト（gto-eval 31, gto-cfr 5）全パス
- **Web UI**: シングルページアプリケーション（7タブ、日英対応）

---

## プロジェクト構成

```
GTO/
├── Cargo.toml              # ワークスペース定義 + メインバイナリ
├── src/
│   ├── main.rs             # CLI エントリポイント (252行)
│   └── web/
│       ├── mod.rs           # モジュール宣言
│       ├── server.rs        # axum ルーター定義 (39行)
│       ├── state.rs         # 共有状態 AppState (21行)
│       └── handlers.rs      # APIハンドラー全11個 (977行)
├── static/
│   └── index.html           # Web UI SPA (1,331行)
└── crates/
    ├── gto-core/            # カード・アクション基本型 (247行)
    ├── gto-eval/            # ハンド評価・エクイティ計算 (1,454行)
    ├── gto-cfr/             # CFRソルバーエンジン (640行)
    ├── gto-games/           # ゲーム実装 (1,333行)
    └── gto-abstraction/     # 将来拡張用 (1行)
```

---

## クレート詳細

### gto-core（基盤型）

ポーカーの基本データ型を定義。

| 型 | 説明 |
|---|---|
| `Card(u8)` | 0-51のID。`rank = id / 4`, `suit = id % 4` |
| `CardSet(u64)` | ビットマスクによるカード集合（高速な包含判定） |
| `Action` | `Fold`, `Check`, `Call`, `Bet(i32)`, `Raise(i32)` |
| `GameState` | `Terminal(Vec<f64>)` / `Chance(Vec<Action>)` / `Decision{player, actions}` |

`Card::new(rank, suit)` でランク0-12（2-A）、スート0-3（c/d/h/s）から生成。

### gto-eval（評価エンジン）

**evaluator.rs** — 5枚/7枚ハンドの役判定
- `evaluate_5(cards: &[Card; 5]) -> HandRank` — 全10カテゴリ（HighCard〜RoyalFlush）
- `evaluate_7(cards: &[Card; 7]) -> HandRank` — 7枚から最強5枚を選択
- `HandRank` — u32ベースの比較可能な役ランク

**equity.rs** — エクイティ計算
- `equity_exact(hand1, hand2, board) -> EquityResult` — 完全列挙エクイティ
- `equity_monte_carlo(hand1, hand2, samples, rng) -> EquityResult` — MC近似
- `hand_class(c1, c2) -> usize` — 169クラスへの分類
- `hand_class_name(c1, c2) -> String` — "AKs", "QQ"等の名前

**hand_rank.rs** — `HandRank`構造体とカテゴリ定数

**range_equity.rs** — レンジ対レンジ計算（666行）
- `range_vs_range_equity(range1, range2, board) -> RangeEquityResult` — ボード指定時の完全列挙
- `range_vs_range_monte_carlo(range1, range2, samples, rng) -> RangeEquityResult` — プリフロップ用MC
- `hand_vs_range_equity(hero, villain_range, board) -> f64` — ハンド対レンジ
- `expand_class_combos(class, freq, dead) -> Vec<([Card; 2], f64)>` — クラスをコンボに展開
- `count_outs(hero, villain_range, board) -> u32` — アウツ数カウント
- `board_texture(board) -> BoardTexture` — ボードテクスチャ分析
- `class_index_to_name(class) -> String` — 169クラスインデックスから名前
- `range_stats(range) -> (f64, f64)` — レンジ%とコンボ数

169ハンドクラスのインデックス体系:
- 0-12: ペア（22〜AA）
- 13-90: スーテッド（32s〜AKs）
- 91-168: オフスーテッド（32o〜AKo）
- 計算式: `pair = rank`, `suited = 13 + high*(high-1)/2 + low`, `offsuit = 91 + high*(high-1)/2 + low`

### gto-cfr（ソルバーエンジン）

CFR（Counterfactual Regret Minimization）アルゴリズムの実装。

- `train(game, config) -> CfrSolver` — 指定イテレーション数でCFR学習
- `CfrSolver` — 累積リグレット・戦略を保持
- `Strategy` — 学習済み戦略の取得・表示
- `InformationSet` — 情報集合ノード（リグレットマッチング）
- `TrainerConfig` — `iterations`, `use_cfr_plus`, `print_interval`

`Game` トレイトを実装すれば任意のゲームに対応可能。

### gto-games（ゲーム実装）

| ゲーム | ファイル | 説明 |
|---|---|---|
| `KuhnPoker` | kuhn.rs (379行) | 3枚カードの最小ポーカー。ナッシュ均衡の検証用 |
| `LeducHoldem` | leduc.rs (488行) | 6枚2ラウンドのミニポーカー。プリフロップ+フロップ |
| `PushFoldGame` | push_fold.rs (459行) | SB Push or Fold / BB Call or Fold の2人ゲーム |
| `PostflopGame` | postflop.rs | ポストフロップ3ストリート。MCCFR対応、バケット抽象化対応 |

Push/Fold:
- `PushFoldData::compute(samples)` — 169x169エクイティ行列をMCで計算（rayon並列）
- `extract_push_range()` / `extract_call_range()` — 戦略から13x13チャート抽出
- `display_chart()` — ターミナル表示

Postflop:
- **設定**: フロップ指定、ポット/スタック、OOP/IPレンジ、ベットサイズ設定
- **抽象化**: `num_buckets > 0` でハンド強度バケットを使用。169クラス → Nバケットに圧縮
  - フロップ: `gto-abstraction` のプリコンピュートバケッティング
  - ターン: 残りリバーカードの平均hand strengthからバケット
  - リバー: `evaluate_7` の直接hand strengthからバケット
- **CFR**: MCCFR（チャンスサンプリング）+ CFR+で学習
- `extract_postflop_strategies()` — 決定点ごとの13x13グリッド戦略を抽出

---

## CLI使用方法

```bash
# ビルド
cargo build --release

# 各モード実行
cargo run -- kuhn        # Kuhn Poker CFR学習
cargo run -- leduc       # Leduc Hold'em CFR学習
cargo run -- equity      # ハンドエクイティ表示
cargo run -- pushfold    # Push/Fold チャート計算
cargo run -- web [port]  # Web UI起動（デフォルト8080）
```

---

## Web UI

### 技術スタック
- バックエンド: axum 0.8 + tokio
- フロントエンド: バニラHTML/CSS/JS（フレームワークなし）
- HTMLは `include_str!("../../static/index.html")` でバイナリに埋め込み
- ダークテーマ、日英切り替え対応

### 7タブ構成

| タブ | 説明 |
|---|---|
| **Kuhn Poker** | CFRで学習した最適戦略を表示 |
| **Leduc Hold'em** | プリフロップ+フロップの最適戦略を表示 |
| **Equity Calculator** | 2ハンド間のエクイティ計算（ボード指定可） |
| **Push/Fold** | スタックサイズ別のPush/Callチャート（非同期計算） |
| **Range Editor** | 13x13インタラクティブグリッドでレンジ作成・保存 |
| **Range vs Range** | 2つのレンジのエクイティ比較+ヒートマップ |
| **Postflop Review** | ポストフロップのEV分析・推奨アクション |

### APIエンドポイント

| エンドポイント | メソッド | 説明 |
|---|---|---|
| `/` | GET | Web UI HTML |
| `/api/kuhn` | POST | Kuhn CFR学習実行 |
| `/api/leduc` | POST | Leduc CFR学習実行 |
| `/api/equity` | POST | 2ハンド間エクイティ |
| `/api/pushfold/start` | POST | Push/Fold計算開始（非同期） |
| `/api/pushfold/status/{job_id}` | GET | Push/Foldジョブ状態 |
| `/api/range-equity` | POST | レンジ対レンジエクイティ |
| `/api/range-equity/status/{job_id}` | GET | レンジエクイティジョブ状態 |
| `/api/postflop-review` | POST | ポストフロップEV分析 |
| `/api/range-presets` | GET | ポジション別プリセットレンジ |

### 非同期ジョブパターン

Push/FoldとRange Equity（プリフロップ）は計算に時間がかかるため非同期パターンを使用:

1. `POST /api/.../start` → `{ "job_id": "uuid" }` を即返却
2. バックグラウンドで `tokio::task::spawn_blocking` 実行
3. フロントエンドが2秒間隔でポーリング: `GET /api/.../status/{job_id}`
4. 完了時に `{ "status": "done", "result": {...} }` を返却

状態管理: `AppState.jobs: Arc<Mutex<HashMap<String, JobStatus>>>`

### 13x13レンジグリッド

インタラクティブなハンドレンジ入力UI:

- 対角線: ペア (AA, KK, ... 22)
- 右上三角: スーテッド (AKs, AQs, ...)
- 左下三角: オフスーテッド (AKo, AQo, ...)
- クリックでON/OFF、ドラッグで複数選択
- 緑（高頻度）〜赤（低頻度）のグラデーション
- プリセット読み込み + localStorage保存

### プリセットレンジ

9種のポジション別プリセット:

| プリセット | レンジ% | 概要 |
|---|---|---|
| UTG Open | ~15% | タイトなオープンレンジ |
| MP Open | ~20% | ミドルポジション |
| CO Open | ~27% | カットオフ |
| BTN Open | ~40% | ボタン |
| SB Open | ~45% | スモールブラインド |
| BB Defense | ~38% | ビッグブラインドディフェンス |
| Top 10% | 10% | プレミアムハンド |
| Top 20% | 20% | 上位レンジ |
| Top 50% | 50% | ルースレンジ |

### Postflop Review

ポストフロップのアクション分析:

**入力:**
- ヒーローハンド（2枚）
- ボード（3-5枚）
- ポットサイズ、エフェクティブスタック、ヴィランベット額
- ヴィランレンジ（プリセットまたはカスタム）

**出力:**
- エクイティ（%）
- ポットオッズ
- EV比較テーブル:
  - `EV(Call) = equity × (pot + bet) - (1 - equity) × bet`
  - `EV(Fold) = 0`（ベースライン）
  - `EV(Raise) = fold_eq × pot - (1 - fold_eq) × raise_cost`（簡易推定）
- アウツ数
- ボードテクスチャ（Paired / Monotone / Flush Draw / Straight Draw）
- 推奨アクション（Call / Fold / Raise バッジ）

---

## 依存関係

| クレート | バージョン | 用途 |
|---|---|---|
| axum | 0.8 | Webフレームワーク |
| tokio | 1 (full) | 非同期ランタイム |
| tower-http | 0.6 | CORSミドルウェア |
| uuid | 1.10 | ジョブID生成 |
| rand | 0.8 | 乱数生成 |
| rand_chacha | 0.3 | 決定的RNG |
| rayon | 1.10 | データ並列処理 |
| rustc-hash | 2 | 高速ハッシュマップ |
| serde / serde_json | 1 | JSON シリアライズ |

外部ポーカーライブラリへの依存なし。ハンド評価・エクイティ計算はすべて自前実装。

---

## アルゴリズム

### CFR (Counterfactual Regret Minimization)

各情報集合で累積リグレットを追跡し、リグレットマッチングで戦略を更新:

```
σ(a) = max(R(a), 0) / Σ max(R(a'), 0)
```

イテレーションを重ねると平均戦略がナッシュ均衡に収束（O(1/√T)）。

#### CFR計算フロー

```
cfr(game, state, reach_probs, traversing_player):
  1. 終端ノード → ペイオフを返却
  2. チャンスノード → 全アウトカムの期待値を返却
  3. 意思決定ノード:
     a. 情報集合キーを取得（プレイヤーの観測可能情報）
     b. リグレットマッチングで現在の戦略を計算（キャッシュ利用）
     c. 累積戦略を到達確率で重み付け更新
     d. 各アクションについて再帰的にCFR値を計算
     e. traversing_playerの場合、反事実リグレットを更新しキャッシュ無効化
```

#### CFRバリアント

| バリアント | 特徴 | 用途 |
|---|---|---|
| Vanilla CFR | 全チャンスアウトカムを列挙 | 小規模ゲーム（Kuhn, Leduc） |
| CFR+ | 負のリグレットを0にクランプ（収束高速化） | 中規模ゲーム |
| Chance Sampling MCCFR | チャンスノードで1つだけサンプリング | 大規模ゲーム（Push/Fold, Preflop） |
| Chance Sampling + CFR+ | 上記2つの組み合わせ | 最大規模ゲーム |

#### Exploitability（搾取可能度）計算

ナッシュ均衡への収束度を測定する指標。値が0に近いほど最適戦略に近い。

```
exploitability = (BR_value(player0) + BR_value(player1)) / 2
```

1. 全ノードの平均戦略を一括計算してキャッシュ
2. 各プレイヤーについてBest Response（最適応答）戦略を反復計算（最大20回）
3. BR戦略が収束したら、その戦略でのゲーム価値を評価

### メモ化（キャッシュ機構）

計算の重複を排除するため、以下の3層のメモ化を実装:

#### 1. current_strategy() キャッシュ（InfoSetNode）

```rust
pub struct InfoSetNode {
    cumulative_regret: Vec<f64>,
    cumulative_strategy: Vec<f64>,
    cached_strategy: Option<Vec<f64>>,  // ← メモ化フィールド
}
```

- リグレットマッチングの計算結果を `cached_strategy` にキャッシュ
- `cumulative_regret` が更新されたときのみ `invalidate_cache()` でキャッシュを無効化
- 同一イテレーション内で同じノードに複数回アクセスする場合、2回目以降はキャッシュヒット

**無効化タイミング:**
- CFR traversalでリグレット更新後
- CFR+/MCCFR+の負リグレットクランプ後

#### 2. accumulate_strategy の二重計算除去

```
変更前: strategy = node.current_strategy()   ← 1回目の計算
        node.accumulate_strategy(reach_prob)  ← 内部で2回目の計算

変更後: strategy = node.current_strategy()   ← 1回だけ計算
        node.accumulate_strategy_with(&strategy, reach_prob)  ← 結果を再利用
```

CFRの各イテレーションで全ノードに対して発生する重複計算を排除。`accumulate_strategy_with()` は外部から渡された戦略をそのまま使用し、内部で再計算しない。

#### 3. exploitability計算の average_strategy() 一括キャッシュ

```
変更前: collect_br_action_values() → 各ノードで node.average_strategy() を都度計算
        eval_with_br_strategy()    → 各ノードで node.average_strategy() を都度計算
        × 最大20回のBR反復 → 同じ計算が数百〜数千回発生

変更後: avg_strategies = 全ノードの average_strategy() を1回だけ計算してMapに保存
        collect_br_action_values() → avg_strategies から O(1) ルックアップ
        eval_with_br_strategy()    → avg_strategies から O(1) ルックアップ
```

exploitability計算中は平均戦略が変化しないため、事前計算が安全。

### 並列化（rayon）

計算量の大きい処理をrayonで並列化:

| 対象 | 方式 | 効果 |
|---|---|---|
| `PushFoldData::compute()` | MCサンプルをスレッド数で分割、各スレッドが独立にサンプリング後マージ | MC計算がCPUコア数に比例して高速化 |
| `range_vs_range_equity()` | ヒーロークラスごとに`par_iter`で並列評価 | レンジ対レンジ計算がコア数に比例して高速化 |
| `exploitability()` | プレイヤー別Best Response計算を`par_iter`で並列実行 | 2人ゲームで約2倍高速化 |

### evaluate_7() ルックアップテーブル

非フラッシュの7枚ハンド評価を事前計算テーブルで O(1) 化:

```
ランク分布の総数: C(19, 7) = 50,388 通り
テーブルサイズ: 50,388 × 4 bytes ≈ 200KB
インデックス: コンビナトリアル数体系（多重集合のランキング）
```

**仕組み:**
1. 7枚のランク分布（各ランク0-4枚の13要素配列、合計7）を一意のインデックスに変換
2. インデックスでフラット配列を直接参照 → HandStrength を O(1) で取得
3. フラッシュ判定は従来通りスート情報で行い、非フラッシュの役判定のみLUT化
4. `LazyLock` で初回アクセス時に1回だけテーブルを構築

**インデックス計算:**
```
sorted_ranks[7] = rank_countsから昇順に展開
x[i] = sorted_ranks[i] + i  （重複を排除して狭義単調増加に変換）
index = Σ C(x[i], i+1)      （コンビナトリアル数体系）
```

### エクイティ計算

- **完全列挙**: 残りカードの全組み合わせを評価（リバー: 1通り、ターン: 44通り、フロップ: C(44,2)=990通り、プリフロップ: C(48,5)=1,712,304通り）
- **モンテカルロ**: プリフロップ用、指定サンプル数でランダムボードを生成

### レンジ対レンジ

```
for each combo1 in range1:
    for each combo2 in range2 (カード重複除外):
        eq = equity_exact(combo1, combo2, board)
        weighted_sum += eq * weight1 * weight2
```

パフォーマンス（20%レンジ同士）:
- リバー: <100ms
- ターン: <500ms
- フロップ: 1-3秒
- プリフロップ: モンテカルロで2-5秒（非同期ジョブ）

---

## テスト

```bash
# 全テスト実行
cargo test

# クレート別
cargo test -p gto-eval    # 31テスト（評価・エクイティ・レンジ）
cargo test -p gto-cfr     # 5テスト（リグレットマッチング）
```

主要テスト:
- ハンド評価: 各役の判定（HighCard〜RoyalFlush）、7枚からの最強選択
- エクイティ: AA vs KK ≈ 81%、AKs vs QQ ≈ 46%（既知値との照合）
- レンジ: クラス展開（ペア6コンボ、スーテッド4コンボ、オフスーテッド12コンボ）
- モンテカルロ: 完全列挙との誤差 ±2%以内
- ボードテクスチャ: ペアド/モノトーン検出

---

## 開発フェーズ

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 1 | Kuhn Poker CFR | 完了 |
| Phase 2 | Leduc Hold'em CFR | 完了 |
| Phase 3 | ハンド評価 + エクイティ | 完了 |
| Phase 4 | Push/Fold ソルバー | 完了 |
| Phase 5 | Web UI (4タブ + i18n) | 完了 |
| Phase 6 | 実用ツール (Range Editor, Range vs Range, Postflop Review) | 完了 |
| Phase 7 | メモ化による計算最適化 (CFR strategy cache, BR average strategy cache) | 完了 |
| Phase 8 | パフォーマンス最適化 (rayon並列化, evaluate_7 LUT, range並列バッチ) | 完了 |
| Phase 9 | gto-abstraction統合 + Postflop本格CFR対応 (バケット抽象化, 収束テスト) | 完了 |
