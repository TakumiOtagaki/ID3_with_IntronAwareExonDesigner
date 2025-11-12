# Intron対応 mRNA 設計 README（日本語）

このドキュメントは、ID3 フレームワークを用いた **intron-aware exon 設計** フローのみをまとめたものです。既存デモの DeepRaccess 最適化は利用せず、ViennaRNA による構造ペナルティ（ensemble free energy と末端塩基の塩基対確率）だけで最適化を行います。

---

## 1. 必要環境

- Python 3.11+
- `pip install -r requirements.txt` を実行して依存ライブラリを導入  
  - ViennaRNA の Python バインディング（`ViennaRNA`）は必須です。  
  - GPU は不要です（CPU で十分）。

---

## 2. 入力ファイル形式

1. **アミノ酸配列**  
   - `--protein-seq` で直接渡すか、`--protein-file` で FASTA を指定します。
2. **マルチ FASTA (`--structure-fasta`)**  
   - エントリ順は `>5utr` → `>main` → `>3utr`。  
   - `>main` では exon を **大文字**、intron を **小文字**で記述します。  
   - 5'/3' UTR は固定で使われ、intron も書かれた通り固定されます。変化させるのは exon 領域のみです。

例)

```
>5utr
GGGAAAUAAGAGAGAAAAGAAGAGUAAGAAGAAAUAUAAGAGCCACC
>main
AUGGUGGUGGUGgtaagtt...ttcagAAAGAA...
>3utr
UGAA
```

初期 exon 配列が指定アミノ酸配列と矛盾している場合は警告が出ます（処理は継続）。

---

## 3. 目的関数

- **Window EFE loss**: intron 全域と、上流 60 nt / 下流 30 nt を含むウィンドウの ensemble free energy を高くする（`loss = -EFE`）。
- **Boundary BPP loss**: 各 intron の 5'/3' 端から 3 塩基ずつ計 6 塩基について、全長配列の塩基対確率合計を最小化。

DeepRaccess 由来の損失は使用しません。

---

## 4. コマンドライン実行

```
python demo.py \
  --structure-fasta data/intron_examples/egfp_with_intron.fasta \
  --protein-seq MSKGEELFTGVVPILVELDGDVNGHKFSVSGEG \
  --constraint lagrangian \
  --mode det.hard \
  --iterations 200 \
  --efe-weight 2.0 \
  --boundary-weight 0.5 \
  --window-upstream 60 \
  --window-downstream 30 \
  --boundary-flank 3
```

### 重みを変えるには？
- `--efe-weight`: intron window の `-EFE` ペナルティをどれだけ重くするか（大きいほど構造を取りづらくする）。
- `--boundary-weight`: intron 両端の塩基対確率ペナルティの強さ。

いずれも実行時にフラグを書き換えるだけで変更できます。

---

## 5. YAML コンフィグの利用

長いコマンドラインを書きたくない場合は `--config path/to/config.yaml` を渡すことでパラメータをまとめて設定できます。  
キー名は CLI 引数から `--` を除いて `_` にしたものを使います（例: `efe_weight`, `structure_fasta`）。

サンプル: `id3/config/intron_design_example.yaml`

```yaml
structure_fasta: data/intron_examples/egfp_with_intron.fasta
protein_seq: MSKGEELFTGVVPILVELDGDVNGHKFSVSGEG
constraint: lagrangian
mode: det.hard
iterations: 200
learning_rate: 0.01
efe_weight: 2.0
boundary_weight: 0.5
window_upstream: 60
window_downstream: 30
boundary_flank: 3
```

使用例:

```
python demo.py --config id3/config/intron_design_example.yaml
```

CLI で同時に指定した場合は、CLI が YAML を上書きします。

---

## 6. 出力

- 進捗は `tqdm` で表示され、反復ごとの `total loss / -EFE / boundary` を確認できます。
- 最終的な exon 配列と UTR を含む pre-mRNA 全体が標準出力に表示されます。必要であればカスタム保存処理を追加してください。

---

## 7. よくある質問

- **ViennaRNA が見つからないと言われる**  
  → `uv pip install ViennaRNA` を実行済みか確認してください。  
- **設定を変えても結果が同じ**  
  → `--mode sto.*` にして探索ノイズを増やす、Iterations を増やす、learning rate を調整するなどを試してください。

---

この README は intron-aware 設計ワークフロー専用です。本体 README の DeepRaccess デモとは目的関数が異なる点に注意してください。
