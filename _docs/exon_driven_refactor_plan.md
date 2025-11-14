# Exon-Driven Designer リファクタリング計画

## 1. 背景と目的
- イントロンは固定・エクソンだけが可変という本質を強調するため、現在の「Intron-aware exon design」を **Exon-Driven Designer** という名称で再定義する。
- `demo.py` に集中しすぎている処理をモジュール分割し、可読性・テスト可能性・再利用性を高めたい。
- 関数／クラス名には “ExonDriven” 系のプレフィックスや語感を取り入れ、ドキュメントやコード全体で一貫したブランドを構築する。

## 2. 要件
1. **仕様・用語の整理**
   - マルチFASTA 入力（5'UTR/main/3'UTR）、Exon/Introns のケース記法、ウィンドウ EFE、境界 BPP という要件を `_docs` の新文書で整理する。
   - 新しいドキュメント内で “Exon-Driven Designer” という名称を採用し、目的・制約・出力形式も明記する。
2. **コアユーティリティの分離**
   - `IntronDesignContext` を `ExonDrivenDesignContext` に刷新し、Exon だけがデザイン可能であることを強調する。
   - 共通のシーケンス入出力：FASTA からアミノ酸／UTR を読み、RNA→DNA 変換と UTR デフォルト取得を行うユーティリティモジュールを作成。
3. **デモのリファクタリング**
   - 現在 `demo.py` に書かれている `run_intron_structural_optimization` と `run_accessibility_optimization` をそれぞれ `id3.apps.exon_driven_structural` と `id3.apps.accessibility_runner` に分け、`demo.py` は CLI と高レベルの起動制御のみとする。
   - モード構成（`MODE_CONFIG`）も再配置し、CLI と各ランナーが共通で利用できるようにする。
   - 可能であれば、新モジュール内の関数にも “ExonDriven” を含めた命名（例：`run_exon_driven_structural_optimization`）を採用し、名称の統一感を担保。
4. **ドキュメント/呼び出し元の同期**
   - README や `_docs/intron_aware_design_requirements.md` を起点に、新しいモジュール名やクラス名に更新（例：ドキュメントで `IntronDesignContext` → `ExonDrivenDesignContext`）。
   - CLI ヘルプ文言や出力も可能な限り “Exon-Driven Designer” へ言い換える。

## 3. 実施ステップ
1. **設計仕様ドキュメント (_docs/exon_driven_refactor_plan.md) の作成**  
   - 主要なルール・入出力・目的関数をまとめ、今後のモジュール設計の基準とする。
2. **共有ユーティリティの整備**
   - `id3/utils/sequence_io.py` を追加し、FASTA → シーケンス、UTR のロード、RNA↔DNA 変換を提供。
   - `id3/utils/intron_design.py` を `ExonDrivenDesignContext` としてリネーム／コメント更新し、古い名前の互換エイリアスを置いて移行を滑らかにする。
3. **モード／実行機能の分割**
   - `id3/apps/constants.py` にモード辞書 (`MODE_CONFIG`) を移動。
   - `id3/apps/exon_driven_structural.py` に構造的最適化の処理を移し、フィードバックログ、保存処理も継承。
   - `id3/apps/accessibility_runner.py` に DeepRaccess ベースの実行を分離。
4. **CLI (= demo.py) の整理**
   - 上記ランナー関数を import して呼び出すようにし、引き続き `apply_config_overrides` を担当。
   - CLI ヘルプ／出力は必要に応じて “Exon-Driven Designer” に言い換える。

## 4. 次のアクション
1. モジュールを実装しながら `_docs/exon_driven_refactor_plan.md` の要件を満たすよう改修。
2. 各ユーティリティをテスト（現状テストが難しい場合でも `python -m compileall` 相当で構文確認）。
3. 変更内容を README や Japanese README に反映し、ユーザへの説明責任を果たす。
