# 実行手順

- `data_formatter.py`
  - スクレイピング結果を整形する
- `data_converter.py`
  - 整形したデータをLLMで文章から数値に変換する
- `visualize_raw.py`
  - ユーザーごとに状態の遷移を可視化する
- `train.py`
  - 変換された数値データを使ってモデルを学習する
- `predict.py`
  - 学習後のモデルを使って1ステップ先の予測を行う
