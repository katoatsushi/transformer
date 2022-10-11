# transformer


- `transformer/DATA/data.csv` が学習に使用するデータ
  - アミノ酸配列 , 該当する二次構造情報の配列 の　2列のデータが `data.csv` に格納
- 学習を行わせるプログラムは `bio_transformer.py` 

- 学習後、テストの結果は、 `DATA/logs.csv` に吐き出される
  - 結果の出力は `入力のアミノ酸配列` ・ `予測の2次構造情報` ・ `正解の2次構造情報` の3列がcsvの形式で一覧で書き込まれる
