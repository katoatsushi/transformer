import csv
import random


dictionary = {
  "A": "#",
  "B": "S",
  "C": "H"
}
amino_index = ["A","B","C"]

DATA_NUM = 12000

CSV_PATH = './test_data.csv'
with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)

for _ in range(DATA_NUM):
    residue_list = []
    secondary_list = []

    # ランダムに100 ~ 300の範囲の数値を作成
    amino_length = random.randint(100, 300)
    
    amino_list = []
    structure_list = []
    for _ in range(amino_length):
        index = random.randint(0, len(amino_index) - 1)
        amino_list.append(amino_index[index])
        structure_list.append(dictionary[amino_index[index]])

    insert_data = [' '.join(amino_list), ' '.join(structure_list)]
    # csvモジュールを使って1行の内容をCSVファイルに書き込み
    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(insert_data)

