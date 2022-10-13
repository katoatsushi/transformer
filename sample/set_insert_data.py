import csv
import random

AMINOS = [ "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y" ]
STRUCTURE = [ "HB", "HE", "SB", "SE", "#B", "#E" ]
DATA_NUM = 10000

CSV_PATH = './test_data.csv'

with open('test.csv', 'w', newline='') as f:
    writer = csv.writer(f)

for _ in range(DATA_NUM):
    residue_list = []
    secondary_list = []

    # ランダムに100 ~ 300の範囲の数値を作成
    # amino_length = random.randint(100, 300)
    amino_length = random.randint(100, 300)
    
    for i in range(amino_length):
        random_amino = random.randint(0, len(AMINOS) - 1)
        residue_list.append(AMINOS[random_amino])

    each_str_num = amino_length//len(STRUCTURE)
    amari = amino_length%len(STRUCTURE)
    for index in range(len(STRUCTURE)):
        for __ in range(each_str_num):
            secondary_list.append(STRUCTURE[index])
    for i in range(amari):
        secondary_list.append(STRUCTURE[i])
    random.shuffle(secondary_list)

    insert_data = [' '.join(residue_list), ' '.join(secondary_list)]
    # csvモジュールを使って1行の内容をCSVファイルに書き込み
    with open('test.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(insert_data)
