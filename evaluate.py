import csv

# サンプル
path = './DATA/logs.csv'

dictionary = {"A": "#","B": "S", "C": "H"}

f = open(path,'r',encoding="utf-8")
train_rows = csv.reader(f)
header = next(train_rows)
train_rows = list(train_rows)

result_rate = []
for row in train_rows:
    amino_acids = row[0].split(' ')
    structures = row[1].split(' ')
    correct, error = 0, 0
    for amino, structure in zip(amino_acids, structures):
        # print(amino, structure)
        if(dictionary[amino] == structure):
            correct += 1
        else:
            error += 1

    # 正解率を入力
    total = correct + error
    if(total == 0):
        result_rate.append(0)
    else:
        result_rate.append(correct/total)
# print(result_rate)

total_rate = 0
for rate in result_rate:
    total_rate += rate

print(total_rate/len(result_rate))
# 0.33712942641010013
# 10000件*10Epochでの結果

# 10000件*30Epochでの結果 => ?
# 10000件*50Epochでの結果 => ?
