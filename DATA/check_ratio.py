import csv

path = './onestr_data.csv'
f = open(path,'r',encoding="utf-8")
train_rows = csv.reader(f)
train_rows = list(train_rows)

res = {}
for row in train_rows:
  secondary_residues = row[1].split(' ')
  # キーが存在したら追加、なければ作成
  for residue in secondary_residues:
    if residue in res:
      res[residue] += 1
    else:
      res[residue] = 1

print(res)
# "二次構造 埋もれ度"
# {'#': 1235948, 'S': 540195, 'H': 1192590, '': 28373, 'N': 122916, 'B': 1980808, 'E': 358743}
