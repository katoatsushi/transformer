import csv

"""
  # ./data.csvの先頭1000行を ./head_1000_from_data.csv に書き込むコマンド
  head -n 1000 ./data.csv > ./head_1000_from_data.csv
"""

BORDER = 30

default_counter = {"H": 0, "#": 0, "S": 0}
EQUALITY = round((1/len(default_counter.keys()))*100, 1)


def calculate_gap(params):
  counter = {"H": 0, "#": 0, "S": 0}
  cal_equality_dic = {}
  for residues in params:
    counter = {"H": 0, "#": 0, "S": 0}
    for struc in params.split(' '):
      counter[struc] += 1
    total_count = sum(counter.values())
    # どれほど散らばっているかを確認する
    count_to_percentages = [ abs(round((count_val/total_count)*100, 1) - EQUALITY) for count_val in counter.values() ]
    if(sum(count_to_percentages) < BORDER):
      return True
    else:
      return False


# 全体感としてどのくらいのばらつきがあるのか測定
def analysis_equality(input_residues):
  # ASAの情報を除いた情報でまずは学習させる
  cal_equality_dic = {}
  for index,residues in enumerate(input_residues):
    counter = {"H": 0, "#": 0, "S": 0}
    structures = residues[1].split(' ')
    for struc in structures:
      counter[struc] += 1
    
    total_count = sum(counter.values())
    # どれほど散らばっているかを確認する
    count_to_percentages = [ abs(round((count_val/total_count)*100, 1) - EQUALITY) for count_val in counter.values() ]
    cal_equality_dic[str(index)] = sum(count_to_percentages)

    # 大体、50以下であれば 5000件のレコードを抽出できる
  sort_by_equality = sorted(cal_equality_dic.items(), key=lambda x:x[1])


def data_init():
  WITH_ASA = './head_10000_from_data.csv'
  f = open(WITH_ASA,'r',encoding="utf-8")
  input_data_list = csv.reader(f)
  input_data_list = list(input_data_list)
  init_data = [ [data[0], data[1]] for data in input_data_list ]
  return init_data

# CSVからデータを取り出す
datas = data_init()

# # 二次構造の配列数とアミノ酸の配列数は同じ
# counter = 0
# for d in datas:
#   if(len(d[0].split(' ')) != len(d[1].split(' '))):
#     counter += 1
# print("間違え:", str(counter))
# # ここまで

# # データを分析する
# analysis_equality(datas)

good_datas = []
for data in datas:
  if calculate_gap(data[1]):
    good_datas.append(data)
  
print("取り出されたデータ群は:", len(good_datas))

WRITE_TO = './equality_data.csv'
with open(WRITE_TO, 'w') as f:
  writer = csv.writer(f)
  for residues in good_datas:
    writer.writerow(residues)

# ここでの出力は少しずれている
"""
  CSV内部の行数 == "辞書のキーの部分 + 1行"
"""
