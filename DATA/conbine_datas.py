import csv

# ASAの情報を除いた情報でまずは学習させる¥
WITH_ASA = './data.csv'
f = open(WITH_ASA,'r',encoding="utf-8")
input_data_list = csv.reader(f)
input_data_list = list(input_data_list)
input_residues = [ [data[0], data[1]] for data in input_data_list ]

WRITE_TO = './without_asa_data.csv'
with open(WRITE_TO, 'w') as f:
  writer = csv.writer(f)
  for residues in input_residues:
    writer.writerow(residues)
