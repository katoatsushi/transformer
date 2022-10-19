import csv

path_a = './without_gap_input.csv'
f = open(path_a,'r',encoding="utf-8")
input_data_list = csv.reader(f)
input_data_list = list(input_data_list)
input_residues = [ ' '.join(i) for i in input_data_list ]

path_b = './without_gap_output.csv'
f = open(path_b,'r',encoding="utf-8")
output_data_list = csv.reader(f)
output_data_list = list(output_data_list)
output_residues = [ ' '.join(i) for i in output_data_list ]

WRITE_TO = './onestr_data.csv'

with open(WRITE_TO, 'w') as f:
  writer = csv.writer(f)
  for acids, structure in zip(input_residues, output_residues):
    writer.writerow([acids, structure])
