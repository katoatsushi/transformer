import requests
import json
import csv
import time

URL = 'https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt'
PDB_ID_LIST = './pdb_ids.txt'

# response = requests.get(URL)
# response_body = response.text.split('\n') # 338113行あった
# # 各行の頭だけを持ってくる
# head_pdb_list = [ res.split(' ')[0] for res in response_body ]

# f = open(PDB_ID_LIST, 'a', newline='\n')
# f.close()
# f = open(PDB_ID_LIST, 'w', newline='\n')
# for pdb in head_pdb_list:
#     f.write(pdb+'\n')
# f.close()
# CSVに書き込む

f = open(PDB_ID_LIST, newline='\n')
pdbIdList = f.readlines()
pdbIdList = [ pdbId.replace('\n','') for pdbId in pdbIdList ]

def n_length_array(num, kind):
    if(kind == "structure"):
        str = "#"
    elif(kind == "asa"):
        str = "-1"
    res = []
    for _ in range(num):
        res.append("#")
    return res

PDB_API_ENDPOINT = "https://data.rcsb.org/rest/v1/core/"
CSV_PATH = '../data.csv'

def reset_data_csv():
    with open(CSV_PATH, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')

# # data.csvをまっさらにする
# reset_data_csv()
# failed_pdb_csv()

# "https://data.rcsb.org/rest/v1/core/polymer_entity/3T5V/2"
# 3T5V_2 まではうまくいった　"23102行目" 12:41
# 6BTM_3 まではうまくいった　"33382行目" 13:35
# 1YRT_1 まではうまくいった　"33581行目" 13:56
# 7F9I_1 69149

failed_pdb = []

# AF_AFP46308F1_1が10行目だった場合　pdbIdList[9:] で行う
for pdb in pdbIdList[69149:]:
    print(pdb, "......................")
    res = {
        "amino_residues": [],
        "secondary_list": [],
        "asa_list": []
    }
    items = pdb.split('_')
    entity_id = items[-1]
    entry_id = '_'.join(items[:len(items)-1])
    if (items[0] == "AF"):
        print("AlpheFoldによる構造決定は無視   PDBID ",pdb)
        continue
    # X線構造解析かどうか？
    entry_url = PDB_API_ENDPOINT + "entry/" + entry_id
    response = requests.get(entry_url)
    jsonData = response.json()
    # X線構造解析でなく、3Åより大きかったら捨てる
    try:
        ("X-ray" in jsonData["rcsb_entry_info"]["experimental_method"]) and (jsonData["rcsb_entry_info"]["resolution_combined"][0] < 3)
    except:
        # failed_pdb.append(pdb)
        print("PDB_ID:", pdb, " で詳細情報取得失敗")
        continue
    # ChainIDを取得
    entity_url = PDB_API_ENDPOINT + "polymer_entity/" + entry_id + "/" + entity_id
    response = requests.get(entity_url)
    jsonData = response.json()
    
    # アミノ酸配列
    amino_residues = jsonData["entity_poly"]["pdbx_seq_one_letter_code_can"]

    chain_ids = jsonData["rcsb_polymer_entity_container_identifiers"]["asym_ids"]
    chain_url = PDB_API_ENDPOINT + "polymer_entity_instance/" + entry_id + "/" + chain_ids[0]
    jsonData = requests.get(chain_url).json()

    secondary_list = n_length_array(len(list(amino_residues)), "structure")
    asa_list = n_length_array(len(list(amino_residues)), "asa")
    try:
        for feature in jsonData["rcsb_polymer_instance_feature"]:
            if(feature["name"] == "sheet"):
                for feature_position in feature["feature_positions"]:
                    beg_seq_id, end_seq_id = feature_position["beg_seq_id"], feature_position["end_seq_id"]
                    for index in range(beg_seq_id - 1,end_seq_id):
                        secondary_list[index] = "S"
            elif(feature["name"] == "helix"):
                for feature_position in feature["feature_positions"]:
                    beg_seq_id, end_seq_id = feature_position["beg_seq_id"], feature_position["end_seq_id"]
                    for index in range(beg_seq_id - 1,end_seq_id):
                        secondary_list[index] = "S"
            elif(feature["type"] == "ASA"):
                for feature_position in feature["feature_positions"]:
                    beg_seq_id, end_seq_id, values = feature_position["beg_seq_id"], feature_position["end_seq_id"], feature_position["values"]
                    asa_list[beg_seq_id-1:end_seq_id] = [ str(round(value, 2)) for value in values ]
    except:
        # failed_pdb.append(pdb)
        print("PDB_ID:", pdb, " で詳細情報取得失敗")
        continue

    res["amino_residues"] = ' '.join(list(amino_residues))
    res["secondary_list"] = ' '.join(secondary_list)
    res["asa_list"] = ' '.join(asa_list)

    print("元データ長さ:", len(list(amino_residues)))
    print("amino_residuesの長さは:", len(res["amino_residues"].split(' ')))
    print("secondary_listの長さは:", len(res["secondary_list"].split(' ')))
    print("asa_listの長さは:", len(res["asa_list"].split(' ')))

    # CSVに書き込み
    with open(CSV_PATH, 'a') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([res["amino_residues"], res["secondary_list"], res["asa_list"]])
    # 失敗したPDBを書き込み
    f = open('../fetched_pdb.txt', 'a', newline='\n')
    f.write(pdb+'\n')
    f.close()
    # f = open('./failed_pdb.txt', 'a', newline='\n')
    # f.write(pdb+'\n')
    # f.close()
