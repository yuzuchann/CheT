import requests
import csv
from tqdm import tqdm
import time
import os

def get_all_material_data_from_oqmd():
    initial_url = "http://oqmd.org/oqmdapi/formationenergy?fields=name,spacegroup,band_gap,delta_e"
    csv_name = "all_materials_data2.csv"

    if not os.path.exists(csv_name):
        with open(csv_name, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Material_Name", "Space_Group", "Band_Gap", "Stability"])
            f.close()

    with open(csv_name, "r") as f:
        total_downloaded = sum(1 for line in f) - 1  # 减去标题行

    link = initial_url + "&offset=" + str(total_downloaded)
    response = requests.get(link)
    if response.status_code == 200:
        data = response.json()
        total_entries = data["meta"]["data_available"]
        pbar = tqdm(total=total_entries, initial=total_downloaded)
        while link is not None:
            link = get_all_links_and_data(link, csv_name, pbar)  # 更新link的值
            time.sleep(1)  # 在每次请求之间添加一秒的延迟
        pbar.close()
    else:
        print("请求失败，状态码：", response.status_code)
    print("数据已经全部下载")

def get_all_links_and_data(link, csv_name, pbar):
    response = requests.get(link)
    if response.status_code == 200:
        data = response.json()
        with open(csv_name, "a", newline="") as f:
            csv_writer = csv.writer(f)
            for entry in data["data"]:
                material_name = entry["name"]
                space_group = entry["spacegroup"]
                band_gap = entry["band_gap"]
                stability = entry["delta_e"]
                csv_writer.writerow([material_name, space_group, band_gap, stability])
                pbar.update(1)
            f.close()
        return data["links"]["next"]  # 返回下一个链接
    else:
        print("请求失败，状态码：", response.status_code)

# 调用函数
get_all_material_data_from_oqmd()
