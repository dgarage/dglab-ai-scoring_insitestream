import pandas as pd
import datetime
import sys
sys.path.append('./utils')
from utils import *
from process import * 
from add_list_master import *
import japanize_matplotlib
from tqdm.notebook import tqdm
import numpy as np
from numba import jit
from tqdm import tqdm


# データ保存先
data_folda = '../data/'

target_column="target6"
make_genre_list=True

train_name="train_data_fix_dtypes.pkl"
test_name="test_data_fix_dtypes.pkl"

#train_data = pd.read_pickle(data_folda+train_name)
test_data = pd.read_pickle(data_folda+test_name)


# 距離計算関数（id 引数を除去）
@jit(nopython=True)
def calculate_distance_from_lat_lon(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    r_lat1 = np.radians(lat1)
    r_lat2 = np.radians(lat2)

    a = np.sin(d_lat / 2.0) ** 2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(d_lon / 2.0) ** 2
    haversine = 2 * R * np.arcsin(np.sqrt(a))

    return haversine

# 近隣店舗を見つける関数
@jit(nopython=True)
def find_nearby_stores(lat, lon, lats, lons, ids, threshold=0.5):
    nearby_ids = np.empty(0, dtype=np.int64)  # 近隣店舗のIDを格納するためのNumPy配列
    for i in range(len(lats)):
        dist = calculate_distance_from_lat_lon(lat, lon, lats[i], lons[i])
        #距離が短くて自分以外ならpush_back
        if ((dist < threshold)& (ids[i] != i)):
            nearby_ids = np.append(nearby_ids, ids[i])
    
    #自分自身を除く
    #nearby_ids = nearby_ids[nearby_ids != id]
    
    return nearby_ids

# 初期化
# 仮定: test_data は既にある DataFrame で、特定の列が含まれている
test_data["nearby_ids"] = None  # 列を None で初期化

# object 型のリストとして扱えるようにする
test_data = test_data.astype({'nearby_ids': 'object'})

# 県別に処理
for prefecture_name in test_data['prefecture_name'].unique():
    print(prefecture_name)
    mask = test_data['prefecture_name'] == prefecture_name
    filtered_data = test_data[mask].copy()
    filtered_lats = filtered_data['northlatitude'].values
    filtered_lons = filtered_data['eastlongitude'].values
    filtered_ids = filtered_data.index.values

    # 近隣店舗のIDリストを保持するための一時的なリスト
    nearby_ids_list = []

    for i in tqdm(range(len(filtered_data))):
        row = filtered_data.iloc[i]
        nearby_ids = find_nearby_stores(row['northlatitude'], row['eastlongitude'], filtered_lats, filtered_lons, filtered_ids, 0.5)
        nearby_ids_list.append(nearby_ids)

    # test_date のうち mask に該当する行の 'nearby_ids' 列に代入
    for ind, m in enumerate(test_data[mask].index):
        #print(m,test_data.at[m, 'nearby_ids'],nearby_ids_list[ind])
        test_data.at[m, 'nearby_ids'] = nearby_ids_list[ind]

test_data.to_pickle(data_folda+"test_data_nearby_ids.pkl")       

