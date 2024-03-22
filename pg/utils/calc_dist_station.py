import pandas as pd
import sys
from utils import *
from rank import *
from process import * 
from tqdm.notebook import tqdm
import seaborn as sns

# データ保存先
data_folda = '../../data/'

#stationのlat,lonとpseudo_time_seriesの'northlatitude','eastlongitude'の距離を計算
# station は世界測地系、pseudo_time_series は日本測地系なので、両方を世界測地系に変換してから距離を計算する
from geopy.distance import geodesic
from pyproj import Proj, transform
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
tqdm.pandas(desc="Processing")

station=pd.read_csv(data_folda+"/station_address/station20240318free.csv")
# 同じ station_cd の数をカウントして、付与する
staion_cd_count=station.groupby('station_g_cd').size().reset_index(name='count').sort_values('count', ascending=False)
station=pd.merge(station,staion_cd_count, on='station_g_cd', how='left')
pseudo_time_series = pd.read_pickle(data_folda+'train_with_master.pkl')
pseudo_time_series_unique_rest=pseudo_time_series.drop_duplicates(subset=['restaurant_id'])

#stationのlat,lonを日本測地系に変換
def wgs2jgd(station):
    inProj = Proj(init='epsg:4326')
    outProj = Proj(init='epsg:2451')
    x1,y1 =station['lon'],station['lat']
    x2,y2 = transform(inProj,outProj,x1,y1)
    return x2,y2

res = station[['lon','lat']].progress_apply(lambda x: wgs2jgd(x), axis=1)
station[['eastlongitude','northlatitude']] = pd.DataFrame(res.tolist(), columns=['eastlongitude','northlatitude'])

station.to_pickle(data_folda+'station_address_jp_axis.pkl')

station = pd.read_pickle(data_folda+'station_address_jp_axis.pkl')

# pseudo_time_series_unique_rest の rat_lon と station の lat_lon の距離を計算
# 計算負荷を減らすために、seudo_time_series_unique_rest に対し、±0.1度以内の station のみを抽出して距離を計算する
# 0.1度はおおよそ11km
# 0.05度はおおよそ5.5km

def get_near_stations(lat_lon, distance=0.05):
    near_stations = []
    #print(station['lat'],lat_lon.values[0])
    #print(len(station[abs(station['lat']-lat_lon.values[0])<distance]))
    #print(len(station[abs(station['lon']-lat_lon.values[1])<distance]))
    
    temp=station[(abs(station['lat']-lat_lon.values[0])<distance)&(abs(station['lon']-lat_lon.values[1])<distance)]
    lat_lon = (lat_lon.values[0],lat_lon.values[1])
    
    #最も近い駅を探す

    """
    for i, row in temp.iterrows():
        dist=geodesic(lat_lon,(row['lat'],row['lon'])).m
        #最も短いstation_g_cdを返す
        near_stations.append((row['station_g_cd'],dist))
    """
    """
    #loopが遅いので、applyで回す
    temp["dist"] = temp.apply(lambda row: geodesic(lat_lon,(row['lat'],row['lon'])).m,axis=1)
    """

    #loopが遅いので、applyで回す
    near_stations = temp.apply(lambda row: [row['station_cd'],geodesic(lat_lon,(row['lat'],row['lon'])).m],axis=1).values.tolist()
    
    if len(near_stations)==0:
        return [np.nan,np.nan]
    
    #print(near_stations)
    # near_stations 2次元リストからを pandas に変換
    near_stations = pd.DataFrame(near_stations, columns=['station_cd','dist']) 
    #print("pd",near_stations)
    
    near_stations = near_stations.sort_values('dist',ascending=True)[['station_cd','dist']].values.tolist()
    return near_stations[0]
        
         
    
pseudo_time_series_unique_rest.rename(columns={'northlatitude':'lat','eastlongitude':'lon'},inplace=True)
res = pseudo_time_series_unique_rest[['lat','lon']].progress_apply(lambda x: get_near_stations(x),axis=1)

res = pd.DataFrame(res.tolist(), columns=['station_cd','dist'])
temp=pseudo_time_series_unique_rest.reset_index()
temp['station_cd'] = res['station_cd']
temp['dist_from_station'] = res['dist']
temp.to_pickle(data_folda+'train_with_station_dist.pkl')

