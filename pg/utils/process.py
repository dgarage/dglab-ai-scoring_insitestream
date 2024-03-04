import pandas as pd
from tqdm import tqdm

from tqdm.notebook import tqdm

def make_train_data(augumented_data, base_month, explain_columns, target_column,  data_folda, genre_list, read_from_pickle=False):
    
    if(read_from_pickle==True):
        augumented_data_train=pd.read_pickle(data_folda+'train'+str(base_month)+".pkl")
    else:
        augumented_data_train=augumented_data[(augumented_data['base_date'].dt.strftime('%Y%m')==base_month)
                                            &(augumented_data["eval_period"]==True)]

        # augumented_data_train["closed_flg_lag3"]を予測するためのカラムのみ選択
        # 説明変数に使うのは、以下のカラム
        # name,latitue,longitude,genre,area_name,prefecture,genres,"openning_days"
        # 目的変数はclosed_flg_lag3
        augumented_data_train=augumented_data_train[explain_columns+[target_column]]
 
        #genre_listでgenresをダミー変数化
        glist = [pd.DataFrame() for i in range(len(genre_list))]
        
        # 進行状況を表示
        for i in tqdm(range(len(genre_list))):
            glist[i]=augumented_data_train["genres"].str.contains(genre_list[i]).astype(int)
        
        augumented_data_train=pd.concat([augumented_data_train]+glist,axis=1)
        
        augumented_data_train.columns=explain_columns+[target_column]+genre_list
               
        # prefecture をダミー変数化
        augumented_data_train=pd.get_dummies(augumented_data_train,columns=["prefecture"])
        
        # nameは同名店数で置き換え
        name_size=augumented_data_train["name"].value_counts().reset_index().rename(columns={"index":"name","name":"name_size"})
        augumented_data_train=pd.merge(augumented_data_train,name_size,on="name",how="left")

        # area_nameは使わずにモデルを作成、タミー化したもの、および同名店に置き返したnameは削除
        augumented_data_train.drop(columns=["name","genres"],inplace=True)

        #営業日は日数->整数に変換
        augumented_data_train["openning_days"]=augumented_data_train["openning_days"].dt.days.astype(int)

        # longitude,latitudeはfloatに変換
        augumented_data_train["longitude"]=augumented_data_train["longitude"].astype(float)
        augumented_data_train["latitude"]=augumented_data_train["latitude"].astype(float)

        augumented_data_train.to_pickle(data_folda+'train'+str(base_month)+".pkl")
    
    return augumented_data_train

price_list=['～￥999',
 '￥1,000～￥1,999',
 '￥2,000～￥2,999',
 '￥3,000～￥3,999',
 '￥4,000～￥4,999',
 '￥5,000～￥5,999',
 '￥6,000～￥7,999',
 '￥8,000～￥9,999',
 '￥10,000～￥14,999',
 '￥15,000～￥19,999',
 '￥20,000～￥29,999',
 '￥30,000～￥39,999',
 '￥40,000～￥49,999',
 '￥50,000～￥59,999',
 '￥60,000～￥79,999',
 '￥80,000～￥99,999',
 '￥100,000～']

def get_price_range_dummy(tabelog,column_name):
    
    tabelog[column_name+"_num"]=tabelog[column_name]
    
    for p in range(len(price_list)):
        print(price_list[p])
        # tabelog[column_name]について、price_list[p]をstr(p)に置換する
        tabelog[column_name+"_num"]=tabelog[column_name+"_num"].str.replace(price_list[p],str(p))
        
        #tabelog[column_name+"_num"]=tabelog[column_name].str.replace(price_list[p],p)
        
    return tabelog


def extract_features(data_folda,fname,explain_columns,target_column):
    pseudo_time_series_with = pd.read_pickle(data_folda+fname)
    pseudo_time_series_with=pseudo_time_series_with[pseudo_time_series_with["general_rank_GOOGLE"].isin(["S","A","B"])]
    pseudo_time_series_with=pseudo_time_series_with[pseudo_time_series_with["general_rank_RETTY"].isin(["S","A","B"])]
    pseudo_time_series_with=pseudo_time_series_with[pseudo_time_series_with["general_rank_HOTPEPPER"].isin(["S","A","B"])]
    pseudo_time_series_with=pseudo_time_series_with[explain_columns+[target_column]]
    
    return pseudo_time_series_with
