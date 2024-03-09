import pandas as pd
from tqdm import tqdm
import numpy as np
from tqdm.notebook import tqdm
import rank
from process import * 
from utils import *

def add_master_data(pseudo_time_series, data_folda, master_name="restaurant_data.csv"):

    tabelog = pd.read_csv(data_folda+master_name)

    ### master data から加えるデータを定義
    add_list=['name',"prefecture_name",'head_branch','opened_on', 'created_at','genre_first_name', 
              'genre_second_name','genre_third_name','northlatitude', 'eastlongitude',
              "net_reservation_flg",'price_range_lunch_owner','price_range_dinner_owner', 
              'price_range_lunch_user','price_range_dinner_user']

    # カウント系のカラムを追加
    count_columns=tabelog.columns[tabelog.columns.str.contains('count')].tolist()

    add_list+=count_columns
    
    pseudo_time_series=pd.merge(pseudo_time_series, tabelog[["id"]+add_list], left_on="original_id",right_on='id', how='left')

    # 営業日の計算。* 本来は opened_date と created at の古い方を採用したいが、opened_date が古すぎて datetime型に変換出来ないので、created at のみで対応。     
    pseudo_time_series["real_opened_date"]=pd.to_datetime(pseudo_time_series["created_at"])

    # "created_at" と base_date の差分を計算
    pseudo_time_series["openning_days"] = pd.to_datetime(pseudo_time_series["base_date"])-pd.to_datetime(pseudo_time_series["real_opened_date"], errors='coerce')

    """
    # price range をダミー化
    p_list=['price_range_lunch_by_owner','price_range_dinner_by_owner', 'price_range_lunch_by_review','price_range_dinner_by_review']
    for p in p_list:
        pseudo_time_series=get_price_range_dummy(pseudo_time_series,p)
    """
    
    return pseudo_time_series


def add_v_google(pseudo_time_series, data_folda, add_list, master_name="V_GOOGLE"):

    pd_sf = read_table(data_folda,master_name,columns="all",read_snowflake=False,col="*")

    # general rank を与える
    ranker=rank.Ranker()
    pd_sf=ranker.add_general_rank(pd_sf)


    pseudo_time_series=pd.merge(pseudo_time_series,pd_sf[add_list],on="restaurant_id",how="left")
    pseudo_time_series.rename(columns={"general_rank":"general_rank_GOOGLE"},inplace=True)
    
    return pseudo_time_series

def add_v_retty(pseudo_time_series, data_folda, add_list, master_name="V_RETTY",read_from_pickle=False):

    pd_sf = read_table(data_folda,master_name,columns="all",read_snowflake=read_from_pickle,col="*")

    # general rank を与える
    ranker=rank.Ranker()
    pd_sf=ranker.add_general_rank(pd_sf)

    pd_sf.rename(columns={"general_rank":"general_rank_RETTY"},inplace=True)

    # info.isOfficial の空文字を0, trueを1に変換
    pd_sf["infos.isOfficial"]=pd_sf["infos.isOfficial"].replace({"":0,"true":1}).astype(int)

    # infos.onlineReservation falseを0, trueを1に変換
    pd_sf["infos.onlineReservation"]=pd_sf["infos.onlineReservation"].replace({"false":0,"true":1}).astype(int)
    # infos.access.transferTime1 徒歩X分のXを取り出す。intに変換
    pd_sf["infos.access.transferTime1"]=pd_sf["infos.access.transferTime1"].str.extract("(\d+)").astype(float)
    # infos.numberOfSeats.valueの "席"を取り除く
    pd_sf["infos.numberOfSeats.value"]=pd_sf["infos.numberOfSeats.value"].str.replace("席","")
    #空文字はNaNに変換にしてfloatに変換
    pd_sf["infos.numberOfSeats.value"]=pd_sf["infos.numberOfSeats.value"].replace("",np.nan)
    pd_sf["infos.numberOfSeats.value"]=pd_sf["infos.numberOfSeats.value"].astype(float)

    # infos.updateInfo.firstReviewDate の空文字をNaNに変換後、datetimeに変換
    pd_sf["infos.updateInfo.firstReviewDate"]=pd_sf["infos.updateInfo.firstReviewDate"].replace("",np.nan)
    pd_sf["infos.updateInfo.firstReviewDate"]=pd.to_datetime(pd_sf["infos.updateInfo.firstReviewDate"],format="%Y年%m月%d日")


    # pd_sf["infos.wantToGo"]の空文字をNaNに変換後、floatに変換
    pd_sf["infos.wantToGo"]=pd_sf["infos.wantToGo"].replace("",np.nan).astype(float)
    # pd_sf["infos.went"]の空文字をNaNに変換後、floatに変換
    pd_sf["infos.went"]=pd_sf["infos.went"].replace("",np.nan).astype(float)
    # infos.rateByWentの%を取り除いて、空文字をNaNに変換後、floatに変換
    pd_sf["infos.rateByWent"]=pd_sf["infos.rateByWent"].str.replace("%","").replace("",np.nan).astype(float)

    # infos.excellentの空文字をNaNに変換後、floatに変換
    pd_sf["infos.excellent"]=pd_sf["infos.excellent"].replace("",np.nan).astype(float)
    pd_sf[["infos.good","infos.average"]]=pd_sf[["infos.good","infos.average"]].replace("",np.nan).astype(float)

    # infos.onlineReservation falseを0, trueを1に変換
    pd_sf["infos.coupon"]=pd_sf["infos.coupon"].replace({"false":0,"true":1}).astype(int)

    # float_listをfloatに変換
    float_list=['infos.familiar.users.count',"infos.photoCount.Photograph","infos.photoCount.Cooking","infos.photoCount.Interior",
            "infos.photoCount.Exterior","infos.photoCount.Menu","infos.reviewCount.Review",
            "infos.reviewCount.Lunch","infos.reviewCount.Dinner",
            "menus.course_list_count","menus.dishes_list_count","menus.lunch_list_count",
            "menus.drink_list_count","menus.takeout_list_count"]
    
    for col in float_list:
        pd_sf[col]=pd_sf[col].replace("",np.nan).astype(float)
    
    pseudo_time_series=pd.merge(pseudo_time_series,pd_sf[add_list],on="restaurant_id",how="left")
    
    return pseudo_time_series


def add_v_hotpepper(pseudo_time_series, data_folda, add_list, master_name="V_HOTPEPPER",read_from_pickle=False):
    pd_sf = read_table(data_folda,master_name,columns="all",read_snowflake=read_from_pickle,col="*")

    # general rank を与える
    ranker=rank.Ranker()
    pd_sf=ranker.add_general_rank(pd_sf)

    pd_sf.rename(columns={"general_rank":"general_rank_HOTPEPPER"},inplace=True)
    
    # info.isOfficial の空文字を0, trueを1に変換
    pd_sf['infos.aggregateRating'].replace('',np.nan,inplace=True)
    pd_sf['infos.aggregateRating']=pd_sf['infos.aggregateRating'].astype(float)
    pd_sf['infos.ratingReview'].replace('',np.nan,inplace=True)
    pd_sf['infos.ratingReview']=pd_sf['infos.ratingReview'].astype(float)
    pd_sf['infos.satisfaction.percentage1'].replace('',np.nan,inplace=True)
    pd_sf['infos.satisfaction.percentage1']=pd_sf['infos.satisfaction.percentage1'].astype(float)

    hotpepper_list=['infos.fanCount','infos.reviewTags.count1','infos.reviewTags.count2','infos.reviewTags.count3', 'infos.reviewTags.count4',
       'infos.reviewTags.count5', 'infos.reviewTags.count6','infos.reviewTags.count7', 'infos.reviewTags.count8','infos.reviewTags.count9']
    for h in hotpepper_list:
        pd_sf[h].replace('',np.nan,inplace=True)
        pd_sf[h]=pd_sf[h].astype(float)
        pd_sf[h].value_counts()

    add_list=["general_rank_HOTPEPPER"]+add_list+hotpepper_list
    
    pseudo_time_series=pd.merge(pseudo_time_series,pd_sf[add_list],on="restaurant_id",how="left")
    
    return pseudo_time_series

def make_train_data(augumented_data, base_month, explain_columns, target_column,  data_folda, genre_list, read_from_pickle=False):
    
    if(read_from_pickle==True):
        augumented_data_train=pd.read_pickle(data_folda+'train'+str(base_month)+".pkl")
    else:
        augumented_data_train=augumented_data[(augumented_data['base_date'].dt.strftime('%Y%m')==base_month)
                                            &(augumented_data["eval_period"]==True)]

        # 目的変数はclosed_flg_lag3
        augumented_data_train=augumented_data_train[explain_columns+[target_column]]

        #genre_listでgenre_first_nameをダミー変数化。
        #genre_first_nameのみ、genre_second_name,genre_third_name はどうするか。
        
        glist = [pd.DataFrame() for i in range(len(genre_list))]       
        # 進行状況を表示
        for i in tqdm(range(len(genre_list))):
            glist[i]=augumented_data_train["genre_first_name"].str.contains(genre_list[i]).astype(int)

        augumented_data_train=pd.concat([augumented_data_train]+glist,axis=1)
        
        augumented_data_train.columns=explain_columns+[target_column]+["genre_"+g for g in genre_list]
        
        # prefecture をダミー変数化
        augumented_data_train=pd.get_dummies(augumented_data_train,columns=["prefecture_name"])
        
        # 各月の店舗数をカウント
        name_size=augumented_data_train[["name","base_date"]].value_counts().reset_index()
        name_size.rename(columns={0:"name_size"},inplace=True)
        augumented_data_train = pd.merge(augumented_data_train,name_size, on=["name","base_date"],how="left")
                
        #半径1km以内の店舗数をカウント
        #augumented_data_train["name_size_1km"]=augumented_data_train["northlatitude"].apply(lambda x:augumented_data_train[(augumented_data_train["northlatitude"]-x)**2+(augumented_data_train["eastlongitude"]-augumented_data_train["eastlongitude"])**2<0.00001].shape[0])
        
        
        # 前月/前々月からの店舗数の変化率をカウント
        #augumented_data_train["name_size_lag1_ratio"]=augumented_data_train.groupby("restaurant_id")["name_size"].shift(1).bfill()/augumented_data_train["name_size"]
        #augumented_data_train["name_size_lag2_ratio"]=augumented_data_train.groupby("restaurant_id")["name_size"].shift(2).bfill()/augumented_data_train["name_size"]
        
        """
        pref_count=pd.DataFrame()
        # prefecture 毎の月の店舗数の変化率をカウント
        for p in augumented_data_train["prefecture_name"].unique():
            pnum=augumented_data_train[augumented_data_train["prefecture_name"]==p].groupby("base_date")["name"].count()
            pref_count=pd.concat([pref_count,pnum.reset_index],axis=1)
   
        
        # augumented_data_trainにpref_countを結合
        augumented_data_train=pd.merge(augumented_data_train,pref_count,on=["base_date","pre"],how="left")
        """                                  
        #営業日は日数->整数に変換
        augumented_data_train["openning_days"]=augumented_data_train["openning_days"].dt.days.astype(int)

        # longitude,latitudeはfloatに変換
        augumented_data_train[['northlatitude', 'eastlongitude']]=augumented_data_train[['northlatitude', 'eastlongitude']].astype(float)

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

def compare_test_train_dist(data_folda,train='pseudo_time_series_with_v_hotpepper.pkl',test='pseudo_time_series_with_v_hotpepper_test.pkl'):
    #augumented_data = pd.read_pickle(data_folda+'feature_added_after2021.pkl')
    pseudo_time_series_with = train
    pseudo_time_series_with_test = test
   
    train_20231130_groupby_count=pseudo_time_series_with.groupby("target3").count().T.rename(columns={0:"target0_train_count",1:"target1_train_count"})
    train_20231130_test_groupby_count=pseudo_time_series_with_test.groupby("target3").count().T.rename(columns={0:"target0_test_count",1:"target1_test_count"})
    train_20231130_groupby_mean=pseudo_time_series_with.groupby("target3").mean().T.rename(columns={0:"target0_train_mean",1:"target1_train_mean"})
    train_20231130_test_groupby_mean=pseudo_time_series_with_test.groupby("target3").mean().T.rename(columns={0:"target0_test_mean",1:"target1_test_mean"})

        
    train_test=pd.concat([train_20231130_groupby_count,train_20231130_test_groupby_count,train_20231130_groupby_mean,train_20231130_test_groupby_mean],axis=1)
    
    train_test["test_ratio"]=train_test["target1_test_mean"]/train_test["target0_test_mean"]
    train_test["train_ratio"]=train_test["target1_train_mean"]/train_test["target0_train_mean"]

    train_test["test_ratio_over_train_ratio"]=train_test["test_ratio"]/train_test["train_ratio"]
    
    return train_test
    
    
def add_v_structured(pseudo_time_series, data_folda, add_list, master_name="V_STRUCTURED",read_from_pickle=False):
       
    pd_sf = read_table(data_folda,master_name,columns="all",read_snowflake=read_from_pickle,col=add_list)
    
    # pd_sf で 各rank が空のものを除外
    pd_sf=pd_sf[pd_sf["name_rank"].isnull()==False]
    pd_sf=pd_sf[pd_sf["address_rank"].isnull()==False]
    pd_sf=pd_sf[pd_sf["telephone_rank"].isnull()==False]
    
    # general rank を与える
    ranker=rank.Ranker()
    pd_sf=ranker.add_general_rank(pd_sf)

    pd_sf.rename(columns={"general_rank":"general_rank_STRUCTURED"},inplace=True)
    
    pd_sf.drop(columns=["name_rank","address_rank","telephone_rank"],inplace=True)
    
    V_MEDIA_SITE_LIST = read_table(data_folda,table_name="V_MEDIA_SITE_LIST",columns="all",read_snowflake=True,col='"name","hostname"')
    pd_sf["hostname"]=""
    
    
    for i in range(0,len(V_MEDIA_SITE_LIST)):
        print(V_MEDIA_SITE_LIST["hostname"].values[i])
        ### V_MEDIA_SITE_LIST["name"].values[i]　が　pd_sf["hostname"]に含まれている場合、pd_sf["hostname2"]にV_MEDIA_SITE_LIST["name"].values[i]を代入 
        pd_sf.loc[pd_sf["url"].str.contains(V_MEDIA_SITE_LIST["hostname"].values[i]),"hostname"] = V_MEDIA_SITE_LIST["name"].values[i]
        
    # 食べログのみを抽出
    pd_sf=pd_sf[pd_sf["hostname"]=="食べログ"]
    
    
    # general rankを general_rank_num に変更。S=1, A=2, B=3, L=4, F=5
    pd_sf["general_rank_num"]=pd_sf["general_rank_STRUCTURED"].map({"S":1,"A":2,"B":3,"L":4,"F":5})
    # 同じrestaurant_idで複数のデータがある場合、最もランクが低いものを選択
    pd_sf=pd_sf.sort_values(by=["restaurant_id","general_rank_num"],ascending=True)
    pd_sf=pd_sf.drop_duplicates(subset="restaurant_id",keep="first")
    pd_sf.drop(columns=["url"],inplace=True)

    #pseudo_time_series , pseudo_time_series_test に pd_sf を結合
    pseudo_time_series = pd.merge(pseudo_time_series,pd_sf,on="restaurant_id",how="left")
    
    return pseudo_time_series


def fix_structured_data(pseudo_time_series_train):
    float_list=["aggregateRating.ratingCount","aggregateRating.ratingValue", "aggregateRating.reviewCount", 
       "aggregateRating.bestRating", "aggregateRating.worstRating",  
       "interactionCount.went", 
       "interactionCount.wanttogo","paymentAccepted"]

    remove_list=["price","priceRange","servesCuisine","ReserveAction.result.name", "OrderAction.target.actionPlatform", 
        "OrderAction.target.inLanguage", "OrderAction.target.urlTemplate",]

    true_flag_list=["acceptsReservations"]

    pseudo_time_series_train[float_list]=pseudo_time_series_train[float_list].replace('',np.nan).astype(float)

    #acceptsReservationsはTrueか否かで分ける
    pseudo_time_series_train.loc[pseudo_time_series_train["acceptsReservations"]=="True","acceptsReservations"]=1
    pseudo_time_series_train.loc[pseudo_time_series_train["acceptsReservations"]!="True","acceptsReservations"]=0

    pseudo_time_series_train["acceptsReservations"]=pseudo_time_series_train["acceptsReservations"].astype(int)

    pseudo_time_series_train.drop(remove_list,axis=1,inplace=True)
    
    return pseudo_time_series_train