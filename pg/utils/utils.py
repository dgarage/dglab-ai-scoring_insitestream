import pandas as pd
import snowflake.connector
import matplotlib.pyplot as plt
# matplotlib で日本語を使うための設定
import matplotlib as mpl
from utils import *
import seaborn as sns
from scipy import stats

first_columns = ['acquisition_id', 'url_id', 'url', 'crawl_stamp', 'restaurant_id',
       'tabelog_name', 'tabelog_branch_name', 'tabelog_address',
       'tabelog_telephone', 'tabelog_reservation_telephone', 'name_rank',
       'name_rank_system', 'name_ratio.ratio',
       'name_ratio.master_normalized_name', 'name_ratio.normalized_name',
       'name_ratio.include', 'name_ratio.diff', 'name_only_ratio.ratio',
       'name_only_ratio.master_normalized_name',
       'name_only_ratio.normalized_name', 'name_only_ratio.include',
       'name_only_ratio.diff', 'address_rank', 'address_rank_system',
       'postal.ratio', 'postal.master_value', 'postal.value', 'postal.diff',
       'region.ratio', 'region.master_value', 'region.value', 'region.diff',
       'street_address.ratio', 'street_address.master_value',
       'street_address.value', 'street_address.diff', 'building_name.ratio',
       'building_name.master_value', 'building_name.value',
       'building_name.diff', 'telephone_rank', 'telephone_rank_system',
       'telephone_ratio.ratio', 'telephone_ratio.master_value',
       'telephone_ratio.value', 'telephone_ratio.diff',
       'google_search_title_ratio']


second_columns = ['restaurant_id','url','geodesic', 'name_rank',
                  'address_rank','telephone_rank','name', 'alternateName',
       'description', 'telephone', 'faxNumber', 'website_url', 'logo',
       'address', 'address.addressCountry', 'address.postalCode',
       'address.addressRegion', 'address.addressLocality',
       'address.streetAddress', 'map', 'areaServed', 'geo.latitude',
       'geo.longitude', 'aggregateRating.ratingCount',
       'aggregateRating.ratingValue', 'aggregateRating.reviewCount',
       'aggregateRating.bestRating', 'aggregateRating.worstRating', 'price',
       'priceRange', 'servesCuisine', 'image_1', 'image_2', 'image_3',
       'image_4', 'image_5', 'interactionCount.went',
       'interactionCount.wanttogo',
       'menu', 'openingHours', 'openingHours_1',
       'openingHours_2', 'openingHours_3', 'openingHours_4', 'openingHours_5',
       'openingHours_6', 'openingHours_7', 'openingHoursSpecification.opens',
       'openingHoursSpecification.closes',
       'openingHoursSpecification.dayOfWeek', 'paymentAccepted',
       'acceptsReservations', 'potentialAction', 'OrderAction.deliveryMethod',
       'ReserveAction.result.name', 'OrderAction.target.actionPlatform',
       'OrderAction.target.inLanguage', 'OrderAction.target.urlTemplate']

hotpepper_columns1 = ['acquisition_id', 'url_id', 'url', 'crawl_stamp', 'restaurant_id',
       'tabelog_name', 'tabelog_branch_name', 'tabelog_address',
       'tabelog_telephone', 'tabelog_reservation_telephone', 'name_rank',
       'name_rank_system', 'name_ratio.ratio',
       'name_ratio.master_normalized_name', 'name_ratio.normalized_name',
       'name_ratio.include', 'name_ratio.diff', 'name_only_ratio.ratio',
       'name_only_ratio.master_normalized_name',
       'name_only_ratio.normalized_name', 'name_only_ratio.include',
       'name_only_ratio.diff', 'address_rank', 'address_rank_system',
       'postal.ratio', 'postal.master_value', 'postal.value', 'postal.diff',
       'region.ratio', 'region.master_value', 'region.value', 'region.diff',
       'street_address.ratio', 'street_address.master_value',
       'street_address.value', 'street_address.diff', 'building_name.ratio',
       'building_name.master_value', 'building_name.value',
       'building_name.diff', 'telephone_rank', 'telephone_rank_system',
       'telephone_ratio.ratio', 'telephone_ratio.master_value',
       'telephone_ratio.value', 'telephone_ratio.diff',
       'google_search_title_ratio', 'geodesic']


hotpepper_columns2 = ['acquisition_id', 'url_id', 'url', 'crawl_stamp', 'restaurant_id',
       'infos.name', 'infos.aggregateRating', 'infos.ratingReview',
       'infos.usePointsForOnlineReservations', 'infos.priceRange1',
       'infos.priceRange2', 'infos.genre', 'infos.area',
       'infos.satisfaction.percentage1', 'infos.satisfaction.badge1',
       'infos.satisfaction.percentage2', 'infos.satisfaction.badge2',
       'infos.satisfaction.percentage3', 'infos.satisfaction.badge3',
       'infos.satisfaction.percentage4', 'infos.satisfaction.badge4',
       'infos.useScenes1', 'infos.useScenes2', 'infos.useScenes3',
       'infos.useScenes4', 'infos.useScenes5', 'infos.useScenes6',
       'infos.useScenes7', 'infos.useScenes8', 'infos.useScenes9',
       'infos.atmosphere1.label', 'infos.atmosphere1.value',
       'infos.atmosphere2.label', 'infos.atmosphere2.value',
       'infos.atmosphere3.label', 'infos.atmosphere3.value', 'infos.address',
       'infos.access', 'infos.telephone', 'infos.openingHours',
       'infos.receptionHours', 'infos.onlineReceptionHours',
       'infos.regularHoliday', 'infos.useablePoints', 'infos.smartPayment',
       'infos.creditCard.comment', 'infos.creditCard.value',
       'infos.electronicMoney.comment', 'infos.electronicMoney.value',
       'infos.qrCodePayment.comment', 'infos.qrCodePayment.value',
       'infos.feeNotes', 'infos.infectiousDiseaseCountermeasures.employee',
       'infos.infectiousDiseaseCountermeasures.restaurant',
       'infos.infectiousDiseaseCountermeasures.customer', 'infos.smoking',
       'infos.smokingRoom', 'infos.capacity', 'infos.maxBanquetCapacity',
       'infos.privateRoom.comment', 'infos.privateRoom.value',
       'infos.tatamiRoom.value', 'infos.tatamiRoom.comment',
       'infos.horigotatsu.comment', 'infos.counterSeats', 'infos.sofaSeats',
       'infos.terraceSeats', 'infos.reserveEntireStore', 'infos.wifi',
       'infos.barrierFree', 'infos.parking', 'infos.otherEquipment',
       'infos.bottomlessCup', 'infos.smorgasbord', 'infos.alcohol',
       'infos.children', 'infos.weddingParty', 'infos.storeFeatures',
       'infos.memo', 'infos.fanCount', 'infos.couponCount',
       'infos.onlineReservation', 'infos.ownerRegistration', 'infos.ticket']

hotpepper_columns3 = ['acquisition_id', 'url_id', 'url', 'crawl_stamp', 'restaurant_id',
       'photos.allCount', 'photos.appearanceCount', 'photos.drinkCount',
       'photos.foodCount', 'photos.otherCount', 'photos.postCount',
       'menus.courseListCount', 'menus.courseListModified',
       'menus.dishListCount', 'menus.dishListModified', 'menus.drinkListCount',
       'menus.drinkListModified', 'menus.lunchListCount',
       'menus.lunchListModified', 'menus.takeoutListCount',
       'menus.takeoutListModified', 'infos.recentReviewer1.datePublished',
       'infos.recentReviewer1.dateVisited', 'infos.recentReviewer1.gender',
       'infos.recentReviewer1.generation',
       'infos.recentReviewer2.datePublished',
       'infos.recentReviewer2.dateVisited', 'infos.recentReviewer2.gender',
       'infos.recentReviewer2.generation',
       'infos.recentReviewer3.datePublished',
       'infos.recentReviewer3.dateVisited', 'infos.recentReviewer3.gender',
       'infos.recentReviewer3.generation',
       'infos.recentReviewer4.datePublished',
       'infos.recentReviewer4.dateVisited', 'infos.recentReviewer4.gender',
       'infos.recentReviewer4.generation',
       'infos.recentReviewer5.datePublished',
       'infos.recentReviewer5.dateVisited', 'infos.recentReviewer5.gender',
       'infos.recentReviewer5.generation', 'infos.reviewTags.count1',
       'infos.reviewTags.count2', 'infos.reviewTags.count3',
       'infos.reviewTags.count4', 'infos.reviewTags.count5',
       'infos.reviewTags.count6', 'infos.reviewTags.count7',
       'infos.reviewTags.count8', 'infos.reviewTags.count9', 'infos.latitude',
       'infos.longitude']

# snowflakeからデータを取得
def read_data_from_snowflake(table_name,col=""):
    ctx = snowflake.connector.connect(
        user='DIGITALGARAGE',
        password='scZ0eN*mTPNY',
        account='DNB54940'
    )

    cs = ctx.cursor()
    try:
        print('select '+col+' from KRAKEN_DG.KRAKEN_USER.'+table_name)
        cs.execute('select '+col+' from KRAKEN_DG.KRAKEN_USER.'+table_name)
        pd_sf = cs.fetch_pandas_all()


    finally:
        cs.close()
    ctx.close()
    
    return pd_sf

def read_table(data_folder,table_name,columns=first_columns,read_snowflake=False,col=""):
    if read_snowflake == True:
        pd_sf = read_data_from_snowflake(table_name,col)
        pd_sf.to_pickle(data_folder+'/'+table_name+'.pkl')
    
    if(columns=="all"):
        pd_sf = pd.read_pickle(data_folder+'/'+table_name+'.pkl')
    else:
        pd_sf = pd.read_pickle(data_folder+'/'+table_name+'.pkl')[columns]
    
    return pd_sf

def get_restaurant_master(data_folder):
    return pd.read_pickle(data_folder+'/V_RESTAURANT_NAME_MASTER.pkl')

#型をエクセルと同じにする
def align_columns(pd_sf,data_folder,table):
    table_profiles = data_folder+'飲食データ収集基盤_TBL List.xlsx'
    table_profiles = pd.read_excel(table_profiles,sheet_name=table,skiprows=4)
    table_profiles_filter=table_profiles[["Physical Column Name (snowflake)","DATA_TYPE"]]

    for n, col in enumerate(pd_sf.columns):
        print(col)
        dtype=table_profiles_filter[table_profiles_filter["Physical Column Name (snowflake)"]==col]["DATA_TYPE"].values[0]
        pd_sf[col] = pd_sf[col].astype(dtype)   

    return pd_sf

# 掲載状態を紐づけて返却
def merge_name_master(pd_sf,V_RESTAURANT_NAME_MASTER,master_columns = ['restaurant_id','business_status', 'closed_date','available']):
    return pd.merge(pd_sf,V_RESTAURANT_NAME_MASTER[master_columns], on=['restaurant_id'], how='left') 

# urlをkeyにmeadia を紐づけて返却
def merge_meadia(pd_sf,V_RANKING):
    return pd.merge(pd_sf,V_RANKING, on=['url'], how='left') 

def get_stats(df,colum='crawl_stamp'):
    return df[colum].min(),df[colum].max(),df[colum].mean(),df[colum].mean()


def get_freq_dist(pd_sf,columns=first_columns):
    # prepare the dataframe
    df = pd.DataFrame(columns=["name","num of entries","min","max","mean","std","idmax"])

    for c in columns:
        # 全体のユニーク数の分布
        rid_count = pd_sf[c].value_counts()
        
        try:
            # pushback results into the dataframe
            df = df.append({"name":c,"num of entries":len(rid_count),"min":rid_count.min(),"max":rid_count.max(),"mean":rid_count.mean(),"std":rid_count.std(),"idmax":rid_count.idxmax()},ignore_index=True)
        except:
            df = df.append({"name":c,"num of entries":len(rid_count),"min":rid_count.min(),"max":rid_count.max(),"mean":rid_count.mean(),"std":rid_count.std(),"idmax":"nan"},ignore_index=True)
    
    return df

# 頻度 Top50 を集計
def close_ratio_top20(pd_sf,column):

    pd_sf[column].value_counts().to_frame().head(20).plot(kind="bar")
    # title of the plot
    plt.title("Frequancy of "+column+" for all")

    pd_sf[pd_sf["business_status"]=="閉店"][column].value_counts().to_frame().head(20).plot(kind="bar")
    # title of the plot
    plt.title("Frequancy of "+column+" for 閉店")

    # more than 10 same branch names in the tabelog_branch_name
    bname = pd_sf[column].value_counts().to_frame()[pd_sf[column].value_counts()>10]
    
    # ratio of 閉店 and non閉店
    ratio = (pd_sf[pd_sf["business_status"]=="閉店"][column].value_counts()/pd_sf[column].value_counts())
    print(ratio)
    
    print("plotting")
    # ratio of 閉店 and non閉店 for more than 10 same branch names in the tabelog_branch_name
    # merge bname and ratio
    bname = bname.merge(ratio,left_index=True,right_index=True) 

    return bname.sort_values(by=[column+"_y",column+"_x"],ascending=False).rename(columns={column+"_x":"number of 閉店",column+"_y":"閉店 ratio"}).head(20)

def rank_by_hostname(pd_sf,column):

    # calculate ratio of each rank by host name
    num = pd_sf[["hostname",column]].value_counts().unstack() 

    # sum of each row
    num = num.div(num.sum(axis=1), axis=0)

    # S,A,B,L,F,N の順に並べ替え
    num = num.reindex(columns=["S","A","B","L","F","N"])

    num.sort_values(by="S",ascending=False).plot(kind="bar",stacked=True)
    # put the legend outside of the plot
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.title("Ratio of each orginal "+column+" by host name")
    
    return num

# check distance distribution 
def ratio_by_hostname_rank(pd_sf,column,rank="name_rank"):
    
    num_hostname = pd_sf[["hostname",column]].groupby("hostname").agg(["mean","std","max","min"])
    num_rank = pd_sf[[rank,column]].groupby(rank).agg(["mean","std","max","min"])
    
    # plot by each hostname
    # wider plot
    group_means = pd_sf.groupby('hostname')[column].mean().reset_index()
    # 平均値でソート
    sorted_groups = group_means.sort_values(column, ascending=False)
    # ソートされた順序で元のDataFrameを並び替え
    sorted_df = pd_sf.set_index('hostname').loc[sorted_groups['hostname']].reset_index()
    plt.figure(figsize=(30,10))
    sns.boxplot(x='hostname', y=column, data=sorted_df, color='lightgray', linewidth=2.5)
    # rotate x-axis label
    plt.xticks(rotation=90)
    # make larger font size of x-axis label
    plt.xticks(fontsize=20)
    plt.ｙticks(fontsize=20)
    # make larger title
    plt.xlabel(rank,fontsize=20)
    plt.ylabel(column,fontsize=20)
    plt.xlabel("hostname",fontsize=20)
    
    # plot by each original rank
    group_means = pd_sf.groupby(rank)[column].mean().reset_index()
    # 平均値でソート
    sorted_groups = group_means.sort_values(column, ascending=False)
    # ソートされた順序で元のDataFrameを並び替え
    sorted_df = pd_sf.set_index(rank).loc[sorted_groups[rank]].reset_index()
    plt.figure(figsize=(30,10))
    sns.boxplot(x=rank, y=column, data=sorted_df, color='lightgray', linewidth=2.5)
    # rotate x-axis label
    # plt.xticks(rotation=90)
    # make larger font size of axis label
    plt.xticks(fontsize=20)
    plt.ｙticks(fontsize=20)
    # make larger title
    plt.xlabel(rank,fontsize=20)
    plt.ylabel(column,fontsize=20)
    
    return num_hostname,num_rank


def make_restautant_number_plots(ps_df,column="region.master_value"):
    # Number of appearnce of name_only_ratio.diff, top 20
    region_hostname_p=pd_sf[[column,"hostname"]].value_counts().unstack().fillna(0)
    
    
    fname=['食べログ', 'AutoReserve', 'ヒトサラ', 'ぐるなび', 'エキテン',
       'Line place', 'Yahoo!ロコ', 'TripAdvisor', 'Retty', 'ホットペッパー']
    
    region_hostname_p=region_hostname_p[fname]
    
    # sort columns  東京都's value
    region_hostname_p=region_hostname_p.T.sort_values("東京都",ascending=False).T
    region_hostname_p=region_hostname_p.sort_values("食べログ",ascending=False).T
    region_hostname_p=region_hostname_p.astype(int)


    
    
    # 2 dimention plot for region_hostname_p
    # make wider plot
    plt.figure(figsize=(30,10))
    sns.heatmap(region_hostname_p.head(10))
    # color scale direction convert

    # make larger font size of axis label
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel("Prefecture",fontsize=20)
    plt.ylabel("# of restaurants",fontsize=20)
    plt.title("All regions")

    # plot entires for each hostname
    # wider plot
    # xaxis region, each line is hostname
    plt.figure(figsize=(30,10))
    plot = sns.lineplot(data=region_hostname_p.head(10).T, dashes=False, markers=True)

    # rotate x-axis label
    plt.xticks(rotation=90)
    # make larger font size of axis label
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # make larger legend
    plt.legend(fontsize=30)

    plt.xlabel("Prefecture",fontsize=20)
    plt.ylabel("# of restaurants",fontsize=20)
    plt.title("All regions")

    # 2 dimention plot for region_hostname_p
    plt.figure(figsize=(30,10))
    plot = sns.lineplot(data=region_hostname_p.head(10).T.tail(40), dashes=False, markers=True)

    # rotate x-axis label
    plt.xticks(rotation=90)
    # make larger font size of axis label
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # make larger legend
    plt.legend(fontsize=30)

    plt.xlabel("Prefecture",fontsize=20)
    plt.ylabel("# of restaurants",fontsize=20)
    # make larger title
    plt.title("Without top sevens",fontsize=40)
    

def show_diff(pd_sf,column="street_address.diff"):
    diff_pref = pd_sf[column].value_counts().to_frame()
    display(diff_pref.head(6).tail(5))
    display(diff_pref[diff_pref.index.str.contains("\[\]")==False])
    
    
def check_distribusion_hitosara(pd_sf,c):
    all_data=pd_sf[pd_sf[c]!=''][c].dropna().astype(float)
    close_data=pd_sf[(pd_sf[c]!='')
        &(pd_sf["business_status"]=="閉店")][c].dropna().astype(float)

    # 同じ値は横に並べる
    plt.hist([all_data.values, close_data.values], bins=20, density=True, histtype='bar', color=['blue', 'red'], label=['all', 'close'])
    # all と close の分布の違いを検定

    plt.xlabel("Density distribution of "+c)
    plt.ylabel("Density")
    plt.title(c)
    plt.legend()
    plt.savefig("../results/png/hotpepper/"+c+".png")
    
    if(len(all_data)!=0 and len(close_data)!=0):
        p_value=stats.ks_2samp(np.array(all_data),np.array(close_data)).pvalue
        print(p_value)
    else:
        p_value=np.nan
        print("no data")
    
    plt.clf()
    
    return c,all_data.shape[0],close_data.shape[0],all_data.mean(),close_data.mean(),all_data.std(),close_data.std(),p_value