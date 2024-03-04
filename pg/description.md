# make_timeseries_data.ipynb
### 最低限の情報（'original_id', 'available', 'closed_date', 'business_status')を元に、時系列データの骨組みを作成

###  - base_date（予測モデルを作成する基準日）の列を追加する。
* yyyymmdd より後日のデータは同じ行の説明変数に入れない
* target は翌月1日から三か月間

### 他のデータの内容は変わらず、2022年12月末日から2023年10月末日のデータを用意する。

###  closed_flg を追加する
###  closed_date 該当月以降はflg=1とする

### 予測対象flg(eval_period)を追加する
### base_date < closed_date は予測対象外とする

###  基準日を定めて、base_date 翌日から3カ月以内に、target が0->1に代わるものを予測する。

### closed_flg を3ヶ月前にずらしたものを予測（yyyymmdd=月末の翌日から3カ月以内にclosed_flgが1になる）
###  評価対象日のflag(eval_period)は、base_date が閉店日より前のもの

# add_features_from_tabelog_master.ipynb

### - tabelog のオリジナルデータを追加
* "name","latitude","longitude","prefecture","genres","openning_days"

### - 履歴系のデータは11月の断面データなので追加しない。
### - price_range, status系はダミー化
* price_range のNaNはNaNのままで良いか？
    * 一旦、−１にする。
* フリー記述は追加しない
* flg系を追加 
* base_dateがcreated_atよりも前の場合は eval_period を False にする

### add_features_from_V_GOOGLE.ipynb
* aggregate_review
* count

### add_features_from_V_RETTY.ipynb
* 下記は要ダミー化
    * infos.familiar.users.genge
    * infos.familiar.stars
    * infos.reserveEntireStore
    * infos.counterSeats.value
    * infos.privateRoom.value

# test_models.ipynb
### 2023年1月1日からと2023年4月から3ヶ月間の閉店について、学習
* "name"は規模に変更
* "genres"は三つのgenresをミックスしてダミー化
* "prefecture"もダミー化
* 2023年8月から
* AUC=0.724,  573939 restaurants
* AUC=0.813,  354547 restaurants
 
# make_nonbias_data.ipynb
* 3月に受領したデータを読み込みむ
* トレーニングデータの2023年10月分を読み込む。
* 2023年10月31日時点で、営業していて、1月までに閉店したお店にフラグをつける。
* 予測してみる。

# 2023年11月30日時点で閉店していないお店、2023年11月30日時点で開店していて

* 可能であれば、元マスターデータを、新マスターデータを使って、
* 2023年11月末時点で開店していたお店だけに絞る。

# 2023年1月-6月の閉店データで学習。12月-2月の閉店を予測
* AUC=0.60
# 2023年5月-10月の閉店データで学習。12月-2月の閉店を予測
* AUC=0.60
* 予測対象は、閉店した（掲載保留等除く）、close_flg が 12月~2月のお店とする
* AUC=0.61


# メディアは閉店した店舗が登場しにくくなる。
# 検索結果があるIDを過去に当てはめると、母集団（閉店が沢山より多い過去）が変わってしまう。
　例えば、

# 対象法として、あまり過去まで、検索結果を当てはめないようにする。
* 2023年8月から10月で学習
* AUC=0.61 のまま
* 2023年9月から1月で学習
* AUC=0.60