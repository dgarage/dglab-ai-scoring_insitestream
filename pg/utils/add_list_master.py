add_list_master=['name',"prefecture_name",'head_branch','opened_on', 'created_at','genre_first_name', 
          'genre_second_name','genre_third_name','northlatitude', 'eastlongitude',
          'net_reservation_flg','price_range_lunch_owner','price_range_dinner_owner', 
          'price_range_lunch_user','price_range_dinner_user',
          # 20240312追加
          'all_photo_count','photo_food_count','photo_drink_count',
          'photo_interior_count','photo_exterior_count','official_info_flg','days_from_lastvisit']

add_list_google=["restaurant_id","overview.aggregateRating","overview.reviewCount","general_rank_GOOGLE"]

add_list_retty=["restaurant_id",'infos.isOfficial', 
       'infos.familiar.users.count', 'infos.familiar.stars',
       'general_rank_RETTY', 
       'infos.onlineReservation_RETTY', 'infos.access.transferTime1','infos.access.transferTime2','infos.access.transferTime3',
       'infos.numberOfSeats.value', 'infos.counterSeats.value',
       'infos.privateRoom.value', 'infos.updateInfo.firstReviewDate',
       'infos.wantToGo', 'infos.updateInfo.lastReviewDate',
       'infos.went', 'infos.rateByWent', 'infos.excellent', 'infos.coupon',
       'infos.photoCount.Photograph', 'infos.photoCount.Cooking',
       'infos.photoCount.Interior', 'infos.photoCount.Exterior',
       'infos.photoCount.Menu', 'infos.reviewCount.Review',
       'infos.reviewCount.Lunch', 'infos.reviewCount.Dinner',
       'menus.course_list_count',
       'menus.dishes_list_count', 'menus.lunch_list_count',
       'menus.drink_list_count', 'menus.takeout_list_count']

add_list_hotpepper=['restaurant_id','general_rank_HOTPEPPER','infos.aggregateRating','infos.ratingReview',
          'infos.satisfaction.percentage1','infos.satisfaction.percentage2','infos.satisfaction.percentage3',
          'infos.couponCount','infos.onlineReservation_HOTPEPPER','infos.ownerRegistration','infos.ticket',
          'photos.allCount','photos.appearanceCount','photos.drinkCount','photos.foodCount','photos.otherCount',
          'photos.postCount','menus.courseListCount']

add_list_tripadvisor=["restaurant_id",'general_rank_TRIPADVISOR','infos.aggregateRating_TRIPADVISOR',"infos.reviewCount","infos.qa_count"]


add_list_hitosara=["restaurant_id","general_rank_HITOSARA","infos.premium","infos.onlineReservation_HITOSARA"]