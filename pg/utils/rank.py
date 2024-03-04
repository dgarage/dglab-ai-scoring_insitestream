import pandas as pd
import numpy as np
class Ranker:
    @classmethod
    def add_general_rank(cls, df:pd.DataFrame) -> pd.DataFrame:
        """
        Add "general_rank" column to df.
        df: DataFrame of restaurant info. Must have name_rank, address_rank, telephone_rank
        return: DataFrame with "general_rank" column
        """
        original_cols = df.columns.tolist()
        # apply F, N, L rank
        df = cls.__f_l_n2_detector(df)
        
        # apply S,A,B rank to restaurants with empty("") f_l_n_rank
        s_a_b_detector_df = cls.__s_a_b_detector(df[df["f_l_n_rank"]==""])
        f_l_n_detector_df = df[~(df["f_l_n_rank"]=="")]
        df = pd.concat([s_a_b_detector_df, f_l_n_detector_df])
        # replace NaN with empty string in "s_a_b_rank" column
        df["s_a_b_rank"] = df["s_a_b_rank"].fillna("")
        del s_a_b_detector_df, f_l_n_detector_df

        # sort index
        df = df.sort_index()

        #make a general column that reflect results f_rank, n_rank, l_rank, s_a_b rank
        df = df.assign(general_rank = df["f_l_n_rank"] + df["s_a_b_rank"])
        
        # get needed columns
        df = df[original_cols+["general_rank"]]
        return df

    def __f_l_n2_detector(df):
        """
        detect f in telephone_rank, address_rank, name_rank, L in telephone_rank, address_rank, name_rank, and 2Ns in telephone_rank, address_rank, name_rank and alphabetically rank them as F, L, N respectively
        return: F, L, N ranks
        """

        #when telephone_rank or address_rank or name_rank has F
        condition_f = [(df["telephone_rank"].isin(["F"])) | (df["address_rank"].isin(["F"])) | (df["name_rank"].isin(["F"]))]
        choice_f = ["F"]
        df["f_rank"] = np.select(condition_f, choice_f, default="")

        #when 2Ns are detected in telephone_rank or address_rank or name_rank
        condition_2n_tel_add = [((df["telephone_rank"].isin(["N"])) & (df["address_rank"].isin(["N"]))) | ((df["telephone_rank"].isin(["N"])) & (df["name_rank"].isin(["N"]))) | ((df["address_rank"].isin(["N"])) & (df["name_rank"].isin(["N"])))] 
        choice_2n_tel_add = ["N"]
        df["n_rank"] = np.select(condition_2n_tel_add, choice_2n_tel_add, default="")
        
        #when telephone_rank or address_rank or name_rank has L
        condition_l = [(df["telephone_rank"].isin(["L"])) | (df["address_rank"].isin(["L"])) | (df["name_rank"].isin(["L"]))]
        choice_l = ["L"]
        df["l_rank"] = np.select(condition_l, choice_l, default="")

        #make a general f_n_l column that reflect results f_rank, n_rank, l_rank
        f_n_l_df = df.assign(f_l_n_rank = df["f_rank"] + df["n_rank"] + df["l_rank"])
        # delete 2nd letter if any
        f_n_l_df["f_l_n_rank"] = f_n_l_df["f_l_n_rank"].map(lambda x: x[0] if len(x)>0 else x)

        return f_n_l_df

    def __s_a_b_detector(df):
        """
        detect S, A, B rank based on ranks given to telephone_rank, address_rank, name_rank
        all rows in df must not have other rank than S, A, B
        return: S, A, B ranks
        """
        # Total of 1 or 0 N rank
        df["n_rank_count"] = df["telephone_rank"].isin(["N"]).astype(int) + df["address_rank"].isin(["N"]).astype(int) + df["name_rank"].isin(["N"]).astype(int)
        assert df["n_rank_count"].max() <= 1, "df must have n_rank_count less than 2"
        assert df["n_rank_count"].min() >= 0, "df must have at least 0 n_rank_count"
        condition = [(df["n_rank_count"]==1), (df["n_rank_count"]==0)]
        choice = [2, 3]
        df["denominator"] = np.select(condition, choice, default=-1)

        #when telephonr_rank has S
        condition_tel = [(df["telephone_rank"].isin(["S"]))]
        choice_tel = [4]
        df["tel_numerator"] = np.select(condition_tel, choice_tel, default=0) # rank N = 0

        #when address_rank has S, A, B, (L, N)
        condition_add = [(df["address_rank"].isin(["S"])), (df["address_rank"].isin(["A"])), (df["address_rank"].isin(["B"]))]
        choice_add = [4, 3, 2]
        df["add_numerator"] = np.select(condition_add, choice_add, default=0) # rank N = 0

        #when name_rank has S, A, B, (N, L, F)
        condition_name = [(df["name_rank"].isin(["S"])), (df["name_rank"].isin(["A"])), (df["name_rank"].isin(["B"]))]
        choice_name = [4, 3, 2]
        df["name_numerator"] = np.select(condition_name, choice_name, default=0) # rank N = 0

        #create a new column that has a sum of tel_numerator, add_numerator, name_numerator
        df["general_numerator"] = df["tel_numerator"] + df["add_numerator"] + df["name_numerator"]

        #create a new column that has a result of deviding the general_numerator value by denominator_value
        df["division_result_value"] = df["general_numerator"] / df["denominator"]

        #creaye a new column that has a rank in alphabet based on the result of division_result_value
        condition_result = [(df["division_result_value"]==4), ((df["division_result_value"]<4) & (df["division_result_value"]>=3)), ((df["division_result_value"]<3) & (df["division_result_value"]>=2))]
        choice_result = ["S", "A", "B"]
        df["s_a_b_rank"] = np.select(condition_result, choice_result, default="?")
        assert df[df["s_a_b_rank"]=="?"].shape[0] == 0, f"s_a_b_rank could not be set properly for {df[df['s_a_b_rank']=='?'].shape[0]} rows"

        return df