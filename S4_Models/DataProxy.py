from Tools.DataTools import load_data
import pandas as pd


class DataProxy:
    @staticmethod
    def get_train_valid_xy(data_link):
        df = load_data(data_link)
        df_u_train = df[['渔船ID']].drop_duplicates('渔船ID', keep='last').sample(frac=0.7)
        df_train = pd.merge(df, df_u_train, on=['渔船ID'])
        df_valid = pd.concat([df, df_train]).drop_duplicates(df.columns, keep=False)
        df_valid = df_valid.sort_values(['time']).drop_duplicates(['渔船ID'], keep='last')

        return (df_train[['x', 'y', '速度', '方向']].values, df_valid[['x', 'y', '速度', '方向']].values,
                df_train[['type']].values.ravel(), df_valid[['type']].values.ravel())

    @staticmethod
    def get_test_x(data_link):
        df = load_data(data_link)
        df = df.sort_values(['渔船ID', 'time']).drop_duplicates(['渔船ID'], keep='last')
        return df[['x', 'y', '速度', '方向']].values, df[['渔船ID']].values.ravel()
