from Tools.DataTools import load_data
import pandas as pd


def calculate_loc_dis(df, choice):
    df_loc = df.sort_values([choice])[['渔船ID', choice]]
    df_min_loc = df_loc.drop_duplicates(['渔船ID'], keep='first')
    df_max_loc = df_loc.drop_duplicates(['渔船ID'], keep='last')
    df_min_loc.columns = ['渔船ID', 'min']
    df_max_loc.columns = ['渔船ID', 'max']
    df_dis = pd.merge(df_min_loc, df_max_loc, on=['渔船ID'])
    df_dis['dis_' + choice] = df_dis['max'] - df_dis['min']
    return df_dis[['渔船ID', 'dis_' + choice]]


def generate_dis(df):
    df_dis_x = calculate_loc_dis(df, 'x')
    df_dis_y = calculate_loc_dis(df, 'y')
    df_dis = pd.merge(df_dis_x, df_dis_y, on=['渔船ID'])
    df_dis['dis_n1'] = df_dis['dis_x'] + df_dis['dis_y']
    df_dis['dis_n2'] = df_dis['dis_x'] ** 2 + df_dis['dis_y'] ** 2
    df_dis['dis_nwq'] = df_dis.apply(lambda x: max(x.dis_x, x.dis_y), axis=1)
    return df_dis


class DataProxy:
    @staticmethod
    def get_train_valid_xy(data_link):
        df = load_data(data_link)

        # calculate the loc dis
        df_dis = generate_dis(df)
        df = pd.merge(df, df_dis, on=['渔船ID'])

        # split to train and valid
        df_u_train = df[['渔船ID']].drop_duplicates('渔船ID', keep='last').sample(frac=0.7)
        df_train = pd.merge(df, df_u_train, on=['渔船ID'])
        df_valid = pd.concat([df, df_train]).drop_duplicates(df.columns, keep=False)
        df_valid = df_valid.sort_values(['time']).drop_duplicates(['渔船ID'], keep='last')

        return (df_train[['x', 'y', '速度', '方向']].values, df_valid[['x', 'y', '速度', '方向']].values,
                df_train[['type']].values.ravel(), df_valid[['type']].values.ravel())

    @staticmethod
    def get_test_x(data_link):
        df = load_data(data_link)

        # calculate the loc dis
        df_dis = generate_dis(df)
        df = pd.merge(df, df_dis, on=['渔船ID'])

        df = df.sort_values(['渔船ID', 'time']).drop_duplicates(['渔船ID'], keep='last')
        return df[['x', 'y', '速度', '方向']].values, df[['渔船ID']].values.ravel()
