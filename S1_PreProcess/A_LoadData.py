import sys; sys.path.append('../')
import pathlib
from Tools.DataTools import *
import DataLinkSet as DLSet


# merge all data to one csv
def merge(data_link, store_link):
    data_path_set = list(pathlib.Path(data_link).iterdir())
    n_user = len(data_path_set)
    # visit all data
    user_id = 0
    df_all = None
    for data_path in data_path_set:
        # load the data
        df_temp = load_data(data_path)

        # concat
        if user_id == 0:
            df_all = df_temp
        else:
            df_all = pd.concat([df_all, df_temp])

        user_id += 1
        if user_id % 300 == 0:
            print('---------------- {} users data had been merge ----------------'.format(user_id))
    df_all = df_all.sort_values(['time'])

    print('#user = {}'.format(n_user))
    df_all.to_csv(store_link, index=None)


# reflect label with most num
def cal_label_distribute(data_link):
    df = load_data(data_link)
    df = df.drop_duplicates(['渔船ID'], keep='last')
    df['type_count'] = df.groupby(['type']).cumcount() + 1
    df = df.drop_duplicates(['type'], keep='last')
    df['type_count'] /= 7000
    print(df)


def fill_by_most(data_link, store_link):
    df = load_data(data_link)
    df = df.sort_values(['渔船ID'])
    df = df.drop_duplicates(['渔船ID'], keep='last')
    df['type'] = '拖网'
    df.to_csv(store_link, columns=['渔船ID', 'type'], header=None, index=False)


def main():
    # merge(DLSet.raw_train_link, DLSet.merge_train_link)
    merge(DLSet.raw_test_link, DLSet.merge_test_link)

    # cal_label_distribute(DLSet.merge_train_link)
    # fill_by_most(DLSet.merge_test_link, DLSet.result_link)


if __name__ == '__main__':
    main()
