import sys;sys.path.append('../')
from S4_Models import GBDT
from S4_Models import XGBoost
from sklearn.model_selection import train_test_split
import DataLinkSet as DLSet
from sklearn.metrics import f1_score
from S4_Models.ModelProxy import ModelProxy
from S4_Models.DataProxy import DataProxy


def get_clf(choice):
    if choice == 'GBDT':
        return GBDT.generate_clf()
    elif choice == 'XGBoost':
        return XGBoost.generate_clf()


def main():
    model_choice = 'GBDT'
    # model_choice = 'XGBoost'

    # state clf
    model = ModelProxy(clf=get_clf(model_choice))

    # load train data
    train_x, valid_x, train_y, valid_y = DataProxy.get_train_valid_xy(DLSet.merge_train_link)
    # train model
    print('#train_data = {}'.format(train_y.shape))
    model.fit(train_x, train_y)
    model.save(DLSet.model_link % model_choice)

    # evaluate model
    pred_valid_y = model.predict(valid_x)
    result = f1_score(valid_y, pred_valid_y, average='macro')
    print('F1 = {}'.format(result))

    # predict
    model = ModelProxy(data_link=DLSet.model_link % model_choice)
    te_x, te_id = DataProxy.get_test_x(DLSet.merge_test_link)
    te_y = model.predict(te_x)

    with open(DLSet.result_link % model_choice, 'w') as f:
        for i in range(len(te_id)):
            f.write(str(te_id[i]) + ',' + te_y[i] + '\n')


if __name__ == '__main__':
    main()
