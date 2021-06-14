import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import lightgbm as lgb

# Load Data
item_categories = pd.read_csv("item_categories.csv")
items = pd.read_csv("items.csv")
train_df = pd.read_csv("sales_train.csv")
sample_sub = pd.read_csv("sample_submission.csv")
shops = pd.read_csv("shops.csv")
test_df = pd.read_csv("test.csv")

# Data Cleaning
train_df[train_df["item_cnt_day"] < 0] = 0
train_df = train_df[(train_df["item_price"] < 100000) & (train_df["item_price"] > 0)]
train_df = train_df[train_df["item_cnt_day"] < 1001]
X = train_df.copy()

# Summarize sales in month
X = X.groupby(["date_block_num", "shop_id", "item_id"], as_index=False).sum()
X = X.rename(columns={'item_cnt_day':'item_cnt_month'})

# Transform time series 0 ~ 33
X = X.pivot_table(index=["shop_id", "item_id"], columns="date_block_num", values="item_cnt_month", fill_value=0)
X.reset_index(inplace=True)

# Prepare training data
X_train = np.array(X.values[:, 0:-1])
Y_train = np.array(X.values[:, -1])
print(X_train.shape)
print(Y_train.shape)

# LightGBM
params = {
    'objective': 'rmse',
    'metric': 'rmse',
    'num_leaves': 1023,
    'min_data_in_leaf':10,
    'feature_fraction':0.7,
    'learning_rate': 0.01,
    'num_rounds': 1000,
    'early_stopping_rounds': 30,
    'seed': 1
}
lgb_train = lgb.Dataset(X_train[:-300], Y_train[:-300])
lgb_valid = lgb.Dataset(X_train[-300:], Y_train[-300:])
model_lgb = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_valid])
model_lgb.save_model('model_lgb.txt')

# Predict Test
model = model_lgb
id_list = []
pred_list = []
for idx in range(len(test_df)):
    if idx % 1000 == 0:
        print(idx)
    row_id = test_df.iloc[idx]["ID"]
    shop_id = test_df.iloc[idx]["shop_id"]
    item_id = test_df.iloc[idx]["item_id"]
    if X[(X["shop_id"]==shop_id) & (X["item_id"]==item_id)].empty:
        id_list.append(row_id)
        pred_list.append(0.0)
    else:
        temp = X[(X["shop_id"]==shop_id) & (X["item_id"]==item_id)]
        historys = np.hstack([temp.values[:,0], temp.values[:,1], temp.values[0,3:]])
        historys = historys[np.newaxis, :]
        pred = model.predict(historys)
        id_list.append(row_id)
        if pred[0] < 0:
            pred_list.append(0.0)
        elif pred[0] > 20:
            pred_list.append(20.0)
        else:
            pred_list.append(pred[0])

# Output Submission
submission = pd.DataFrame()
submission['ID'] = id_list
submission['item_cnt_month'] = pred_list
submission.to_csv('submission.csv', index=False)