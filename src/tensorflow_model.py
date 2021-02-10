#%%

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
from keras.utils import plot_model
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import numpy as np
import pandas as pd

items = pd.read_csv(r"..\Data\Raw\items.csv", delimiter=",")
print(items.head())
print(items.shape)
items = items.drop_duplicates(subset=['item_name'], keep=False)
print(items.shape)

#%%

test = pd.read_csv(r"..\Data\Raw\test.csv", delimiter=",")
print(test.head())
print(test['shop_id'].nunique())
print(test['ID'].nunique())

#%%

sales = pd.read_csv(r"..\Data\Raw\sales_train.csv", delimiter=",")
print(sales.head())
print(sales.shape)
sales = sales.drop_duplicates(subset=['date', 'shop_id', 'item_id'], keep=False)
print(sales.shape)
sales = sales[sales.item_cnt_day<1000]
sales = sales[sales.item_price<60000]
sales = sales[sales.item_price>0]
print(sales.shape)
sales['revenue'] = sales['item_cnt_day'] * sales['item_price']
sales_monthly = sales.groupby(["date_block_num", "shop_id", "item_id"]).agg({'item_cnt_day': ['sum'], 'item_price':['median'], 'revenue':['sum']})
print(sales_monthly.head(20))
sales_monthly.sample(10)

#%%

#join to test set to get only necessary IDs
month_range = range(1, 34)
month_df = pd.DataFrame(month_range)

month_df.rename(columns={0 :'date_block_num'}, inplace=True )
print(month_df.head())
month_df['temp'] = 1
test['temp'] = 1
test_monthly = test.merge(month_df, on=['temp'])
test_monthly.drop(['temp'], axis=1, inplace=True)
whole_df = test_monthly.merge(sales_monthly, how='left', on=['date_block_num', 'item_id', 'shop_id'])
whole_df.sample(10)
print(whole_df.shape)

#%%

categories = pd.read_csv(r"..\Data\Raw\item_categories.csv", delimiter=",")
print(categories.head())


#%%

print(items.shape)
print(categories.shape)
item_cats = items.merge(categories, how='left', on=['item_category_id'])
print(item_cats.shape)
print(item_cats.head())
item_cats.drop(['item_category_name', 'item_name'], axis=1, inplace=True)
print(item_cats.head())
print(item_cats['item_id'].nunique())
print(item_cats.shape)
whole_df = whole_df.merge(item_cats, how='left', on=['item_id'])
whole_df.sample(10)

#%%

whole_df.fillna(0, inplace=True)
list(whole_df.columns)
whole_df.rename(columns={('item_cnt_day', 'sum'):'total_count', ('item_price', 'median'): 'median_price', ('revenue', 'sum'):'total_revenue'}, inplace=True )
whole_df['total_count'].clip(0,20)
print(whole_df.sample(10))
whole_df.to_csv('../Data/Processed/pre_normalised_whole_df.csv', index=False)

#%%

whole_df.sort_values(by=['ID', 'date_block_num'])
whole_df.drop(['ID'], axis=1, inplace=True)
whole_df = whole_df[['total_count', 'date_block_num', 'median_price', 'total_revenue', 'shop_id', 'item_id', 'item_category_id']]
print(whole_df.head(34))
whole_df_arr = whole_df.values
print(whole_df_arr.shape)
whole_df_arr = whole_df_arr.reshape(214200, 33, 7)
print(whole_df_arr)#change back to 214200


#%%

#prepare training and validation sets
train_arr = whole_df_arr[:, 26:]
val_arr = whole_df_arr[:, 26:]
print(train_arr.shape)
print(val_arr.shape)
train_arr_first_test = train_arr[:, 19:25]
print(train_arr_first_test.shape)
print(train_arr)

#%%

#normalise data

train_mean = train_arr.mean(axis=(0, 1))
train_std = train_arr.std(axis=(0, 1))
train_arr = (train_arr - train_mean)/train_std
print(train_arr)
val_arr = (val_arr - train_mean)/train_std


#%%

#train_arr = list(train_arr)

#train_arr = train_arr[0:3]
train_list = list(train_arr)
val_list = list(val_arr)
first_test_list = list(train_arr_first_test)
#def split_sequences(sequences, n_steps):
#    pass

#%%

#get sliding windows for each ID
train_x, train_y = [],[]
val_x, val_y = [],[]
def get_windows(sequence, n_steps, x, y):
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix][0]
        x.append(seq_x)
        y.append(seq_y)

for example in train_list:
    get_windows(example, 6, train_x, train_y)
for example in val_list:
    get_windows(example, 6, val_x, val_y)
print(val_y[0])

train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
val_x = np.asarray(val_x)
val_y = np.asarray(val_y)
print(train_x.shape)
print(val_y.shape)

from keras import backend as K


#%%
@tf.function
def output_clip(x):
    return K.switch(x >= -0.06461986, x, -0.06461986)


train_input_tensor = tf.keras.Input(shape=(6,7))
preprocessed_tensors = []
preprocessed_tensors.append(train_input_tensor[:, :, 0:2])


for i in range(4,7):
    tensor = tf.strings.as_string(train_input_tensor[:, :, i], precision=10)
    vocab = np.unique(train_x[:, :, i]).astype(np.str)
    layer   = preprocessing.StringLookup(vocabulary=vocab)
    encoded_tensor = layer(tensor)
    embedded_tensor = tf.keras.layers.Embedding(input_dim=layer.vocab_size()+1, input_length=(6,1), output_dim=3)(encoded_tensor)
    preprocessed_tensors.append(embedded_tensor)

combine_input = tf.keras.layers.Concatenate()(preprocessed_tensors)

lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(combine_input)
lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(lstm_1)
lstm_3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(lstm_2)
dense_1 = tf.keras.layers.Dense(1, activation=output_clip, dynamic=True)(lstm_3)

total_model = tf.keras.Model(train_input_tensor, dense_1)

total_model.compile(loss="mse", optimizer="adam", metrics="mae")

history = total_model.fit(x=train_x, y=train_y, validation_data=(val_x, val_y), batch_size=400, epochs=1, verbose=1)




total_model.summary()

total_model.save('../Models/Trained models/stacked LSTM model', save_format='tf')

plot_model(total_model, to_file='../Models/Trained models/stacked LSTM model/assets/LSTM_model.png')

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training mse')
plt.plot(epochs, val_loss, 'r', label='Validation mse')
plt.title('Training and validation mse')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

epochs = range(1, len(loss) + 1)
plt.plot(epochs, mae, 'bo', label='Training mae')
plt.plot(epochs, val_mae, 'r', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()
#print(val_arr)
#print(train_arr[:,-7:-1])
#create prediction window excluding last value to compare
#predictions = total_model.predict(train_arr[:,-7:-1])
predictions = total_model.predict(val_arr[:, 1:])

denormalised_predictions = predictions*train_std[0] + train_mean[0]

denormalised_predictions = list(denormalised_predictions.flatten())

all_ids = test['ID'].unique().astype(int)
all_ids.sort

submission = [list(i) for i in zip(all_ids, denormalised_predictions)]

np.savetxt("../Models/Predictions/submissions.csv", submission, header="ID,item_cnt_month", delimiter=",", fmt="%i,%f", comments='')

print(submission[0:200])
