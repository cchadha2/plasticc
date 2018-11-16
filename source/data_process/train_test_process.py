import pandas as pd 
from feature_engineering_lgb_1_1 import data_process
import time

data = 'data/'
output = 'output/'
chunk_size = 20000000
train_save = False
test_save = True

train_name = 'processed_train_1.1.csv'
test_name = 'processed_test_1.1.csv'

if train_save:
    start = time.time()
    train = pd.read_csv(data + 'training_set.csv', index_col=False)
    print('Processing train dataset. train shape: {}'.format(train.shape))
    train_df = data_process(train)
    print('Processed train. Train shape: {};'.format(train_df.shape))
    train_df.to_csv(output + train_name, index=False)
    end = time.time()
    print('Time taken to process train set: {:.2f}s'.format(end-start))

if test_save:
    start = time.time()
    chunk_iter = pd.read_csv(data + 'test_set.csv', iterator=True, chunksize=chunk_size, index_col=False)
    test_meta = pd.read_csv(data + 'test_set_metadata.csv', index_col=False)
    # processed_test_1.1:
    test_meta = test_meta.drop('hostgal_specz', axis=1)

    test = pd.DataFrame()
    for chunk in chunk_iter:
        chunk_start = time.time()
        print('Processing test chunk {}'.format(chunk.shape))
        processed_chunk = data_process(chunk, test=True)
        print('Processed test chunk shape: {}'.format(processed_chunk.shape))
        test = test.append(processed_chunk)
        print('Test rows completed processing: {}/{}'.format(test.shape[0],test_meta.shape[0]))
        chunk_end = time.time()
        print('Time taken for chunk: {:.2f} mins'.format((chunk_end - chunk_start)/60))
        print('Time remaining: {:.0f} mins'.format(((test_meta.shape[0]*(chunk_end - start))/test.shape[0] - (chunk_end - start))/60))
        print('________________')
    print('Chunk processing completed: {:.0f} mins'.format((chunk_end-start)/60))
    print('Number of duplicate object_ids after aggregation: {}'.format((test.groupby('object_id').size() > 1).sum()))
    test = test.groupby('object_id').mean()
    end = time.time()
    print('Overall time taken to process test set: {:.0f} mins'.format((end-start)/60))

    test = test_meta.merge(test, on='object_id')
    test.to_csv(output + test_name, index=False)
    print('Complete test set saved to disk. Shape: {}'.format(test.shape))