# Run mongo docker
docker run --name mongo -p 27017:27017 -d mongo:latest

# Dump mongo archive
docker exec reporting_mongodb_1 sh -c 'exec mongodump -d reporting --archive' > /Users/cchadha2/Documents/github-private/plasticc/reporting/data/lgb_valid.archive

# Dump archive as csv
docker exec reporting_mongodb_1 sh -c 'mongoexport --db reporting --collection validation --type csv --fields objective,num_class,boosting_type,learning_rate,num_leaves,colsample_bytree,subsample,subsample_freq,max_depth,reg_alpha,reg_lambda,min_split_gain,min_child_weight,seed,training_dataset,early_stopping_rounds,num_folds,stratified,score,notes' > /Users/cchadha2/Documents/github-private/plasticc/reporting/data/lgb_valid.csv

# Run with archive volume
docker run --name mongo -p 27017:27017 -v /Users/cchadha2/Documents/github-private/plasticc/reporting/data/lgb_valid.archive:/data/lgb_valid.archive -d mongo:latest

# Restore mongo container with dump (inside bash shell in container)
mongorestore --nsFrom reporting.validation --nsTo reporting.validation --archive=/data/lgb_valid.archive