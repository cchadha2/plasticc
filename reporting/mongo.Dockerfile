FROM mongo

COPY /data/all-collections.archive /data/lgb_valid.archive

CMD mongorestore --host mongodb --nsFrom reporting.validation --nsTo reporting.validation --archive=/data/lgb_valid.archive