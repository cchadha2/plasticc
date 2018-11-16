FROM mongo

COPY /data/all-collections.archive /data/all-collections.archive

CMD mongorestore --host mongodb --nsFrom reporting.validation --nsTo reporting.validation --archive=/data/all-collections.archive