import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

val = pd.read_csv('reporting/data/lgb_valid.csv')

val = val.drop('notes', axis = 1)
val = val[val['training_dataset'].isna() == False]
val = val[val['score'].isna() == False]

le = LabelEncoder()

for column in ['objective', 'boosting_type', 'stratified', 'training_dataset']:
    val[column] = le.fit_transform(val[column])

# Plot outputs
plt.scatter(val['training_dataset'], val['score'],  color='black')
plt.xlabel('Index')
plt.ylabel('Score')
plt.show()