from google.cloud import storage
import pandas as pd


name = "heart_load/test_1.csv"
# Instantiates a client

print(f"Download blob from bucket {name}")
storage_client = storage.Client()
bucket = storage_client.bucket("ml2_bucket")
blob = bucket.blob(name)
df = pd.read_csv(f'gs://ml2_bucket/{name}')
#df = dd.read_csv(f'gs://ml2_bucket/{name}')
print(df)

df.to_csv(f'gs://ml2_bucket/heart_load/test_pred.csv', index=False)

