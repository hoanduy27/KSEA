import pandas as pd 

INIT_SIZE = 10000
TEST_SIZE = 10000
POOL_SIZE = 100000

df = pd.read_csv('data/training.csv')
df = df.sort_values(by="id", ascending=True)

df_init = df.head(INIT_SIZE)
df_rest = df.iloc[INIT_SIZE:, :]

df_test = df_rest.sample(TEST_SIZE)
df_pool = df_rest[~df_rest.index.isin(df_test.index)].sample(POOL_SIZE)

df_init.to_csv('data/training_init.csv', index=False)
df_test.to_csv('data/test.csv', index=False)
df_pool.to_csv('data/training_pool.csv', index=False)

out_df = pd.concat([df_init, df_test, df_pool])

out_df[['target', 'id', 'user']].to_csv('data/label.csv', index=False)

print(df_init.groupby("target").apply(len))
print(df_test.groupby("target").apply(len))
print(df_pool.groupby("target").apply(len))