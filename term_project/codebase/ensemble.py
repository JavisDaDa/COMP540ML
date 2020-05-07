import pandas as pd
import os
from collections import defaultdict
from itertools import combinations

path = 'D:\\Rice\\COMP 540\\ml\\Statistical-ML\\term_project\\res\\test'
names = [name for name in os.listdir(path) if name[-4:] == '.csv']

n = len(names)
# for index, names in enumerate(combinations(names, n)):
for index, names in enumerate(names):
    print(f'Predicting {index + 1} combination')
    total_df = None
    for name in names:
        df_path = os.path.join(path, name)
        df = pd.read_csv(df_path)
        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([total_df, df.loc[:, 'label']], axis=1, join='outer')
    n = len(total_df)
    final = []
    for i in range(n):
        cur = list(total_df.iloc[i].values[1:])
        labels = defaultdict(int)
        for ind in cur:
            for label in ind.split(' '):
                labels[label] += 1
        res = []
        for a, _ in sorted(labels.items(), key=lambda d: d[1], reverse=True)[:3]:
            res.append(a)
        new = ' '.join(res)
        final.append(new)
    total_df['res'] = final
    res_df = total_df.loc[:,['img_name','res']]
    res_df.columns = ['img_name','label']
    res_df.to_csv(f'test_{index + 1}.csv', index=False)

