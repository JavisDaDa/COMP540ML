import pandas as pd


def createclass2label(path):
    table = pd.read_table(path, header=None, sep=' ')
    table.columns = ['label', 'class']
    class_dic = {}
    n = len(table)
    for i in range(n):
        class_dic[table.iloc[i]['class']] = table.iloc[i]['label']
    return class_dic