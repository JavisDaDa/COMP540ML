import pandas as pd
from config import test_data_df, name


def vector2label(lists):
    for index, pred_list in enumerate(lists):
        new_pred_list = []
        for i in range(len(pred_list)):
            for j in range(len(pred_list[i][0])):
                new_pred_list.append(pred_list[i][0][j])
        test_data_df[f'pred_label_{index + 1}'] = new_pred_list
    top1_df = pd.DataFrame(data=test_data_df[['img_name', 'pred_label_1']])
    top1_df.columns = ['img_name', 'label']
    test_data_df['Top3'] = test_data_df['pred_label_1'].map(str) + ' ' + test_data_df['pred_label_2'].map(
        str) + ' ' + test_data_df['pred_label_3'].map(str)
    top3_df = pd.DataFrame(data=test_data_df[['img_name', 'Top3']])
    top3_df.columns = ['img_name', 'label']
    top1_df.to_csv(f'./drive/My Drive/COMP540/{name}_1.csv', index=False)
    top3_df.to_csv(f'./drive/My Drive/COMP540/{name}_3.csv', index=False)