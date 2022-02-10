def relabel_ellipse_df(ellipse_df, label_maps={
    'Airway wall thickening': ['Airway wall thickening'],
    'Atelectasis': ['Atelectasis'],
    'Consolidation': ['Consolidation'],
    'Enlarged cardiac silhouette': ['Enlarged cardiac silhouette'],
    'Fibrosis': ['Fibrosis'],
    'Groundglass opacity': ['Groundglass opacity'],
    'Pneumothorax': ['Pneumothorax'],
    'Pulmonary edema': ['Pulmonary edema'],
    'Quality issue': ['Quality issue'],
    'Support devices': ['Support devices'],
    'Wide mediastinum': ['Wide mediastinum'],
    'Abnormal mediastinal contour': ['Abnormal mediastinal contour'],
    'Acute fracture': ['Acute fracture'],
    'Enlarged hilum': ['Enlarged hilum'],
    'Hiatal hernia': ['Hiatal hernia'],
    'High lung volume / emphysema': ['High lung volume / emphysema',  'Emphysema'],
    'Interstitial lung disease': ['Interstitial lung disease'],
    'Lung nodule or mass': ['Lung nodule or mass', 'Mass', 'Nodule'],
    'Pleural abnormality': ['Pleural abnormality', 'Pleural thickening', 'Pleural effusion'],
},
    fixed_columns = ['xmin', 'ymin', 'xmax', 'ymax', 'certainty'],
):
    relabeled_ellipses_df = ellipse_df[fixed_columns]

    # relabel it.
    for k in label_maps.keys():
        relabeled_ellipses_df[k] = ellipse_df[[l for l in label_maps[k] if l in ellipse_df.columns]].any(axis=1)
    
    return relabeled_ellipses_df
