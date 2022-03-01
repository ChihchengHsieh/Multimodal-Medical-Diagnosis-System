from matplotlib.patches import Ellipse


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
    fixed_columns=['xmin', 'ymin', 'xmax', 'ymax', 'certainty'],
):
    relabeled_ellipses_df = ellipse_df[fixed_columns]

    # relabel it.
    for k in label_maps.keys():
        relabeled_ellipses_df[k] = ellipse_df[[
            l for l in label_maps[k] if l in ellipse_df.columns]].any(axis=1)

    return relabeled_ellipses_df


def get_ellipses_patch(relabeled_ellipse_df, d, image_size_x, image_size_y, model_input_image_size, color_code_map=None):
    ellipses = []

    for _, instance in relabeled_ellipse_df[relabeled_ellipse_df[d]].iterrows():
        center_x = (instance['xmin'] + instance['xmax']) / 2
        center_y = (instance['ymin'] + instance['ymax']) / 2
        width = abs(instance['xmax'] - instance['xmin'])
        height = abs(instance['ymax'] - instance['ymin'])
        x_ratio = model_input_image_size / image_size_x
        y_ratio = model_input_image_size / image_size_y

        ellipses.append(Ellipse((center_x * x_ratio, center_y * y_ratio), width=width*x_ratio, height=height*y_ratio,
                        edgecolor=color_code_map[d] if not (color_code_map is None) else "red", facecolor="none", linewidth=2))

    return ellipses


def get_color_coded_ellipses_for_dicom(
        dataset,
        relabeled_ellipse_df,
        image_size_x,
        image_size_y,
        model_input_image_size,
        color_code_map,
        ):

    all_ellipses = []

    for d in dataset.labels_cols:
        all_ellipses.extend(get_ellipses_patch(
            relabeled_ellipse_df, d, image_size_x, image_size_y, model_input_image_size, color_code_map))

    return all_ellipses
