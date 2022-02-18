# Multimodal-XAI-Medical-Diagnosis-System


## I. Datasets used in this project.

### 1. [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/)
In this dataset, it has a `patient` table including the `gender` and `age` that we want to feed into our model.

### 2. [MIMIC-IV ED](https://physionet.org/content/mimic-iv-ed/1.0/)
MIMIC-IV ED contians a `triage` table containg the patiets' heath condition in emergency daparment, such as temperature, heart rate and blood pressure. Those are the features that we considered as clinical data and should be fed into the model as input.

### 3. [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/)
MIMIC-CXR contains the Chest X-ray image in *DICOM* format, which is specific deisgned for medical purpose. In order to process the image data, we have to transform the format to *.jpg* or *.jpeg*, which can be easily transfered to numpy array or tensor. Fortunately, the same author provide **MIMIC-CXR JPG** dataset that has done the format transformation for us. 

### 4. [MIMIC-CXR JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)
This dataset provide all the Chest X-Ray image in MIMIC-CXR in *.jpg* format. Moreover, it provides 2 types of labels as well. One generated from the same labler used in the [*CheXpert*](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset. And, another labler is [**Negbio**](https://github.com/ncbi-nlp/NegBio), which is NLP tool for negation and uncertainty detection in clinical text. *(Note: These lables are used to generate the labels from radiologists' report.)*

### 5. [Eye Gaze Data for Chest X-rays](https://physionet.org/content/egd-cxr/1.0.0/)
In order to introduce eye tracking data into our model to promote explainability, we need a dataset that provide eye tracking data. This dataset provide eye tracking data, time-stamped utterance from one radiologist. However, we didn't import this dataset for our project but use the technique it used to localise the `stay_id` for Chest X-ray images. (More of the stay_id determination will be discussed in the latter section.)

### 6. [REFLACX](https://physionet.org/content/reflacx-xray-localization/1.0.0/)
This dataset is formed by multiple radiologists to provide eye tracking data, bounding ellipse (abnormality ellipse), time-stamped utterance, and lables. This is the dataset we used for our eye tracking data and bouding ellipses. I


## II. Temrs

Before explaining how we joint these datasets and the relationship between them, some terms have to be clarified. 

1. `subject_id` - It can be considered as patient id (Note: 3. [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) actually call it `patient_id` in the master_sheet, but we will call it `subject_id` in the later sections to prevent confusion.). Every patient has one `subject_id` to represent them in the dataset. All of the 6 datasets we mentioned above in the dataset section have this `subject_id`. 

2. `stay_id` - It represents a specific stay that a patient *stay* in the emergency deparement. This `stay_id` are mainly used to in the [MIMIC-IV ED](https://physionet.org/content/mimic-iv-ed/1.0/) dataste. However, in order to determine the age and health condition (*triage* data table) in the time that the patient took chest x-ray image, we need to identify the `stay_id` for each CXR image as well (It's not provided in [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) or [MIMIC-CXR JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)). 

(Note: They method of identifying `stay_id` is provided by [Eye Gaze Data for Chest X-rays](https://physionet.org/content/egd-cxr/1.0.0/). The chest x-ray image only provide `subject_id` and the time that this radiograph taken. Therefore, we need to find the `subject_id` in [Eye Gaze Data for Chest X-rays](https://physionet.org/content/egd-cxr/1.0.0/) and check which `stay_id` has the duration to include this specific CXR.)

3. `study_id` - This field is presented in  [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) and [MIMIC-CXR JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). From single or multiple CXRs, the radiologists can perform a study and result in a text report. Normally, the labels are generated from this text report manually or automatically. 

4. `dicom_id` - This id represents a specific CXR image.

5. `reflacx_id` - It's the `id` field in REFLACX dataset. In REFLACX, they ask differnt radiologists to provide diagnosis for the same CXR image. Therefore, a single `dicom_id` may has multiple `reflacx_id`. Usually, they are studied by different radiologists. 

## III. Relationships between ids

A patient will have a specific `subject_id`. And, s\he can come to this emergency department of this hospital multiple times, resulting multiple `stay_id` for this patient. In each stay, the doctor may ask the patient to take CXR images for them to provide diagnosis. And, each diagnosis can be considered as a study and own a `study_id`. However, during each stay, the doctor may provide multiple diagnosis (studies) for the patient, which means that one `stay_id` can be related to multiple `study_id`. Also, to provide enough information for the radiologists, the patients may be asked to take multiple CXR images, which means a `study_id` can also ba linked with multiple `dicom_id`.

## IV. Dataset issues.

### 1. Version issues amoung MIMIC-IV dataset.

The MIMIC dataset currently has the version issue resulting the decrease of usable data. Three versions of MIMIC-IV have been released. 

![image](https://user-images.githubusercontent.com/37566901/154606393-6ffcacef-66b3-4789-ada0-813cd4ea160d.png)

However, there's big update about the `transfer_id` (can be seen as `subject_id`), which cause the uncompatibility between version 0.4 and 1.0.

![image](https://user-images.githubusercontent.com/37566901/154606877-ed3bd902-7d3c-4c56-920f-0e86494e3067.png)

Unfortunately, *MIMIC-CXR* and *MIMIC-CXR JPG* hasn't been udpated with the new `transfer_id` (`subject_id`). Therefore, we loss some links between *MIMIC-CXR JPG* and *MIMIC-IV*. The unmatching of the `subject_id` causes a significant decresae of the available dataset size.


### 2. REFLACX ellipse lable issues.

In REFLACX, while the radiologists were diagnosing the CXR images, they are asked to point out certain abnomalities using ellipses. However, they seems changed the available abnoramilties in the middle of the experiment. We found two differernt types of labels among cases. Moreover, some of the lables seems repetitive and should be considered as one. The below chart shows the differences between these two versions and the repetitive lables we found. 

![RepetitiveLabels](https://user-images.githubusercontent.com/37566901/154619705-048f8cbd-5f83-4d96-838f-d541499ecc09.png)


(Note: In the preprocessing stage, we simply map the repetitive labels together.)


## V. Task and Model to use.

Before we and retrieve the data and preprocess them, we should determine what's the task we want to perfrom and what's the model architecture (prototype) we can use to perfrom this task. 

In this project, we want to desgin a algorithm (neural network) that can take CXR image and clinical data to detect the abnormalities (diseases). Therefore, the model will apply  multi-modal learninig, which allow the model to process the data from differernt modalities. And the blow diagrams show the model prototype we came out. 

![MultiModalArchitecture](https://user-images.githubusercontent.com/37566901/154620384-b3cdffed-9b4b-484d-85b8-8ea3b68c09cc.png)

At the right hand side, we have a `FullyConnectedLayers` to process clincal data. And, this part is actually like the architecture below.:

![ClinicalNet](https://user-images.githubusercontent.com/37566901/154620575-b751fe8a-de79-42e7-95e2-fd44afd5dbb2.png)

(The only cetegorical data we have is `gender`. And, it will be passed into the embedding layer first before passing through the fully connected layers.)


## VI. Fields used.

### We include these id fields

```python
['id', 'dicom_id', 'subject_id', 'stay_id', 'study_id', 'split']
```
(Note: `id` is the `id` field in REFLACX dataset and can be seen as `reflacx_id`)

### The metadata of CXR images:

1. `image_path`: path to JPG image.
2. `ViewPosition`: Lateral or PA or AP view.
3. `image_size_x`: width of the image.
4. `image_size_y`: high of the image.
5. `anomaly_location_ellipses_path`: path to the bouding ellipse dataframe (.csv file).

### From MIMIC-IV ED triage table, we inlcude these fields:

```python
['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain', 'acuity']
```

(Note: more details of these fields, you can see the [official documentation](https://mimic.mit.edu/docs/iv/modules/ed/triage/))



### All the label columns from REFLACX:

```python
[
'Airway wall thickening', 'Atelectasis', 'Consolidation',
'Enlarged cardiac silhouette', 'Fibrosis',
'Groundglass opacity', 'Other', 'Pneumothorax', 'Pulmonary edema',
'Quality issue', 'Support devices', 'Wide mediastinum',
'Abnormal mediastinal contour', 'Acute fracture', 'Enlarged hilum',
'Hiatal hernia', 'High lung volume / emphysema',
'Interstitial lung disease', 'Lung nodule or mass',
'Pleural abnormality'
]
```

(Note: In the later sections, we usually use the 5 most common diseases in the dataset. They are *Enlarged cardiac silhouette, Atelectasis, Pleural abnormality, Consolidation, Pulmonary edema.)














