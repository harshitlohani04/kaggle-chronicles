# About this competition:
The goal of this competition is to create models that can be used to aid in the detection and classification of degenerative spine conditions using lumbar spine MR images.
Competitors will be developing models that simulate a radiologist's performance in diagnosing spine conditions.

This is a 4 months competition starting from June 2024 and will be ending in October 2024. I will keep on commiting the changes and progress I make on my kaggle notebook
on a daily basis. This Competition requires a good amount of knowledge not only about ML but also medical knowledge.

I started with this competition almost a month late and have to boost my speed and submit my 1st model as fast as I can.

Not only is this my first kaggle competition but it also my first time working with such a huge amount of data (total data size = 35.34 GB).

# About the dataset:
The data provided for this competition contains 2 folders and 5 csv files adding to a total of 147k files.

### 1. [train.csv/test.csv] --> Labels for the train set.

study_id - The study ID. Each study may include multiple series of images.

[condition]_[level] - The target labels, such as spinal_canal_stenosis_l1_l2, with the severity levels of Normal/Mild, Moderate, or Severe. Some entries have incomplete labels.


### 2. train_label_coordinates.csv

study_id

series_id - The imagery series ID

instance_number - The image's order number within the 3D stack.

condition - There are three core conditions: spinal canal stenosis, neural_foraminal_narrowing, and subarticular_stenosis. The latter two are considered for each side of the spine.

level - The relevant vertebrae, such as l3_l4

[x/y] - The x/y coordinates for the center of the area that defined the label.


### 3. sample_submission.csv

row_id - A slug of the study ID, condition, and level such as 12345_spinal_canal_stenosis_l3_l4.

[normal_mild/moderate/severe] - The three prediction columns.

[train/test]_images/[study_id]/[series_id]/[instance_number].dcm The imagery data.

### 4. [train/test]_series_descriptions.csv

study_id

series_id

series_description The scan's orientation.

### 5. test/train images
These are folders that further contain 3 other folders that contain images. The 3 sub-folders divide the image into 3 classes:
* Sagittal T2/STIR
* Sagittal T1
* Axial T2'
