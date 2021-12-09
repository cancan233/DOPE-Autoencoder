# supreme-pancakes
Team research project for CSCI2952G Deep Learning in Genomics in Brown University.

The Research Project Assessment can be found [here](https://docs.google.com/document/d/1e6TNuJMCkX_YlIjgTuEHGVhf9mUIB_UNRDja9nVnY1c/edit)

## TODO

| Completion 	|             Task            	| Due Date 	|
|:----------:	|:---------------------------:	|:--------:	|
|      Y     	|      Project team plan      	|   10/01  	|
|      Y     	|      [Literature Review](./literature_review.pdf)      	|   10/15  	|
|      N     	|         [First draft](./write/first_draft.pdf)         	|   10/28  	|
|      N     	|  Project pitch presentation 	|   11/02  	|
|      N     	|       Project check-in      	|   11/23  	|
|      N     	|         Second draft        	|   12/07  	|
|      N     	| Final Project presentations 	|   12/14  	|
|      N     	|     Final research paper    	|   12/16  	|

## Data
We are using multi-omics data for training our model. 

preprocess.py includes functions that convert the raw text files holding biomedical and clinical data into csvs through pandas dataframes. The first column contains the patients' uuids ('bcr_patient_uuid') and the subsequent columns include features, such as 'pharmaceutical_therapy_type', 'retrospective_collection', 'gender', etc. Some patient uuids of the biomedical data do not correspond to select information collected for the clinical data so those entries currently are NaN, which we will determine how to further process for training.
