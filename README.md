# NLP-FinalProject
NLP Final Project
Model suggestions for q4: Bert, Roberta (Twitter sentiment fine-tuned one)

# How to run
### The notebook to refer to is lora-finetune.ipynb/lora-finetune.ipynb 
### To get the trained weights refer to "upload_nlp" folder

## Method 1 (Recommended): Using Kaggle 
Steps:
1. Open the Kaggle notebook using the link: https://www.kaggle.com/code/reynardsimano/notebookdef5c6c86f
2. Download necessary inputs
   a) Either pip install peft OR Download dataset directly in Kaggle input, "Add Input" --> Click on "dataset" filter --> search for "peft-main" and add input (https://www.kaggle.com/datasets/nbroad/peft-main)
3. Run the Kaggle file with the settings:
   a) Run with internet on in "Settings" (online mode is needed to do any commands such as pip install, otherwise, a separate search for d2l library needs to be added like step 2)
   b) Use accelerator: 2 x T4 GPU (Note: limit of 30 hours usage per week, total training time for the model takes longer than that)
4. (Optional) Load the trained model weights for immediate testing

## Method 2: Not using Kaggle
Steps:
1. Open the lora-finetune.ipynb file
2. Download the necessary libraries (d2l, transformers, torch with cuda support, tqdm, sklearn, pandas, numpy)
3. Change the path references
4. Run the ipynb file
5. (Optional) Load the trained model weights for immediate testing

## Notes:
1. This pipeline can be used for other forms of data as long as the dataset format is similar
