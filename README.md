# NLP-FinalProject
NLP Final Project


First 3 parts complete
Model suggestions for q4: Bert, Roberta (Twitter sentiment fine-tuned one)


Steps:
1. Open the Kaggle notebook using the link: https://www.kaggle.com/code/reynardsimano/notebookdef5c6c86f
2. Download necessary inputs
   a) Either pip install peft OR Download dataset directly in Kaggle input, "Add Input" --> Click dataset filter --> search for "peft-main" and add input (https://www.kaggle.com/datasets/nbroad/peft-main)
3. Run the Kaggle file with the settings:
   a) Run with internet on in "Settings" (online mode is needed to do any commands such as pip install, otherwise, a separate search for d2l library needs to be added like step 2)
   b) Use accelerator: 2 x T4 GPU (Note: limit of 30 hours usage per week, total training time for the model takes longer than that)

Notes:
1. This pipeline can be used for other forms of data as long as the dataset format is similar
