# Relation Extraction First - Using Relation Extraction to Identify Entities

Code for [AIFB-WebScience at SemEval-2022 Task 12: Relation Extraction First - Using Relation Extraction to Identify Entities](https://arxiv.org/abs/2203.05325)

# Table of Contents
1. [Data](#data)
2. [Dependencies](#dependencies)
3. [Reproducing Results (+ link to trained models)](#reproducing-results)
4. [Training](#training)
4. [Relevant Code per Section in Paper](#relevant-code-per-section-in-paper)
4. [Bibtex for Citing](#cite)

# Data
The task data can be found [here](https://competitions.codalab.org/competitions/34011#participate-get_starting_kit).
To use it with the code in this repo, place the training, dev, test (json-)files into the corresponding folders in the data directory.

# Dependencies

Dependencies:
- torch (1.8.0)
- transformers (4.18.0)
- tqdm (4.64.0)
- wandb (0.12.16)
- pylatexenc (2.10)

```
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
python -m pip install -U setuptools
python -m pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install transformers tqdm wandb pylatexenc
```

# Reproducing Results
Checkpoints for 4 trained models can be found [here](https://drive.google.com/drive/folders/1AJv4Q20SH5toUL7H3PCTuzQapRJ7ZpF-?usp=sharing).
Commands for using the models with k=400:
```
python eval.py --checkpoint checkpoints/max_no_preprocessing.pt --pooling max --k_mentions 400
python eval.py --checkpoint checkpoints/mean_no_preprocessing.pt --pooling mean --k_mentions 400
python eval.py --checkpoint checkpoints/max_latex2text.pt --pooling max --preprocessing latex2text --k_mentions 400
python eval.py --checkpoint checkpoints/mean_latex2text.pt --pooling mean --preprocessing latex2text --k_mentions 400
```

# Training
Example of training a model with max pooling and no preprocessing:
```
python train.py --learning_rate 7e-5 --seed_model 3 --num_epochs 60 --k_mentions 50 --pooling max --candidate_downsampling 1000
```

Example of training a model with max pooling and latex2text preprocessing:
```
python train.py --learning_rate 5e-5 --seed_model 1 --num_epochs 60 --k_mentions 50 --pooling max --candidate_downsampling 1000 --preprocessing latex2text
```

# Relevant Code per Section in Paper

## Section 3.1 Input Encoding

Covered in models/data.py (lines 126-147)

## Section 3.2 Soft Mention Detection

Covered in models/base_model.py (lines 68-127)

## Section 3.3 Relation Extraction

Covered in models/base_model.py (lines 129-151)

## Section 3.4 Entity Type Classification

Covered in models/base_model.py (lines 232-244)

# Cite
If you use the code in this repo, please cite this paper:

```
@inproceedings{popovic_semeval_2022, 
 title = "AIFB-WebScience at SemEval-2022 Task 12: Relation Extraction First - Using Relation Extraction to Identify Entities", 
 author = "Popovic, Nicholas and Laurito, Walter and Färber, Michael",
 booktitle = "Proceedings of the 16th International Workshop on Semantic Evaluation ({S}em{E}val-2022)", 
 year = "2022", 
 publisher = "Association for Computational Linguistics"
}
```