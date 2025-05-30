# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

The aim of this project is to leverage various natural language processing techniques to enhance a large language model's ability to generate short traffic reports.
The scope also encompasses a dataset that can be used for any supervised approach.
This dataset is comprised of website data that was obtained from the national traffic news website and the final news reports. 

- **`src/`**  
  - **`consolidate_data.py`** – Data extraction/preparation for DP2.
  - **`gams.py`** – Baseline inference program.
  - **`input.py`** – DP2 reading phase.
  - **`output.py`** – DP2 output phase.
  - **`utils.py`** – Helper functions used for DP2 algorithm. 

- **`fine_tunning/`**
  - **`dp1_inf.py`** – Inference script for the `dp1` model variant
  - **`dp1.sh`** – Shell script to launch `dp1_inf.py`
  - **`dp2_inf.py`** – Inference script for the `dp2` model variant
  - **`dp2.sh`** – Shell script to run `dp2_inf.py`
  - **`fine_tunning.py`** – Main fine-tuning pipeline for gams
  - **`ft.sh`** – Shell script to launch `fine_tunning.py`
 
- **`evaluation/`**
  - **`eval.sh`** – Shell script for launching evaluation jobs
  - **`evaluation.py`** – General evaluation script for computing metrics (e.g., accuracy, BLEU, ROUGE)
  - **`llm_evaluation.py`** – Evaluation routine using LLM (deepseek)
  - **`llm_evaluation_2.py`** – Alternative or extended version of `llm_evaluation.py`
  - **`subset_preparation.py`** – Extract a small subset of input data for DP2

- **`dp1/`**
  - **`extract.py`** – DP1 extraction of data from rtf and processing it for further use (prompting, fine-tunning)
  - **`sentenceMatching.py`** – Sentence matching algorithms used as a helper functio for extract.py

# Report
The report for first submission is available [here](https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-nmlp/blob/main/report/report1.pdf).
The report for second submission is available [here](https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-nmlp/blob/main/report/report2.pdf).
The report for second submission is available [TODO](TODO).

# How to run
In the following section, may reference some files/models that are available on the Arnes HPC cluser in a shared directory under `/d/hpc/projects/onj_fri/nmlp`
## Installation and configuration
Begin by cloning the repository and creating the virtual environment
```bash
git clone https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-nmlp.git
cd ul-fri-nlp-course-project-2024-2025-nmlp
python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-pytorch.txt # You may need to adjust the index-url for your CUDA version
python -m spacy download sl_core_news_lg # Spacy trained pipeline for Slovenian
```
We recommend running any python scripts as modules, e.g.: `python -m path.to.file` instead of `python path/to/file.py`.

## Model download
The scripts will automatically download the language models to `~/.cache/huggingface`.
If you're running this on the HPC cluster, you can just create a symlink to the shared directory:
```bash
# Optionally make a backup of existing directory
# mv ~/.cache/huggingface ~/.cache/huggingface.bak
ln -s /d/hpc/projects/onj_fri/nmlp/huggingface ~/.cache/huggingface
```

## Data preparation
### DP1
TODO
### DP2
The [raw input data](https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-nmlp/blob/main/data/Podatki%20-%20PrometnoPorocilo_2022_2023_2024.xlsx) is part of the repository.
The raw output data however, was too large for comfort, so it is available [here](https://unilj-my.sharepoint.com/:u:/r/personal/slavkozitnik_fri1_uni-lj_si/Documents/Predmeti/ONJ/ONJ_2025_Spring/Projects/RTVSlo.zip?csf=1&web=1&e=zhNDxj) and on the HPC cluster as `RTVSlo.zip`.
After unzipping it and placing `RTVSlo` directory into the project root directory, you can run `python -m src.consolidate_data` to start generating the processed data for DP2, which will be saved to `dp2.jsonl`.
This file is also available on the cluster.

## Fine tuning (FT)
For fine-tuning, we recommend an HPC node with at least on H100 or equivalent.
Before running the whole process, you can adjust some variables in `fine_tunning/fine_tunning.py`:
- `IO_PAIRS_PATH`: path to `dp1.jsonl` or `dp2.jsonl` file (output from data preprocessing stage).
- `MODEL_NAME` to choose the size of the model you want to FT (we used 27B variant).
- `PEFT_DIR` path to where the checkpoints will be saved to.
Now run the process using `python -m fine_tunning.fine_tunning`.

## Inference
Disclaimer: When running inference for multiple prompts, you may need to rerun the script in case you run out of VRAM.
### Basic inference
Basic inference is demonstrated in `src/gams.py`. It takes a subset of the inputs (file `data/dp2_inputs.jsonl`, generated using `evaluation/subset_preparation.py`) and generates reports for every input.
The result of this is saved to `data/basic_outputs.jsonl`.
### Inference using fine-tuned model with DP1
Basic inference is demonstrated in `fine_tunning/dp1_inf.py`. It takes a subset of the inputs (file `data/dp1_inputs.jsonl`) and generates reports for every input.
The result of this is saved to `data/dp1_outputs.jsonl`.
### Inference using fine-tuned model with DP2
Basic inference is demonstrated in `fine_tunning/dp2_inf.py`. It takes a subset of the inputs (file `data/dp2_inputs.jsonl`, generated using `evaluation/subset_preparation.py`) and generates reports for every input.
The result of this is saved to `data/dp2_outputs.jsonl`.
