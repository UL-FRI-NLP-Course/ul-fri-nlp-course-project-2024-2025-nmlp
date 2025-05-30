# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

The aim of this project is to leverage various natural language processing techniques to enhance a large language model's ability to generate short traffic reports.
The scope also encompasses a dataset that can be used for any supervised approach.
This dataset is comprised of website data that was obtained from the national traffic news website and the final news reports. 

- **`src/`**  
  - **`data/consolidate_data.py`** – 
  - **`preprocessing/gams.py`** –
  - **`models/input.py`** –   
  - **`models/output.py`** – 
  - **`utils/utils.py`** –


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
  - **`subset_preparation.py`** – 

# Report
The report for first submission is available [here](https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-nmlp/blob/main/report/report1.pdf).
The report for second submission is available [here](https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-nmlp/blob/main/report/report2.pdf).
