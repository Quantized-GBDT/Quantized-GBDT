<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=300 />

Quantized Training of Gradient Boosting Decision Trees
===============================

This is the repository for experiments of paper [Quantized Training of Gradient Boosting Decision Trees](https://openreview.net/forum?id=Cd-b50MZ0Gc&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions)). The implementation is based on [LightGBM](https://github.com/microsoft/LightGBM).

Instructions to Run Experiments
-------------------------------

To run the experiments, please:
1. Clone this repository, including all the submodules.
2. Download the dataset from <https://pretrain.blob.core.windows.net/quantized-gbdt/dataset.zip> (the .zip file is 38.29 GB).
2. Build the CLI executable files of [LightGBM](https://github.com/microsoft/LightGBM), [LightGBM-master](https://github.com/microsoft/LightGBM), [catboost](https://github.com/catboost/catboost), and [xgboost](https://github.com/dmlc/xgboost) according to the CLI building instructions of these tools.
3. Generate the bash scripts using [experiments/generate_script.py](https://github.com/Quantized-GBDT/Quantized-GBDT/blob/master/experiments/generate_script.py).
4. Run the generated bash script.
5. Parse the results into markdown table using [experiments/parse_logs.py](https://github.com/Quantized-GBDT/Quantized-GBDT/blob/master/experiments/parse_logs.py).

Training logs and sample outputs of experiments in the paper are provided in [experiments/sample_outputs](https://github.com/Quantized-GBDT/Quantized-GBDT/tree/master/experiments/sample_outputs).
