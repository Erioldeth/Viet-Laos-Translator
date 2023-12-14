# Viet-Laos-Translator

The Viet-Laos-Translator project is an open-source project forked from [KC4.0_MultilingualNMT](https://github.com/KCDichDaNgu/KC4.0_MultilingualNMT).

## Setup

### Install Viet-Laos-Translator tool

**Note:**
The current version is only compatible with python 3.11

```bash
git clone https://github.com/Erioldeth/Viet-Laos-Translator.git
cd Viet-Laos-Translator
pip install -r requirements.txt

# Quickstart
```
## Step 1: Prepare the Data

The current project comes with pre-existing data located in data/lo_vi. If you wish to make changes, please ensure that you maintain the correct names and file formats of the data files

The data includes:
* `train.lo`
* `train.vi`
* `dev.lo`
* `dev.vi`
* `test.lo`
* `test.vi`

## Step 2: Train the Model

To train a new model, edit the YAML config file:
Modify the config yml config file to set hyperparameters and the path to the training data:

```yaml
# data setup
data:
  train_data_location: data/lo_vi/train
  valid_data_location: data/lo_vi/dev
  src_lang: .lo
  trg_lang: .vi
build_vocab_kwargs:
  min_freq: 2
# model params
d_model: 512
n_layers: 6
heads: 8
dropout: 0.1
# training config
train_max_length: 150
train_batch_size: 16
epochs: 40
# optimizer & learning args
optimizer_params:
  lr: !!float 1e-4
  betas:
    - 0.9
    - 0.98
  eps: !!float 1e-9
n_warmup_steps: 3000
label_smoothing: 0.1
# inference config
infer_max_length: 150
input_max_length: 175
infer_batch_size: 1
decode_strategy_kwargs:
  beam_size: 3
  length_normalize: 0.6
```

Then, run the following command:

```bash
python main.py train 
```
**Note**:
- After training, the model will be saved in trained_model/, this folder will includes the trained model and vocabulary files.

## Step 3: Translate

The model uses the beam search algorithm and saves the translation at `$your_data_path/translated.vi`.

```bash
python main.py infer --features_file $your_data_path/test.lo --predictions_file $your_data_path/translated.vi
```

## Step 4: Evaluate Quality using BLEU Score

Evaluate BLEU score using multi-bleu:

```bash
perl thrid-party/multi-bleu.perl $your_data_path/translated.vi < $your_data_path/test.vi
```

## Details and References 
If you have any feedback or contributions, please send an email to crystaleye005@gmail.com or dannguyenhai10112003@gmail.com

## Please cite the following paper:
```bash
@inproceedings{ViNMT2022,
  title = {ViNMT: Neural Machine Translation Toolkit},
  author = {Nguyen Hoang Quan, Nguyen Thanh Dat, Nguyen Hoang Minh Cong, Nguyen Van Vinh, Ngo Thi Vinh, Nguyen Phuong Thai, Tran Hong Viet},
  booktitle = {https://arxiv.org/abs/2112.15272},
  year = {2022},
}
```
