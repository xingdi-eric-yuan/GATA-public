# WIP, Please use the master branch
---------------------------------------------------------------------------
Code for paper [Learning Dynamic Knowledge Graphs to Generalize on Text-based Games](https://arxiv.org/abs/2002.09127).

```
# Dependencies
conda create -p /tmp/gata python=3.6 numpy scipy ipython matplotlib cython nltk pillow
source activate /tmp/gata
pip install --upgrade pip
pip install numpy==1.16.2
pip install gym==0.15.4
pip install textworld
pip install -U spacy
python -m spacy download en
pip install tqdm pipreqs h5py pyyaml visdom
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch

# Download FastText Word Embeddings
wget "https://bit.ly/2U3Mde2"
```

## GATA
### Pre-training Graph Updater by Observation Generation
```
# Download data for observation generation / contrastive observation classification
cd obs_gen.0.1 ; wget https://bit.ly/3ep1yhI ; unzip obs_gen.0.1.zip ; cd ..
# Train
python train_obs_generation.py configs/pretrain_observation_generation.yaml

```

### Pre-training Graph Updater by Contrastive Observation Classification
```
# Download data for observation generation / contrastive observation classification
cd obs_gen.0.1 ; wget https://bit.ly/3ep1yhI ; unzip obs_gen.0.1.zip ; cd ..
# Train
python train_obs_infomax.py configs/pretrain_contrastive_observation_classification.yaml

```

### Train Action Scorer with RL
```
# Download games
cd rl.0.2 ; wget https://bit.ly/2Mb4CBR ; unzip rl.0.2.zip ; cd ..
# Modify configs/train_gata_rl.yaml
#   L30: True to load pre-trained graph encoder, False to randomly initialize.
#     L31:  'gata_pretrain_obs_gen_model', 'gata_pretrain_obs_infomax_model'. When L30 is True.
#   L33:  'gata_pretrain_obs_gen_model' or 'gata_pretrain_obs_infomax_model'
#   L84:  3/7/5/9 correspond to the 1/2/3/4 in paper
#   L85:  1/20/100
#   L125: False/True
# To train
python train_rl_with_continuous_belief.py configs/train_gata_rl.yaml

```

## GATA-GTF
### Pre-training Graph Encoder by Action Prediction
```
# Download data
cd ap.0.2 ; wget https://bit.ly/2v6nbC1 ; unzip ap.0.2.zip ; cd ..
# Train
python train_action_prediction.py configs/pretrain_action_prediction_full.yaml

```

### Pre-training Graph Encoder by State Prediction
```
# Download data
cd sp.0.2 ; wget https://bit.ly/2Uyj9wS ; unzip sp.0.2.zip ; cd ..
# Train
python train_state_prediction.py configs/pretrain_state_prediction_full.yaml

```

### Pre-training Graph Encoder by Deep Graph Infomax
```
# Download data
cd dgi.0.2 ; wget https://bit.ly/383vAEQ ; unzip dgi.0.2.zip ; cd ..
# Train
python train_deep_graph_infomax.py configs/pretrain_deep_graph_infomax_full.yaml

```

### Train Action Scorer with RL
```
# Download games
cd rl.0.2 ; wget https://bit.ly/2Mb4CBR ; unzip rl.0.2.zip ; cd ..
# Modify configs/train_gata_gtf_rl.yaml
#   L30: True to load pre-trained graph encoder, False to randomly initialize.
#     L31:  'gata_gtf_pretrain_ap_full_model', 'gata_gtf_pretrain_sp_full_model', or 'gata_gtf_pretrain_dgi_full_model'. When L30 is True.
#   L84:  3/7/5/9 correspond to the 1/2/3/4 in paper
#   L85:  1/20/100
#   L125: False/True
# To train
python train_rl_with_ground_truth_discrete_belief.py configs/train_gata_gtf_rl.yaml

```

## GATA-GTP
### Pre-training Graph Encoder by Action Prediction
```
# Download data
cd ap.0.2 ; wget https://bit.ly/2v6nbC1 ; unzip ap.0.2.zip ; cd ..
# Train
python train_action_prediction.py configs/pretrain_action_prediction_seen.yaml

```

### Pre-training Graph Encoder by State Prediction
```
# Download data
cd sp.0.2 ; wget https://bit.ly/2Uyj9wS ; unzip sp.0.2.zip ; cd ..
# Train
python train_state_prediction.py configs/pretrain_state_prediction_seen.yaml

```

### Pre-training Graph Encoder by Deep Graph Infomax
```
# Download data
cd dgi.0.2 ; wget https://bit.ly/383vAEQ ; unzip dgi.0.2.zip ; cd ..
# Train
python train_deep_graph_infomax.py configs/pretrain_deep_graph_infomax_seen.yaml

```

### Pre-training Graph Updater by Command Generation
```
# Download data for command generation
cd cmd_gen.0.2 ; wget https://bit.ly/385UVOC ; unzip cmd_gen.0.2.zip ; cd ..
# Train
python train_command_generation.py configs/pretrain_command_generation.yaml

```

### Train Action Scorer with RL
```
# Download games
cd rl.0.2 ; wget https://bit.ly/2Mb4CBR ; unzip rl.0.2.zip ; cd ..
# Modify configs/train_gata_gtp_rl.yaml
#   L30: True to load pre-trained graph encoder, False to randomly initialize.
#     L31:  'gata_gtp_pretrain_ap_seen_model', 'gata_gtp_pretrain_sp_seen_model', or 'gata_gtp_pretrain_dgi_seen_model'. When L30 is True.
#   L84:  3/7/5/9 correspond to the 1/2/3/4 in paper
#   L85:  1/20/100
#   L125: False/True
# To train
python train_rl_with_discrete_belief.py configs/train_gata_gtp_rl.yaml

```


### Monitoring training progress
To monitor training progress: set "`visdom: True`" in `config_***.yaml` under the `general` section, and start [Visdom](https://github.com/facebookresearch/visdom) in another terminal using the `visdom` command line. Then, open the link displayed by Visdom in your browser.


## Citation

Please use the following bibtex entry:
```
@article{adhikari2020gata,
  title={Learning Dynamic Belief Graphs to Generalize on Text-Based Games},
  author={Adhikari, Ashutosh and Yuan, Xingdi and C\^ot\'{e}, Marc-Alexandre and Zelinka, Mikul\'{a}\v{s} and Rondeau, Marc-Antoine and Laroche, Romain and Poupart, Pascal and Tang, Jian and Trischler, Adam and Hamilton, William L.},
  journal={CoRR},
  volume={abs/2002.09127},
  year= {2020},
  archivePrefix={arXiv},
  eprint={2002.09127}
}
```

## License

[MIT](./LICENSE)
