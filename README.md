# Learning Dynamic Knowledge Graphs to Generalize on Text-based Games
---------------------------------------------------------------------------
Code for paper [Learning Dynamic Knowledge Graphs to Generalize on Text-based Games](https://arxiv.org/abs/2002.09127).

```
# Dependencies
conda create -p /tmp/gata python=3.6 numpy scipy ipython matplotlib cython nltk pillow
source activate /tmp/gata
pip install --upgrade pip
pip install numpy==1.16.2
pip install gym==0.15.4
pip install https://github.com/microsoft/TextWorld/archive/patch_disable_vocab__auto_extraction.zip
pip install -U spacy
python -m spacy download en
pip install tqdm pipreqs h5py pyyaml visdom
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Pretraining Action Prediction / State Prediction / Deep Graph Infomax / Command Generation
```
# Word embeddings
wget "https://bit.ly/38gklt3"

# Action Prediction
cd ap.0.2 ; wget https://bit.ly/2v6nbC1 ; unzip ap.0.2.zip ; cd ..
# Modify config.yaml
python train_action_prediction.py

# Deep Graph Infomax
cd dgi.0.2 ; wget https://bit.ly/383vAEQ ; unzip dgi.0.2.zip ; cd ..
# Modify config.yaml
python train_deep_graph_infomax.py

# State Prediction
cd sp.0.2 ; wget https://bit.ly/2Uyj9wS ; unzip sp.0.2.zip ; cd ..
# Modify config.yaml
python train_state_prediction.py

# Command generation
cd cmd_gen.0.2 ; wget https://bit.ly/385UVOC ; unzip cmd_gen.0.2.zip ; cd ..
# Modify config.yaml
python train_command_generation.py config.yaml
```

## Training RL agents

```
# Download games
cd rl.0.1 ; wget https://bit.ly/3877id7 ; unzip rl.0.1.zip ; cd ..

# Modify config.yaml
python train_rl.py config.yaml
```

## Citation

Please use the following bibtex entry:
```
@article{yuan2019imrc,
  title={Interactive Machine Comprehension with Information Seeking Agents},
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
