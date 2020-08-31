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
pip install textworld
pip install -U spacy
python -m spacy download en
pip install tqdm pipreqs h5py pyyaml visdom
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Pretraining Action Prediction / State Prediction / Deep Graph Infomax / Command Generation
```
# Word embeddings
wget "https://bit.ly/2U3Mde2"

# Action Prediction
cd ap.0.2 ; wget https://aka.ms/twkg/ap.0.2.zip ; unzip ap.0.2.zip ; cd ..
# Modify config.yaml
python train_action_prediction.py

# Deep Graph Infomax
cd dgi.0.2 ; wget https://aka.ms/twkg/dgi.0.2.zip ; unzip dgi.0.2.zip ; cd ..
# Modify config.yaml
python train_deep_graph_infomax.py

# State Prediction
cd sp.0.2 ; wget https://aka.ms/twkg/sp.0.2.zip ; unzip sp.0.2.zip ; cd ..
# Modify config.yaml
python train_state_prediction.py

# Command generation
cd cmd_gen.0.2 ; wget https://aka.ms/twkg/cmd_gen.0.2.zip ; unzip cmd_gen.0.2.zip ; cd ..
# Modify config.yaml
python train_command_generation.py config.yaml
```

## Training RL agents

```
# Download games
cd rl.0.1 ; wget https://aka.ms/twkg/rl.0.1.zip ; unzip rl.0.1.zip ; cd ..

# Modify config.yaml
python train_rl.py config.yaml
```

### Monitoring training progress
To monitor training progress: set "`visdom: True`" in `config.yaml` under the `general` section, and start [Visdom](https://github.com/facebookresearch/visdom) in another terminal using the `visdom` command line. Then, open the link displayed by Visdom in your browser.


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
