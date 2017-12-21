# MEDLINE abstracts Embedding with Neural Networks

## Setup

Install python packages:

```shell
pip install -r requirements.txt
```

Download MEDLINE citations:

```shell
wget -nH -nc ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.xml.gz -P data/medline
```

Extract abstracts and terms:

```shell
python -m medline.dataprep data/medline
```

## Supervised Embedding with ConvNets

### Word vectors

Random word vectors:

```shell
args="--len 3 4 5 --fl 100 --act relu --pool_d 0.5"

python -m medline.convnet $args
```

Word vectors from SkipGram model:

```shell
python -m medline.word2vec skipgram ./skipgram
python -m medline.convnet $args --wv ./skipgram
```

Word vectors from CBOW model:

```shell
python -m medline.wordvec cbow ./cbow
python -m medline.convnet $args --wv ./cbow
```

### Region sizes

Testing single region sizes:

```shell
args="--fl 100 --act relu --pool_d 0.5"

python -m medline.convnet $args --len 1 
python -m medline.convnet $args --len 2 
python -m medline.convnet $args --len 3 
python -m medline.convnet $args --len 4 
python -m medline.convnet $args --len 5 
python -m medline.convnet $args --len 6 
python -m medline.convnet $args --len 7 
python -m medline.convnet $args --len 10
python -m medline.convnet $args --len 15
```

Testing multiple region sizes:

```shell
args="--fl 100 --act relu --pool_d 0.5"

python -m medline.convnet $args --len 2 3 4
python -m medline.convnet $args --len 4 5 6
python -m medline.convnet $args --len 4 4 4
python -m medline.convnet $args --len 3 4 5 6
python -m medline.convnet $args --len 4 4 4 4
```

### Activation

Testing different activation functions:

```shell
args="--len 4 --fl 100 --pool_d 0.5"

python -m medline.convnet $args
python -m medline.convnet $args --act tanh
python -m medline.convnet $args --act sigmoid
python -m medline.convnet $args --act softplus
```

### Dropout

Testing different dropout rates:

```shell
args="--len 4 --fl 300"

python -m medline.convnet $args --pool_d 0.0
python -m medline.convnet $args --pool_d 0.1
python -m medline.convnet $args --pool_d 0.2
python -m medline.convnet $args --pool_d 0.3
python -m medline.convnet $args --pool_d 0.4
python -m medline.convnet $args --pool_d 0.6
python -m medline.convnet $args --pool_d 0.7
python -m medline.convnet $args --pool_d 0.8
python -m medline.convnet $args --pool_d 0.0 --maxnorm 0
```

### Number of filters

Testing different number of filters:

```shell
args="--len 4 --act tanh --pool_d 0.0"

python -m medline.convnet $args --fl 50
python -m medline.convnet $args --fl 100
python -m medline.convnet $args --fl 200
python -m medline.convnet $args --fl 300
python -m medline.convnet $args --fl 400
python -m medline.convnet $args --fl 500
python -m medline.convnet $args --fl 700
python -m medline.convnet $args --fl 1000
```

### Batch normalization

Testing batch normalization at diffent layers:

```shell
args="--len 4 --fl 1000 --act tanh --pool_d 0.0"

python -m medline.convnet $args --conv_bnorm True
python -m medline.convnet $args --pool_bnorm True
python -m medline.convnet $args --conv_bnorm True --pool_bnorm True
```

## Unsupervised Embedding with Doc2vec

Learn a doc2vec model:
- 'dbow'
- 'dm_concat' for DM model with concatenation
- 'dm_sum' for DM model with summation
- 'dm_mean' for DM model with mean

```shell
python -m medline.doc2vec dbow ./dbow
```

## Classification with Doc2vec

Learn a neural network model on top of a doc2vec model:

```shell
python -m medline.nnet_doc2vec ./dbow --dest ./nnet_dbow
```

## Visualizing groups of text embeddings with different MeSH terms

Create some groups of MeSH terms in yaml file:

```yaml
spine:
- Lumbar Vertebrae
- Thoracic Vertebrae
- Cervical Vertebrae

brain diseases:
- Dementia
- Hydrocephalus
- Epilepsy
```
Create a dataset for each group above. Each dataset will have 1000 texts per term:

```shell
mkdir datasets
python -m medline.dataviz.dataprep ./groups.yml 1000 ./datasets
```

Embed data with ConvNet:

```shell
mkdir embeddings
python -m medline.dataviz.convnet ./convnet/model.hdf5 ./datasets ./embeddings
```

or Doc2vec:

```shell
mkdir embeddings
python -m medline.dataviz.doc2vec ./dbow ./datasets ./embeddings
```

Scale embeddings:

```shell
mkdir embeddings_2d
python -m medline.dataviz.manifold ./embeddings tsne ./embeddings_2d
```

Visualize embeddings:

```shell
mkdir visualizations
python -m medline.dataviz.plot ./embeddings_2d/spine.csv ./visualizations/spine.png
```
