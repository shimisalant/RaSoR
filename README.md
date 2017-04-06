# Learning Recurrent Span Representations for Extractive Question Answering
[https://arxiv.org/abs/1611.01436](https://arxiv.org/abs/1611.01436)

#### Requirements

[Theano](http://deeplearning.net/software/theano/install.html), [Matplotlib](http://matplotlib.org/), [Java](https://www.oracle.com/java/index.html)

#### Initial setup

```bash
$ python setup.py
```
This will download GloVe word embeddings and tokenize raw training / development data.<br />
(download will be skipped if [zipped GloVe file](http://nlp.stanford.edu/data/glove.840B.300d.zip) is manually placed in `data` directory).

#### Training

```bash
$ python rasor.py --device DEVICE --train
```
where `DEVICE` is `cpu`, or an indexed GPU specification e.g. `gpu0`.<br />
When specifying a certain GPU, the [theano device flag](http://deeplearning.net/software/theano/library/config.html#config.device) must be set to `cpu`, i.e. set `device=cpu` in your `.theanorc` file.

#### Making predictions

```bash
$ python rasor.py --device DEVICE test_json_path pred_json_path
```
where `test_json_path` is the path of a JSON file containing articles, paragraphs and questions (see [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) for specification of JSON structure), and `pred_json_path` is the path to write predictions to.

---

Tested in the following environment:

* Ubuntu 14.04
* Python 2.7.6
* NVIDIA CUDA 8.0.44 and cuDNN 5.1.5
* Theano 0.8.2
* Matplotlib 1.3.1
* Oracle JDK 8

