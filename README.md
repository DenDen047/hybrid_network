# Hybrid Network

This open-source project, referred to as **HybridNetwork** aims to provide the scalable and extendable implementations of combining a spiking neural network (SNN) and an artificial neural network (ANN).
On one hand, this project enables a uniform workflow over SNNs, ANNs and cording layers leading to an in-depth understanding of the combination of SNN and ANN.
On the other hand, this project makes it easy to develop and incorporate newly proposed models, so as to expand the territory of techniques on hybrid network.

**Key Features**:

- A number of representative learning-to-rank models, including not only the traditional optimization framework via empirical risk minimization but also the adversarial optimization framework
- Supports widely used benchmark datasets. Meanwhile, random masking of the ground-truth labels with a specified ratio is also supported
- Supports different metrics, such as Precision, MAP, nDCG and nERR
- Highly configurable functionalities for fine-tuning hyper-parameters, e.g., grid-search over hyper-parameters of a specific model
- Provides easy-to-use APIs for developing a new learning-to-rank model

Please refer to the [documentation site](https://wildltr.github.io/ptranking/) for more details.


## Usage

```sh
$ ./run0.sh
$ ./run1.sh
```
