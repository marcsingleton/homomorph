# Homomorph
Homomorph is yet another implementation of essential HMM algorithms. (Its name is a reference to its similarity to the countless other HMM packages available on PyPI.) What sets Homomorph apart is its focus on flexibility, numerical stability, and clarity of exposition. For example, though [hmmlearn](https://github.com/hmmlearn/hmmlearn) is popular, it only natively supports HMMs with normal, mixture normal, or categorical emission distributions. In contrast, Homomorph can directly model states parametrized by arbitrary distributions. Unlike hmmlearn, it does not provide built-in methods for fitting HMMs to data since they would necessarily rely on general-purpose optimization algorithms. Instead, training methods are discussed for specific examples in the [training tutorial](https://github.com/marcsingleton/homomorph/blob/main/tutorials/hmm_training.ipynb). It covers many scenarios and approaches, including categorical and normal emission distributions with labeled and unlabeled data. Although the HMM class is fully documented, the [HMM tutorial](https://github.com/marcsingleton/homomorph/blob/main/tutorials/hmm_intro.ipynb) is the recommended introduction to the Homomorph API.

## Dependencies
Homomorph is designed to be lightweight and only requires NumPy for array calculations and SciPy for its implementation of random variables. The minimum versions were set to the most recent at the time of initial release (1.17 and 1.8, respectively), but no functionality specific to these releases is needed to my knowledge.

## Installation
To install Homomorph, run the following command:

```
pip install homomorph
```
