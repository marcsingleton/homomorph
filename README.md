# Homomorph
Homomorph is yet another implementation of essential HMM algorithms in Python. (Its name is a reference to its similarity to the countless other HMM packages available on PyPI.) What sets Homomorph apart is its focus on flexibility, numerical stability, and clarity of exposition. For example, though [hmmlearn](https://github.com/hmmlearn/hmmlearn) is popular, it only natively supports HMMs with normal, mixture normal, or categorical emission distributions. In contrast, Homomorph can directly model states parametrized by arbitrary distributions. Unlike hmmlearn, it does not provide built-in methods for fitting HMMs to data since they would necessarily rely on general-purpose optimization algorithms. Instead, training methods are discussed for specific examples in the [training tutorial](https://github.com/marcsingleton/homomorph/blob/main/tutorials/hmm_training.ipynb). It covers many scenarios and approaches, including categorical and normal emission distributions with labeled and unlabeled data. Although the HMM class is fully documented, the [HMM tutorial](https://github.com/marcsingleton/homomorph/blob/main/tutorials/hmm_intro.ipynb) is the recommended introduction to the Homomorph API.

## Dependencies
Homomorph is designed to be lightweight and only requires NumPy for array calculations and SciPy for its implementation of random variables. The minimum version for NumPy was set to the most recent at the time of initial release (1.17), but no functionality specific to this release is needed to my knowledge. SciPy, on the other hand, must be at least version 1.9. The reasons for this are highly technical, but the core issue was previous versions of SciPy did not strongly distinguish between discrete and continuous frozen random variables. As a result, reliably extracting the correct probability functions (the pmf and pdf, respectively) was extremely difficult. Although Homomorph does not directly use SciPy in its implementation, its practical use depends so strongly on it that this ambiguity effectively broke Homomorph. Fortunately, versions 1.9 and later have corrected this problem.

## Installation
To install Homomorph, run the following command:

```
pip install homomorph
```
