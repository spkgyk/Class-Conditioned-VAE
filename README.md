# Class-Conditioned-VAE
Deep Generative Models: Class-conditioned Variational Autoencoder (VAE) for generating MNIST digits.

![1.1](images/1.1.png)

![1.1-sol](images/1.1-sol.png)

## 1.2 EM for mixture of multinomials (1 point)

Consider the following mixture-of-multinomials model to analyze a corpus of documents that are represented in the bag-of-words model. Specifically, assume we have a corpus of D documents and a vocabulary of W words from which every word in the corpus is token. We are interested in counting 
![1.2](images/1.2.png)

![1.2-sol-a](images/1.2-sol-a.png)
![1.2-sol-b](images/1.2-sol-b.png)

![2.1](images/2.1.png)

![2.1-sol-a](images/2.1-sol-a.png)
![2.1-sol-b](images/2.1-sol-b.png)


![3](images/3.png)

### Requirements
    1. Following the variational Bayes algorithm of the original VAE, derive the algorithm for this class-conditioned variant (2 points). Hint: You need to design the variational distribution and write down the variational lower bound.
    2. Implement the algorithm with ZhuSuan, and train the model on the whole training set of MNIST (2 points). Then, visualize generations of your learned model. Set y observed as {1, 2, ··· , K}, and generate multiple xs for each y using your learned model (1 point).

![3.1-sol](images/3.1-sol.png)
![3.2-sol](images/3.2-sol.png)