# END 2.0 Capstone project

### Author

* **Deepak Hazarika**

Wednesday, 1 September 2021


## Question and Answer model

This project takes inspiration from Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (https://arxiv.org/abs/2005.11401)

![](images/rag.png)

The model is a combination of Parametric and Non-parametric knowledge to generate factual answers to questions or any other related NLP task.

GPT is an example of a model that utilizes only parametric knowledge. That is to say, the entirety of its knowledge can be found within the weight values in its parameters that constitute the model.

![](images/nonparametric.png)

(source:-https://medium.com/machine-learning-intuition/retrieval-augmented-generation-rag-control-your-models-knowledge-and-hallucinations-ea3c6345a659)

One way to distinguish the two is thinking of parametric knowledge as the knowledge of AI's own, while non-parametric knowledge is something you retrieve from outside sources e.g a search engine on getting a query searches the web for the topic and returns relevant document as shown in the diagram above.

To be more specific , generator is paramteric model whereas retriever is non-parametric.

Since this is a scaled down implementation of ideas from RAG model , there will be 3 components

* An indexed document store of context

* A retriever model like Dense Passage Retriever.

* A generator model like BART.

### dense passage retriever

The goal of dense passage retriever (DPR) is to index all the M passages in a
low-dimensional and continuous space, such that it can retrieve efficiently the top
k passages relevant to the input question for the reader at run-time.

dense passage retriever (DPR) uses a dense encoder Ez which maps any text passage to a
d-dimensional real-valued vectors and builds an index for all the M passages that we will use for retrieval.

At run-time, DPR applies a different encoder Ex that maps the input question to a
d-dimensional vector, and retrieves k passages of which vectors are the closest to the
question vector. The similarity between the question and the passage is defined using
the dot product of their vectors.[[1]](#1)

 <img src="https://render.githubusercontent.com/render/math?math=\Large sim(x,z)%20=Embedding_{Q}(x)^{T}.Embedding_{doc}(z)\qquad(1)"> 

The simpler inner product function is used for similarity calculation to improve the
dense passage retriever by learning better encoders.

#### encoders for question and passage

Use pre-trained BERT encoders for question and passage

#### inference

During inference time, the passage encoder  <img src="https://render.githubusercontent.com/render/math?math=E_{z}">  will be applied to all the passages and index them using FAISS offline.

FAISS is an extremely efficient, open-source library for similarity search and
clustering of dense vectors, which can easily be applied to billions of vectors. 

Given a question *x* at run-time, we derive its embedding  <img src="https://render.githubusercontent.com/render/math?math=v_{x}=E_{x}(x)">  and retrieve the top k passages with embeddings closest to  <img src="https://render.githubusercontent.com/render/math?math=v_{x}"> .

#### Training the encoders

Training the encoders so that the dot-product similarity (Eq. (1)) becomes a good
ranking function for retrieval.The goal is to create a vector space such that relevant
pairs of questions and passages will have smaller distance (i.e., higher similarity )
than the irrelevant ones, by learning a better embedding function.[[1]](#1)

Let  <img src="https://render.githubusercontent.com/render/math?math=D%20=\{(x_{i},z_{i}^{%2B},z_{i,1}^{-},\ldots,z_{i,n}^{-})\}_{i=1}^n">  be the training data that
consists of *m* instances. Each instance contains one question  <img src="https://render.githubusercontent.com/render/math?math=x_{i}">  and one relevant
(positive) passage  <img src="https://render.githubusercontent.com/render/math?math=z_{i}^{%2B}"> , along with *n* irrelevant (negative) passages  <img src="https://render.githubusercontent.com/render/math?math=z_{i,j}^{-}"> . We optimize the loss function as the negative log likelihood of the positive passage:


 <img src="https://render.githubusercontent.com/render/math?math=\Large p(z|x)=\qquad%20L(x_{i},z_{i}^{%2B},z_{i,1}^{-},\ldots,z_{i,n}^{-})\qquad\qquad(2)"> 


 <img src="https://render.githubusercontent.com/render/math?math==\Large\qquad-log\Large\frac{e^{sim(x_{i},z_{i}^{%2B})}}{e^{sim(x_{i},z_{i}^{%2B})}%2B\sum_{j=1}^ne^{sim(x_{i},z_{i,j}^{-})}}"> 


#### Positive and negative passages

For retrieval problems, it is often the case that positive examples are available
explicitly, while negative examples need to be selected from an extremely large pool.
For instance, passages relevant to a question may be given in a QA dataset, or can be
found using the answer. All other passages in the collection, while not specified
explicitly, can be viewed as irrelevant by default. I consider any random passage from the corpus as negative.[[1]](#1)

### Knowledge-Augmented generator 

Given an input x and a retrieved document z, the knowledge-augmented encoder
defines 
 <img src="https://render.githubusercontent.com/render/math?math=\qquad p(y|z,x)"> .
We join x and z into a single sequence that we feed into a Transformer (distinct from
the one used in the retriever). This allows us to perform rich cross-attention between
x and z before predicting y. See Figure 1 for a concrete example.[[2]](#2)

![](images/knowledgeaugmented_generator.png)[[2]](#2)

The model leverage two components: 

* a retriever

 <img src="https://render.githubusercontent.com/render/math?math=\qquad p(z|x)">  with parameters that returns (top-K truncated) distributions over text passages given a query x 

* a generator

 <img src="https://render.githubusercontent.com/render/math?math=\qquad p(y_{i}|x,z,y_{1:i-1})"> that generates a current token based on a context of the previous i - 1 tokens  <img src="https://render.githubusercontent.com/render/math?math=\qquad y_{1:i-1})"> , the original
input x and a retrieved passage z.[[3]](#3)
To train the retriever and generator end-to-end, we treat the retrieved document as a latent variable.We marginalize the latent documents using token, which can predict each target token based on a different document. 

#### Token generator model

Token model we draw a different latent document for each target token and marginalize
accordingly. This allows the generator to choose content from several documents when
producing an answer. Concretely, the top K documents are retrieved using the retriever,
and then the generator produces a distribution for the next output token for each
document, before marginalizing, and repeating the process with the following output token, Formally, we define: [[3]](#3)

 <img src="https://render.githubusercontent.com/render/math?math=\Large \qquad p_{Token}(y|x) \approx \prod_{i}^n \sum_{z\in top-k(p(.|x))}  p(z|x)p(y_{i}|x,z,y_{1:i-1})"> 


#### pre trained BART as generator

The generator component  <img src="https://render.githubusercontent.com/render/math?math=\qquad p(y_{i}|x,z,y_{1:i-1})"> could be modelled using any encoder-decoder. I plan to use
BART-large , a pre-trained seq2seq transformer with 400M parameters. To combine the
input x with the retrieved content z when generating from BART, we simply concatenate them. BART was pre-trained using a denoising objective and a variety of different noising functions. 

#### Decoding

At test time, to approximate 

 <img src="https://render.githubusercontent.com/render/math?math=\qquad arg max_{y} p(y|x)"> 

in Token approach. It can be seen as a standard, autoregressive seq2seq generator with transition probability: 

 <img src="https://render.githubusercontent.com/render/math?math=\Large \qquad p(y_{i}|x,y_{1:i-1}) = \sum_{z\in top-k(p(.|x))}  p(z_{i}|x)p(y_{i}|x,z_{i},y_{1:i-1})"> 

we can plug  <img src="https://render.githubusercontent.com/render/math?math=\Large \qquad p(y_{i}|x,y_{1:i-1})"> 

into a standard beam decoder. [[3]](#3)

### loss function of Generator

calculate similarity of predicted y vs actual y to calculate the model loss.


### Hyper-parameters to consider

* beam depth : try different depth - 2 ,3 

* batch size - 

* variable learning rate

#### Summary 

##### Training strategy

* train the query encoder

* train the generator
 
##### References
<a id="1">[1]</a> 
@misc{karpukhin2020dense,
      title={Dense Passage Retrieval for Open-Domain Question Answering}, 
      author={Vladimir Karpukhin and Barlas Oguz and Sewon Min and Patrick Lewis and Ledell Wu and Sergey Edunov and Danqi Chen and Wen-tau Yih},
      year={2020},
      eprint={2004.04906},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

<a id="2">[2]</a>
@misc{guu2020realm,
      title={REALM: Retrieval-Augmented Language Model Pre-Training}, 
      author={Kelvin Guu and Kenton Lee and Zora Tung and Panupong Pasupat and Ming-Wei Chang},
      year={2020},
      eprint={2002.08909},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

<a id="3">[3]</a>
@misc{lewis2021retrievalaugmented,
      title={Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks}, 
      author={Patrick Lewis and Ethan Perez and Aleksandra Piktus and Fabio Petroni and Vladimir Karpukhin and Naman Goyal and Heinrich Küttler and Mike Lewis and Wen-tau Yih and Tim Rocktäschel and Sebastian Riedel and Douwe Kiela},
      year={2021},
      eprint={2005.11401},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

