# NLP Newsletter: **Tokenizers, TensorFlow 2.1, TextVectorization, TorchIO, NLP Shortfalls,…**

----------


![](https://cdn-images-1.medium.com/max/2400/1*xAQ_xFWA3JXJNtLAEO7WYg.png)


Hello and Happy New Year! Due to numerous requests, I have decided to bring back the **NLP Newsletter**. This time I will keep it short and focused (also maintained in this [repo](https://github.com/dair-ai/nlp_newsletter)). The objective of this newsletter is to keep you informed on some of the interesting and recent stories related to NLP and ML (across a few categories) without taking too much time from your busy day.


# Publications 📙

***AI system for breast cancer screening***
DeepMind published a new paper in Nature titled “[International evaluation of an AI system for breast cancer screening](https://www.nature.com/articles/s41586-019-1799-6)”. According to the authors, the work is about the evaluation of an AI system that surpasses human experts in breast cancer screening. Whether this is actually achievable by current AI systems is still up for debate and there has been continuous criticism on this type of system and how they are evaluated. Here is [short summary](https://www.nature.com/articles/d41586-019-03822-8) of the paper. 

***Information Extraction***
Pankaj Gupta publicly released his Ph.D. Thesis titled “[Neural Information Extraction From Natural Language Text](https://www.researchgate.net/publication/336739252_PhD_Thesis_Neural_Information_Extraction_From_Natural_Language_Text)”. The main discussion is how to efficiently extract the semantic relationships from natural language text using neural-based approaches. Such research effort aims to contribute to building structured knowledge bases, that can be used in a series of downstream NLP applications such as web search, question-answering, among other tasks. 

***Improved recommendations***
Researchers at MIT and IBM developed a [method](http://news.mit.edu/2019/finding-good-read-among-billions-of-choices-1220) (published at NeurIPS last year) for categorizing, surfacing, and searching relevant documents based on a combination of three widely-used text-analysis tools: *topic modeling*, *word embeddings*, and *optimal transport*. The method also gives promising results for sorting documents. Such methods are applicable in a wide variety of scenarios and applications that require improved and faster suggestions on large-scale data such as search and recommendation systems.


# ML and NLP Creativity and Society 🎨

***AI careers***
The 2019 AI Index [report](https://hai.stanford.edu/sites/g/files/sbiybj10986/f/ai_index_2019_report.pdf) suggests that there is more demand that there is a supply of AI practitioners. However, there are various aspects of AI-related jobs such as career transitions and interviews that are still not properly defined. 

In this [post](https://towardsdatascience.com/how-i-found-my-current-job-3fb22e511a1f), Vladimir Iglovivok goes into great detail on his career and ML adventure from building traditional recommender systems to building spectacular computer vision models that won competitions on Kaggle. He now works on autonomous vehicles at Lyft but the [journey](https://towardsdatascience.com/how-i-found-my-current-job-3fb22e511a1f) of getting there wasn’t so easy. 

If you are really interested and serious about a career in AI, Andrew Ng’s company, deeplearning.ai, founded Workera, which aims to specifically help data scientists and machine learning engineers with their AI careers. Obtain their official [report](https://workera.ai/candidates/report) here. 


# ML/NLP Tools and Datasets ⚙️

***An ultra-fast tokenizer***
Hugging Face, the NLP startup behind Transformers, has open-sourced Tokenizers, an ultra-fast implementation of tokenization that can be used in modern NLP pipelines. Check out the [GitHub repo](https://github.com/huggingface/tokenizers) for the documentation on how to use Tokenizers.

![](https://cdn-images-1.medium.com/max/1600/1*BGcXk6Yf9fXGZlEtxz1hcg.jpeg)


*Tokenizers — Python bindings*

TensorFlow 2.1 incorporates a new [TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing/TextVectorization) layer which allows you to easily deal with raw strings and efficiently perform text normalization, tokenization, n-grams generation, and vocabulary indexing. Read the release here and check out Chollet’s [Colab notebook](https://colab.research.google.com/drive/1RvCnR7h0_l4Ekn5vINWToI9TNJdpUZB3) demonstrating how to use the feature for end-to-end text classification.

***NLP and ML for Search***
One of the fields that made tremendous progress this past year was NLP with a range of improvements and new research directions. One of those domains that could potentially benefit from transfer learning NLP is *search*. 

Although search belongs to the field of information retrieval there is an opportunity to build search engines that improve semantic search using modern NLP techniques such as contextualized representations from a transformer-based model like [BERT](https://arxiv.org/abs/1810.04805). Google released a [blog post](https://www.blog.google/products/search/search-language-understanding-bert/) a couple of months back discussing how they are leveraging BERT models for improving and understanding searches. 

If you are curious about how contextualized representations can be applied to search using open-search technologies such as Elasticsearch and TensorFlow, you can take a look at either this [post](https://towardsdatascience.com/elasticsearch-meets-bert-building-search-engine-with-elasticsearch-and-bert-9e74bf5b4cf2) or this [one](https://towardsdatascience.com/building-a-search-engine-with-bert-and-tensorflow-c6fdc0186c8a). 

***Medical image analysis***
[TorchIO](https://github.com/fepegar/torchio) is a Python package based on the popular deep learning library called PyTorch. TorchIO offers functionalities to easily and efficiently read and sample 3D medical images. Features include spatial transforms for data augmentation and preprocessing. 

![](https://cdn-images-1.medium.com/max/1600/0*FSPuSC8TK9X-NQ2q.gif)


[source](https://github.com/fepegar/torchio)


# Ethics in AI 🚨

***Fraudulent behavior in ML community***
This just came in! 1st Place winners of a Kaggle contest were disqualified for fraudulent activity. The team used clever but irresponsible and unacceptable tactics to win first place in the competition. Here is the [full story](https://www.kaggle.com/c/petfinder-adoption-prediction/discussion/125436). This story highlights one in many of the serious and unacceptable behaviors that the machine learning community wants to mitigate. The proper and ethical use of ML technologies is the only way forward. 

***Gender bias in machine translation***
On the topic of whether machine translation systems reflect gender bias, a group of researchers published this excellent [paper](https://arxiv.org/abs/1809.02208) presenting a case study using Google Translate. One of the findings claimed by the authors is that Google Translate “exhibits a strong tendency towards male defaults, in particular for fields linked to unbalanced gender distribution such as STEM jobs.”

***ML Bias and Fairness***
If you want to get all caught up with AI ethics and fairness, this is a nice [podcast episode](https://twimlai.com/twiml-talk-336-trends-in-fairness-and-ai-ethics-with-timnit-gebru/) featuring Timnit Gebru and hosted by TWIML. 

Timnit is a prominent researcher in ML fairness who together with Eun Seo Jo, published a [paper](https://arxiv.org/abs/1912.10389) where they identify five key approaches in document collection practices in archives that can provide more reliable methods for data collection in sociocultural ML. This could potentially lead to more systematic data collection methods gained from interdisciplinary collaborative research. 
Sina Fazelpour and Zachary Lipton recently published a [paper](http://zacklipton.com/media/papers/fairness-non-ideal-fazelpour-lipton-2020.pdf) where they argue that due to the nature of how our non-ideal world arose it is possible that fair ML based on the ideal thinking can potentially lead to misguided policies and interventions. In fact, their analysis demonstrates “that shortcomings of proposed fair ML algorithms reflect broader troubles faced by the ideal approach.”


# Articles and Blog posts ✍️

***NLP shortfalls***
Benjamin Heinzerling published an interesting article in The Gradient where he discusses areas where NLP falls short such as argument comprehension and commonsense reasoning. Benjamin makes reference to a recent [paper](https://www.aclweb.org/anthology/P19-1459/) by Nivin & Kao that challenges and questions the capabilities of transfer learning and language models for high-level natural language understanding. [Read](https://thegradient.pub/nlps-clever-hans-moment-has-arrived/) more about this excellent summary of the analysis performed in the research.

***2019 NLP and ML Highlights***
For the new year, I released a [report](https://medium.com/dair-ai/nlp-year-in-review-2019-fb8d523bcb19) documenting some of the most interesting NLP and ML highlights that I came across in 2019. 
Sebastian Ruder also recently wrote an excellent and detailed [blog post](https://ruder.io/research-highlights-2019/) about the top ten ML and NLP research directions that he found impactful in 2019. Among the list are topics such as universal unsupervised pretraining, ML and NLP applied to science, augmenting pretrained models, efficient and long-range Transformers, among others. 

![](https://cdn-images-1.medium.com/max/1600/0*8zoPc5OnYERIaaMP.png)


*“VideoBERT (*[*Sun et al., 2019*](https://arxiv.org/abs/1904.01766)*), a recent multimodal variant of BERT that generates video “tokens” given a recipe (above) and predicts future tokens at different time scales given a video token (below).” —* [*source*](https://arxiv.org/pdf/1904.01766.pdf)

Google AI Research publishes a [summary](https://ai.googleblog.com/2020/01/google-research-looking-back-at-2019.html) of the research they conducted over the year and the future research directions they are paying attention to. 


# ML/NLP Education 🎓

***Democratizing AI education***
In an effort to democratize AI education and to educate the masses about the implications of AI technology, the University of Helsinki partnered with Reaktor to release a brilliant free course covering AI fundamentals. The [popular course](https://www.elementsofai.com/) is called “Elements of AI” and includes topics such as AI ethics, AI philosophy, neural networks, Naive Bayes rule, among other foundational topics.
Stanford CS224N is back with another [iteration](http://web.stanford.edu/class/cs224n/) of the popular “Natural Language Processing with Deep Learning” course. The course officially started January 7 of this year so if you want to follow, go to their website for the full syllabus, slides, videos, paper reading suggestions, etc.

***Top NLP and ML Books***
I tweeted my top book recommendations for theoretical and practical NLP and ML, it was well-received. I would like to share that list here via the tweet:

***Machine Learning with Kernel Methods***
Kernel methods such as PCA and K-means have been around for quite some time and that’s because they have been successfully applied for a wide variety of applications such as graphs and biological sequences. Check out this comprehensive set of [slides](http://members.cbio.mines-paristech.fr/~jvert/svn/kernelcourse/slides/master2017/master2017.pdf) covering a wide range of kernel methods and their inner workings. Here is also a great [blog](https://francisbach.com/cursed-kernels/) (maintained by Francis Bach) discussing aspects of kernel methods and other machine learning topics. 


# Notable Mentions ⭐️

Here is a list of noteworthy stories that are worth your attention:

- John Langford runs this incredible [blog](https://hunch.net/) discussing machine learning theory
- Many of the industry ML-oriented technologies have been using Gradient Boosting machines for years. Check out this [post](https://opendatascience.com/xgboost-enhancement-over-gradient-boosting-machines/?utm_campaign=Learning%20Posts&utm_content=111061559&utm_medium=social&utm_source=twitter&hss_channel=tw-3018841323) introducing one of the libraries used to apply gradient boosting called XGBoost. 
- If you are interested in learning how to design and build machine learning-powered applications and take them to production, Emmanuel Ameisen has you covered with this [book](https://www.amazon.com/Building-Machine-Learning-Powered-Applications/dp/149204511X/). 
----------

*If you have a story that you would like to see hosted in the next edition of the NLP Newsletter, please send me an email to ellfae@gmail.com or send me a message via* [*Twitter*](https://twitter.com/omarsar0)*.* 

