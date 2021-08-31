# Large-Scale Impact Comparison of Small Changes in the Data on the Outcome and Internals of a Linear ML Classifier
### Uses natural language processing, machine learning, multiprocessing; framworks: scikit-learn, tensorflow

In this analysis, I systemtically asses the impact of small changes in the training data on the prediction performance as well as the internal weights assigned to individual features.

The underlying question is: *How usable are classifiers trained on one dataset for prediction tasks another differing only slightly*, e.g. in the number of authors or by point-in-time.
A high level of stabiltiy in this regard (especially in terms of feature relvenace) would imply a more causal relationship between input and output, while an unstable relationship
hints at correlational relationships which may break down at any point.

That question is especially relevant in the social sciences and for law enforcement as here systematic errors or instable input-output relationships adversely affect individuals.
As linear models are most-commonn in authorship analysis, and still outperform others (such as BERT), I focus on them.

The *experimental setup* and comparison is *implemented in dynAA.py*.

## Experimental Setup:

I train two linear classifer types, one linear SVM and one dynnAA stacked model akin as proposed in [Custodio et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0957417421003079)
on different types of input features (see table below, ranked in terms of captured context) and use them to predict the target gender or age. As variation, I vary the number of authors slightly. I repeat that for different tweet lengths.
That results in 4800 trained models which we can compare.

| Featuretype <sup>1</sup>| Description | N-gram range <sup>2</sup>| Variation No. Authors
| ----------- | ----------- |----------- |----------- |
| DIST | Character-based. Every non-special character is mapped to * | 2-5 | {50,150,500,1000}|
| CHAR | Individual non-special characters |  2-5 | {50,150,500,1000}|
| ASIS | Text is taken including special characters to chreate character-based tokens | 2-5 | {50,150,500,1000}|
| POS| Spacy POS Tokens | 1-3 | {50,150,500,1000}|
|  DEP| Spacy DEP Tokens | 1-3 | {50,150,500,1000}|
| LEMMA | Spacy Lemmas | 1-2 | {50,150,500,1000}|
| WORD | Word-based features | 1-2 | {50,150,500,1000}|

<sup>1</sup> Ranked in descending order in terms of captured context, e.g., topic, time-frame, domain.
<sup>2</sup> N-gram ranges are the best-performing ones as presented in [Custodio et al. 2021](https://www.sciencedirect.com/science/article/abs/pii/S0957417421003079)

## Small Overview over Results

As stated, we train 4800 models. Consequently, I only present a small excerpt of all results here.

In the following figure, we see aggregation boxplots for the *age* prediction task. They *show the performance* of the classifers *as well as the stability of feature relevance*
(calculated on basis of the weight matrix) in terms of Spearman's Rho.

<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/classifier_outcome_comparison/figures/age_f1_scores_500-1.jpg" width="500" height="350"/> 
<img src="https://github.com/mhschubert/Portfolio/blob/main/classifier_outcome_comparison/figures/age_spearman_ext_500-1.jpg" width="500" height="350"/>
</p>

In the left figure, we can see that the performance is comparable to what is reported in the literature of authorship analysis
(see Custodio et al. [2021] for a very recent overview). The number of authors affects that result, but not in such a way that we would need to be concerned.

However, any decline hints at the fact that patterns change or become more varied - otherwise we would see no decline because it is just more of the same.
This is supported by the right figure. When increasing the number of authors, the patterns change fundamentally. That meanst that features predictive before are now losing or gaining relevance.
That means while the predictive stability is high, the stability of the feautre relevance is small.

When we go onto the level of indiviudal authors, we can plot how often an individual is mis-classified.
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/classifier_outcome_comparison/figures/proportions_stacked_dist_char_asis_lemma_word_1_500_color_centered_age.png" width="1000" /> 
</p>
We clearly see a pattern. There are some authors who are misclassified so often that their accuracy lies below the random-guess threshold of 20%. For others, the model performs consistently well.
That means the former are consistently disadvantaged when authorship profiling is used on them in any law enforcement context as they might be confused with others and face negative consequences due to that.

Moreover, these authors belong to specific age groups, e.g., 1987. That means that the performance fails for very specific groups.
Hence, the analysis shows that we have to be careful when using authorship profiling on data containing a lot of such group members.

Overall, we see that using trained models in one context and changing the context slightly (e.g. in the number of authors)
may introduce unseen variation as most input-output relationships seem to be correlational in nature.
As a consequence, the error rate of a otherwise well-performing model might rise during its use on unseen data.
That brings negative normaive implications with it.
