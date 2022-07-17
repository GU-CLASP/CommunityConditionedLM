------------------------------------------------------------------------

### Instructions

The author response period has begun. The reviews for your submission
are displayed on this page. If you want to respond to the points raised
in the reviews, you may do so in the boxes provided below.

Please note: *you are not obligated to respond to the reviews*.

------------------------------------------------------------------------

For reference, you may see the review form that reviewers used to
evaluate your submission. If you do not see some of the filled-in fields
in the reviews below, it means that they were intended to be seen only
by the committee. See the review form
[HERE](https://www.softconf.com/coling2022/papers/user/scmd.cgi?scmd=reviewFormCustom_show&passcode=1107X-H3H6G3B2J6){target="_blank"}.

------------------------------------------------------------------------

### Review #1

>   -------------------------------------------- ---
>                           **Relevance** (1-5): 4
>                 **Readability/clarity** (1-5): 5
>                         **Originality** (1-5): 3
>     **Technical correctness/soundness** (1-5): 4
>                     **Reproducibility** (1-5): 3
>                           **Substance** (1-5): 3
>   -------------------------------------------- ---
>
> #### Detailed Comments
>
> > The study compares linguistic and social \"community\" embeddings,
> > by testing how well linguistic representations correlate with social
> > network-based representation of communities. In particular,
> > subreddits are considered as a community and three conditions are
> > examined: pairs of subreddits with high social similarity and high
> > linguistic similarity, subreddits showing low social similarity but
> > high linguistic similarity and pairs with low linguistic but high
> > social similarity. Results are quite puzzling and are not presented
> > as clearly conclusive. However, the paper is very well written and
> > the methodology employed is well described: although some further
> > qualitative analysis could have been beneficial to the work, it is
> > nonetheless worth presenting as a first approach to correlating
> > (socio)linguistic embeddings with pure network based social
> > representations. With respect to the employment of linguistic
> > embeddings, one of the major weaknesses and obstacles to more
> > conclusive results could be found in the high lexical bias
> > introduced by linguistic embeddings. It would be interesting to see
> > the same methodology applied to more abstract linguistic
> > representations (i.e., based on syntactic profiling for instance).
> >
> > Minor comments/typos:
> >
> > -   on line 47, there\'s a repetition of the word \"our\"
> >
> > -   on line 132, the 3rd footnote can be written as an inline
> >     reference to section 7
> >
> > -   on line 313, reference to \"fig2, left\" should probably be
> >     \"fig 2, centre\"
> >
> > -   on line 487, reference to fig 4 should be to fig 3, as fig 4 is
> >     in the appendix
>
>   ------------------------------------------ --------
>            **Overall recommendation** (1-5): 4
>                        **Confidence** (1-5): 3
>                       **Presentation Type**: Poster
>     **Recommendation for Best Paper Award**: No
>   ------------------------------------------ --------


### Review #2

>   -------------------------------------------- ---
>                           **Relevance** (1-5): 5
>                 **Readability/clarity** (1-5): 4
>                         **Originality** (1-5): 4
>     **Technical correctness/soundness** (1-5): 3
>                     **Reproducibility** (1-5): 5
>                           **Substance** (1-5): 3
>   -------------------------------------------- ---
>
> #### Detailed Comments
>
> > Contributions
> >
> > This paper trains two types of community-conditioned language models
> > (an LSTM and a transformer model) using Reddit data. These models
> > differ from traditional language models in that they concatenate a
> > community embedding parameter, which varies depending on the
> > community, to some layers of the model. The authors show that their
> > approach leads to lower perplexity and higher information gain over
> > a non-conditioned LM.
> >
> > The authors then show that these community embeddings are correlated
> > with embeddings obtained using user-community co-occurrence in two
> > ways. In the first approach, they compare pairwise similarities
> > between communities across the two vector spaces, but demonstrate
> > that the correlation estimate from this is flawed. In their second
> > approach, they align the embeddings across the two spaces using
> > orthogonal procrustes.
> >
> > Strengths
> >
> > The premise and approach of this paper is very interesting. The
> > experiments are carefully done.
> >
> > One of the more interesting findings for me was described in lines
> > 235 about which communities your approach was more effective for. If
> > you could formally test this and connect it with sociolinguistics
> > literature on audience design, I feel like this claim could be
> > illustrated more strongly. That is, you could quantitatively compare
> > the effect of conditioning on community embeddings for subreddits
> > that have a broad audience versus those that have special interests.
> > The challenge here would be to find some way to operationalize the
> > idea of a broad or narrow audience. One possible way would be to
> > compare \"default" subreddits (ones that users are subscribed to
> > when they sign up on Reddit) to non-default ones. Default subreddits
> > were discontinued in around 2017, but your dataset is from 2015, so
> > this may be feasible.
> >
> > Weaknesses
> >
> > In Gonen et al.\'s ACL 2020 paper \"Simple, Interpretable and Stable
> > Method for Detecting Words with Usage Change across Corpora", they
> > describe how the approach of aligning embeddings across different
> > spaces using the Orthogonal Procrustes method can lead to a
> > self-contradicting objective if these embedding spaces can differ
> > from each other in valid ways (e.g. as shown in your section 4.1 for
> > your situation). Please clarify whether your use of Orthogonal
> > Procrustes in the process of measuring the correlation of
> > language-based embeddings with social-network-based ones also has
> > this pitfall. Wouldn\'t finding the minimum distance between
> > transformed L and S lead to overestimating the correlation between
> > them?
> >
> > This paper lacks clear motivation for why I would want community
> > embeddings in the first place, and what the intended downstream use
> > of those embeddings could be.
> >
> > In line 51, the authors state that they \"test how the resulting
> > embeddings correspond to the social structure of subreddits." Later,
> > they examine communities that have high social similarity but low
> > linguistic similarity, and those that have low social similarity and
> > high linguistic similarity. It is unclear to me if the comparison of
> > language-based embeddings to social-network-based ones is meant to
> > be validation for how well language-based embeddings represent a
> > community, or as a way of answering the substantive question of how
> > language and social interactions can align or diverge in online
> > spaces. I think the former would be incorrect to assume:
> > language-based embeddings should not be evaluated against social
> > ones, due to the existence of communities with similar topics but
> > very different audiences (e.g. r/malefashionadvice and
> > r/femalefashionadvice). Clarification of the intentions behind the
> > experiments in section 4.1 is needed.
> >
> > The authors state that \"the focus of this paper is sociolinguistic
> > aspects" in line 544. If this is actually the case, I would have
> > liked to see more discussion (similar to what\'s done in lines
> > 590-592) on how their results may reflect or refute prior findings
> > from sociolinguistics.
> >
> > As in many NLP papers, I find the process for conducting the
> > qualitative analyses in this paper not well-described. It seems like
> > what the authors do in this present work is most similar to
> > qualitative coding methods that categorize text examples into
> > themes. However, the authors do not really describe how they do this
> > analysis, and whether it follows standards set forth by grounded
> > theory.
> >
> > A stronger version of this paper would demonstrate how
> > community-conditional language models lead to better performance on
> > some downstream NLP task that involves better handling of
> > community-specific language, aside from just language modeling.
> >
> > Questions / Other comments
> >
> > In the paper you experiment with two types of models: an LSTM and a
> > transformer-based model. These two seem like they have similar
> > performance based on Figure 1. It might be useful to report the
> > difference in training speed and model size. This would better
> > inform future readers on how they should choose which model to use,
> > e.g. the possible tradeoff, if any, between slightly better
> > perplexity but the inefficiency of a larger model.
> >
> > In the last paragraph of the introduction you mention that you
> > present a new novel method for testing the correlation between two
> > embeddings from different models. You should provide some brief
> > overview of your method here for readers who initially skim your
> > paper and are deciding whether to read further or not. This would
> > highlight your contributions in a more concrete and less vague
> > manner.
> >
> > Why did you sample 42000 messages? 42000 seems like an arbitrary
> > cutoff.
> >
> > Why did you use data from 2015? I only ask this out of curiosity
> > because your submission is in 2022, and Pushshift has more recent
> > data.
> >
> > Line 455 to 458 could be reworded for clarity. That is, the sentence
> > essentially says \"d(L, S) exhibits a mean of \[X\] and a std of
> > \[X\] in their distance from the social embedding, S", but d(L,S) is
> > a distance, and so \"their distance" does not make sense if \"their"
> > refers to d(L,S).
> >
> > What is the purpose of Figure 3? How is it different from Figure 4?
> > What is the main takeaway from this figure?
> >
> > Line 589: \"are positively correlation" -\> \"are positively
> > correlated"
>
>   ------------------------------------------ --------
>            **Overall recommendation** (1-5): 2
>                        **Confidence** (1-5): 3
>                       **Presentation Type**: Poster
>     **Recommendation for Best Paper Award**: No
>   ------------------------------------------ --------

### Review #3

>   -------------------------------------------- ---
>                           **Relevance** (1-5): 4
>                 **Readability/clarity** (1-5): 4
>                         **Originality** (1-5): 3
>     **Technical correctness/soundness** (1-5): 3
>                     **Reproducibility** (1-5): 2
>                           **Substance** (1-5): 1
>   -------------------------------------------- ---
>
> #### Detailed Comments
>
> > The paper addresses an interesting socio-linguistic question: to
> > what degree do communities shape linguistic variation? It
> > investigates linguistic variation occurring across 510 subreddits
> > and finds that linguistic variation across reddit posts correlates
> > with the subreddit (in lieu of community) they have been published
> > in.
> >
> > Even though the paper discusses at length there relation between
> > social embeddings and linguistic distinctiveness, it lacks a clear
> > description of the linguistic features taken into account in this
> > study. (Presumably the linguistic features would be more
> > sophisticated than simple word token metrics, since those would be
> > thematically conditioned by the topic of the subreddits themselves.)
> > It is also not clear from the paper why subreddits would be useable
> > as convincing instances of social and linguistic communities.
>
>   ------------------------------------------ --------
>            **Overall recommendation** (1-5): 1
>                        **Confidence** (1-5): 4
>                       **Presentation Type**: Poster
>     **Recommendation for Best Paper Award**: No
>   ------------------------------------------ --------

------------------------------------------------------------------------

### Submit Response to Reviewers

Use the following boxes to enter your response to the reviews. Please
limit the total amount of words in your comments to 900 words (longer
responses will not be accepted by the system).

[Response to Review #1]{.underline}:

[Response to Review #2]{.underline}:

[Response to Review #3]{.underline}:

[General Response to Reviewers]{.underline}:

------------------------------------------------------------------------

### Response to Chairs

Use this textbox to contact the chairs directly only when there are
serious issues regarding the reviews. Such issues can include reviewers
who grossly misunderstood the submission, or have made unfair
comparisons or requests in their reviews. Most submissions should not
need to use this facility.

------------------------------------------------------------------------

[[START](http://www.softconf.com){target="_blank"} Conference Manager
(V2.61.0 - Rev. 6526)]{.small}

 
