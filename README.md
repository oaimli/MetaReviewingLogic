# A Sentiment Consolidation Framework for Meta-Review Generation

In this project, we hypothesize that there is a hierarchical sentiment consolidation process to write scientific meta-reviews for human meta-reviewers.
Following the consolidation framework, we perform human annotation to extract judgements in the scientific reviews, and we find that GPT-4 has the ability to do this extraction task.
Then, we design enhanced prompting approaches with consideration of the sentiment consolidation framework, and experiments show that integration of the sentiment consolidation logic improves the generation quality in terms of sentiment related metrics.

```
/
├── annotation_analysis/       --> (Analysis on our annotated data to calculate annotator agreement)
├── annotation_data/           --> (The data to be annotated, sampled from the crawled dataset)
├── automate_annotating/       --> (Using GPT-4 to automatically annotate the data)
├── enhancing_prompting/       --> (The python code and prompts for different prompting approaches, and generated results)
├── evaluating_consolidation/  --> (The implementation of FacetEval and FusionEval, and also other evaluation metrics)
├── human_evaluation/          --> (Human evaluation results for the generated meta-reviews)
├── manual_analysis/           --> (Case study on the generated meta-reviews of our approach)   
├── metareview_analysis/       --> (Other analysis of the generated results including dominant facets and sentiment consistency)
├── plot/                      --> (Figures used in the publication)
└── README.md                  --> (This readme file)
```


