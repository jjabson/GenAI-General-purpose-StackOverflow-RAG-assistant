*Note:* Since the dataset is extremely large, this file will show how you access/get the dataset.

This dataset contains questions and answers from the Stack Overflow Data Dump for the purpose of preference model training. Importantly, the questions have been filtered to fit the following criteria for preference models (following closely from Askell et al. 2021): have >=2 answers. This data could also be used for instruction fine-tuning and language model training.

**Source:** Hugging Face HuggingFaceH4/stack-exchange-preferences

**Example:** How to use the dataset:

```
from datasets import Dataset, concatenate_datasets, load_dataset

ds = load_dataset("HuggingFaceH4/stack-exchange-preferences", split="train")
```

Please go to Hugging Face website to see more info on the dataset.
