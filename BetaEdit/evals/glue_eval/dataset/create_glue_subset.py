from datasets import load_dataset
from useful_functions import save_data
import random
random.seed(37)
classwise_size = 100
dataset = load_dataset("glue", "sst2")
eval_dataset = dataset['validation']
classwise = {}
finalized_subset = []
for example in eval_dataset:
    if example['label'] not in classwise:
        classwise[example['label']] = [example]
    else:
        classwise[example['label']].append(example)
for label in classwise:
    random.shuffle(classwise[label])
classwise_size = min(len(examples) for examples in classwise.values())
index = 0
while len(finalized_subset) < classwise_size * len(classwise):
    for label in classwise:
        if index < len(classwise[label]):
            finalized_subset.append(classwise[label][index])
    index += 1
save_data('sst2.pkl', finalized_subset)
