import torch
import torchtext
import random
from utils import *
from dataset import Dataset

input_lang, output_lang, pairs = prepareData('spa.txt')
print("Random")
print(random.choice(pairs))

print("")
print(output_lang.indexesFromSentence('tengo mucha sed .'))
print(output_lang.sentenceFromIndex([68, 5028, 135, 4]))

# separamos datos en train-test
train_size = len(pairs) * 80 // 100
train = pairs[:train_size]
test = pairs[train_size:]

# Datasets
dataset = {
    'train': Dataset(input_lang, output_lang, train),
    'test': Dataset(input_lang, output_lang, test)
}

input_sentence, output_sentence = dataset['train'][1]
print(input_sentence, output_sentence)

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=64, shuffle=True, collate_fn=dataset['train'].collate),
    'test': torch.utils.data.DataLoader(dataset['test'], batch_size=256, shuffle=False, collate_fn=dataset['test'].collate),
}

inputs, outputs = next(iter(dataloader['train']))
print(inputs.shape, outputs.shape)
