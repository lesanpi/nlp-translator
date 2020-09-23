import torch
import torchtext
import random
from utils import *
from dataset import Dataset
from RNN import Encoder, Decoder
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

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
print(input_lang.sentenceFromIndex(input_sentence.tolist()),
      output_lang.sentenceFromIndex(output_sentence.tolist()))

encoder = Encoder(input_size=input_lang.n_words)
hidden = encoder(torch.randint(0, input_lang.n_words, (64, 10)))

decoder = Decoder(input_size=output_lang.n_words)
output, decoder_hidden = decoder(torch.randint(
    0, output_lang.n_words, (64, 1)), hidden)


def fit(encoder, decoder, dataloader, epochs=10):
    encoder.to(device)
    decoder.to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        encoder.train()
        decoder.train()
        train_loss = []
        bar = tqdm(dataloader['train'])
        for batch in bar:
            input_sentences, output_sentences = batch
            bs = input_sentences.shape[0]
            loss = 0
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            # obtenemos el último estado oculto del encoder
            hidden = encoder(input_sentences)
            # calculamos las salidas del decoder de manera recurrente
            decoder_input = torch.tensor(
                [[output_lang.word2index['SOS']] for b in range(bs)], device=device)
            for i in range(output_sentences.shape[1]):
                output, hidden = decoder(decoder_input, hidden)
                loss += criterion(output, output_sentences[:, i].view(bs))
                # el siguiente input será la palbra predicha
                decoder_input = torch.argmax(output, axis=1).view(bs, 1)
            # optimización
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            train_loss.append(loss.item())
            bar.set_description(
                f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f}")

        val_loss = []
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            bar = tqdm(dataloader['test'])
            for batch in bar:
                input_sentences, output_sentences = batch
                bs = input_sentences.shape[0]
                loss = 0
                # obtenemos el último estado oculto del encoder
                hidden = encoder(input_sentences)
                # calculamos las salidas del decoder de manera recurrente
                decoder_input = torch.tensor(
                    [[output_lang.word2index['SOS']] for b in range(bs)], device=device)
                for i in range(output_sentences.shape[1]):
                    output, hidden = decoder(decoder_input, hidden)
                    loss += criterion(output, output_sentences[:, i].view(bs))
                    # el siguiente input será la palbra predicha
                    decoder_input = torch.argmax(output, axis=1).view(bs, 1)
                val_loss.append(loss.item())
                bar.set_description(
                    f"Epoch {epoch}/{epochs} val_loss {np.mean(val_loss):.5f}")
