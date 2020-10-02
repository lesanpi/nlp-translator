import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ix):
        return torch.tensor(self.input_lang.indexesFromSentence(self.pairs[ix][0]), device=device, dtype=torch.long), \
            torch.tensor(self.output_lang.indexesFromSentence(
                self.pairs[ix][1]), device=device, dtype=torch.long)

    def collate(self, batch):
        # calcular longitud máxima en el batch
        max_input_len, max_output_len = 0, 0
        for input_sentence, output_sentence in batch:
            max_input_len = len(input_sentence) if len(
                input_sentence) > max_input_len else max_input_len
            max_output_len = len(output_sentence) if len(
                output_sentence) > max_output_len else max_output_len
        # añadimos padding a las frases cortas para que todas tengan la misma longitud
        input_sentences, output_sentences = [], []
        for input_sentence, output_sentence in batch:
            input_sentences.append(torch.nn.functional.pad(
                input_sentence, (0, max_input_len - len(input_sentence)), 'constant', self.input_lang.word2index['PAD']))
            output_sentences.append(torch.nn.functional.pad(
                output_sentence, (0, max_output_len - len(output_sentence)), 'constant', self.output_lang.word2index['PAD']))
        # opcionalmente, podríamos re-ordenar las frases en el batch (algunos modelos lo requieren)
        return torch.stack(input_sentences), torch.stack(output_sentences)
