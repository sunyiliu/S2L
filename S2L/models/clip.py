import numpy
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

import transformers
transformers.logging.set_verbosity_error()


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    # def __init__(self, version="openai/clip-vit-huge-patch14", device="cuda", max_length=77):
    def __init__(self, path, device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
        self.transformer = CLIPTextModel.from_pretrained(path, subfolder='text_encoder')
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)
    

class TextEmbedder(nn.Module):

    def __init__(self, path, dropout_prob=0.1):
        super().__init__()
        self.text_encodder = FrozenCLIPEmbedder(path=path)
        self.dropout_prob = dropout_prob
    
    def token_drop(self, text_prompts, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = numpy.random.uniform(0, 1, len(text_prompts)) < self.dropout_prob
        else:
            # TODO
            drop_ids = force_drop_ids == 1
        labels = list(numpy.where(drop_ids, "", text_prompts))
        # print(labels)
        return labels

    def forward(self, text_prompts, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            text_prompts = self.token_drop(text_prompts, force_drop_ids)
        embeddings = self.text_encodder(text_prompts)
        return embeddings
    

if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = TextEmbedder(path='/mnt/petrelfs/maxin/work/pretrained/stable-diffusion-2-1-base',
                                dropout_prob=0.00001).to(device)

    text_prompt = [["a photo of a cat", "a photo of a cat"], ["a photo of a dog", "a photo of a cat"], ['a photo of a dog human', "a photo of a cat"]]
    # text_prompt = ('None', 'None', 'None')
    output = text_encoder(text_prompts=text_prompt, train=False)
    # print(output)
    print(output.shape)
    # print(output.shape)