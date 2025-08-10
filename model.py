import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# === ENCODER === #
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        # Freeze all layers
        for param in resnet.parameters():
            param.requires_grad_(False)

        # Unfreeze only the last convolutional block
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Keep layers up to layer4 (no avgpool)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Output: (B, 2048, 7, 7)
        self.embed = nn.Linear(2048, embed_size)
        self.drop = nn.Dropout(p=0.5)

    def forward(self, images):
        features = self.resnet(images)  # (B, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (B, 7, 7, 2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (B, 49, 2048)
        features = self.embed(features)  # (B, 49, embed_size)
        features = self.drop(features)
        return features  # (B, 49, embed_size)


# === ATTENTION === #
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (B, 49, encoder_dim)
        # decoder_hidden: (B, decoder_dim)
        att1 = self.encoder_att(encoder_out)               # (B, 49, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, att_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (B, 49)
        alpha = self.softmax(att)                          # (B, 49)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha


# === DECODER with ATTENTION === #
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, attention_dim=256):
        super(DecoderRNN, self).__init__()
        self.attention = Attention(embed_size, hidden_size, attention_dim)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + embed_size, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

    def forward(self, encoder_out, captions):
        embeddings = self.embed(captions[:, :-1])  # (B, T, embed)
        embeddings = self.drop1(embeddings)

        batch_size, max_len, _ = embeddings.size()
        h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(encoder_out.device)

        outputs = []
        for t in range(max_len):
            context, _ = self.attention(encoder_out, h[0])  # (B, embed)
            rnn_input = torch.cat((embeddings[:, t, :], context), dim=1).unsqueeze(1)  # (B, 1, embed*2)
            output, h = self.rnn(rnn_input, h)  # (B, 1, hidden)
            output = self.linear(self.drop2(output.squeeze(1)))  # (B, vocab_size)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # (B, T, vocab_size)
        return outputs


    def sample(self, encoder_out, max_len=20, temperature=1.0, top_k=None, start_token=0):
        """
        Generate captions with temperature and top-k sampling.

        Args:
            encoder_out (Tensor): (B, 49, embed_size) encoder features
            max_len (int): Maximum length of the caption
            temperature (float): Softmax temperature (default: 1.0)
            top_k (int or None): If set, restrict sampling to top-k tokens
            start_token (int): Index of the <BOS> token
        """
        batch_size = encoder_out.size(0)
        inputs = torch.full((batch_size,), start_token, dtype=torch.long, device=encoder_out.device)
        inputs = self.embed(inputs).unsqueeze(1)  # (B, 1, embed)

        h = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size, device=encoder_out.device)

        predicted_sentence = []

        for _ in range(max_len):
            context, _ = self.attention(encoder_out, h[0])  # (B, embed)
            rnn_input = torch.cat((inputs.squeeze(1), context), dim=1).unsqueeze(1)  # (B, 1, embed*2)
            output, h = self.rnn(rnn_input, h)  # (B, 1, hidden)
            logits = self.linear(output.squeeze(1))  # (B, vocab_size)

            # Apply temperature
            logits = logits / temperature

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Apply top-k filtering
            if top_k is not None:
                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                next_token = topk_indices[torch.arange(batch_size), torch.multinomial(topk_probs, 1).squeeze()]
            else:
                next_token = torch.multinomial(probs, 1).squeeze(1)

            predicted_sentence.append(next_token)

            inputs = self.embed(next_token).unsqueeze(1)

        return torch.stack(predicted_sentence, dim=1)  # (B, max_len)
