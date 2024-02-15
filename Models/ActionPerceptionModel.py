from torch import nn
import torch


class ActionPerceptionModel(nn.Module):

    def __init__(self, out_n=(32, 32, 64), input_dim=15, inference=False, sequence_len=10, num_layers=3):
        super(ActionPerceptionModel, self).__init__()

        self.sequence_len = sequence_len
        self.feed_input = nn.Linear(input_dim, out_n[0])
        self.batch_norm = nn.BatchNorm1d(out_n[0])

        self.feed_input1 = nn.Linear(out_n[1], out_n[1])
        self.batch_norm1 = nn.BatchNorm1d(out_n[1])

        self.rnn_0 = nn.LSTM(input_size=out_n[1], hidden_size=out_n[2], num_layers=num_layers, batch_first=True, dropout=.5)

        self.feed_output1 = nn.Linear(out_n[2], out_n[2])
        self.batch_norm2 = nn.BatchNorm1d(out_n[2])


        self.drop_out = nn.Dropout(.5)
        self.feed_outcome = nn.Linear(out_n[2], 1)


        self.relu = nn.Mish()
        self.inference = inference

        self.hidden_dim = out_n[2]
        self.num_layers = num_layers




    def forward(self, x):

        z = self.relu(self.feed_input(x))
        z = self.batch_norm(z.swapaxes(2, 1))
        z = self.drop_out(z.swapaxes(1, 2))

        z = self.relu(self.feed_input1(z))
        z = self.batch_norm1(z.swapaxes(2, 1))
        z = self.drop_out(z.swapaxes(1, 2))

        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        z, _ = self.rnn_0(z, (h0, c0))

        z = self.relu(self.feed_output1(z))
        z = self.batch_norm2(z.swapaxes(2, 1))
        z = self.drop_out(z.swapaxes(1, 2))


        outcomes = self.feed_outcome(z[:, -1, :])

        if self.inference:
            # outcomes = torch.unsqueeze(torch.flatten(torch.sigmoid(outcomes)), dim=1)
            outcomes = torch.sigmoid(outcomes)


        return outcomes



class SiamseActionPerceptionModel(nn.Module):
    def __init__(self, out_n=(32, 32, 128), input_dim=15, inference=False, sequence_len=10):
        super(SiamseActionPerceptionModel, self).__init__()

        self.sequence_len = sequence_len
        self.feed_input = nn.Linear(input_dim, out_n[0])
        self.batch_norm = nn.BatchNorm1d(out_n[0])

        self.feed_input1 = nn.Linear(out_n[1], out_n[1])
        self.batch_norm1 = nn.BatchNorm1d(out_n[1])

        self.rnn_0 = nn.LSTM(input_size=out_n[1], hidden_size=out_n[2], num_layers=3, batch_first=True, dropout=.5)

        self.feed_output1 = nn.Linear(out_n[2], out_n[2])
        self.batch_norm2 = nn.BatchNorm1d(out_n[2])


        self.drop_out = nn.Dropout(.5)



        self.relu = nn.Mish()
        self.tanh = nn.Tanh()
        self.inference = inference

        self.embeddings = None
        self.labels = None




    def forward(self, x):

        z = self.relu(self.feed_input(x))
        z = self.batch_norm(z.swapaxes(2, 1))
        z = self.drop_out(z.swapaxes(1, 2))

        z = self.relu(self.feed_input1(z))
        z = self.batch_norm1(z.swapaxes(2, 1))
        z = self.drop_out(z.swapaxes(1, 2))

        z, _ = self.rnn_0(z)

        embd = self.feed_output1(z)
        embd = torch.nn.functional.normalize(embd, p=2, dim=-1)


        return embd


    def predict(self, query, n=5, last=True):

        query = torch.unsqueeze(self.forward(query), dim=2)
        dist = (query - self.embeddings).pow(2).sum(-1).sqrt()
        pred_labels = torch.argmin(dist, dim=-1, keepdim=False)



        return pred_labels

    def setEmbedding(self, embeddings, labels):
        pos_embeddings = torch.mean(embeddings[labels == 1], dim=0)
        neg_embeddings = torch.mean(embeddings[labels == 0], dim=0)
        self.embeddings = torch.vstack([neg_embeddings, pos_embeddings])
        self.labels = labels





if __name__ == '__main__':
    import numpy as np
    from pytorch_metric_learning import losses
    inp = torch.randn(2, 5, 25)
    query = torch.randn(2, 32)
    labels = torch.Tensor(np.random.choice([0, 1], size=(10,), p=[1./3, 2./3]))
    loss_func = losses.ContrastiveLoss()

    model = SiamseActionPerceptionModel(input_dim=25)
    embd = model(inp)
    model.setEmbedding(embd.view((-1, 32)), labels)
    print(model.predict(inp))
