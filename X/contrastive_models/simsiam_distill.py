import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiam_distill(nn.Module):
    """
    Build a SimSiam model.
    """

    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam_distill, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        # self.encoder = base_encoder(num_classes=dim)
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)
        # self.load_encoder(self.encoder)
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),  # first layer
                nn.Linear(prev_dim, prev_dim, bias=False),
                nn.BatchNorm1d(prev_dim),
                nn.ReLU(inplace=True),  # second layer
                self.encoder.fc,
                nn.BatchNorm1d(dim, affine=False))  # output layer
        self.encoder.fc[6].bias.requires_grad = False  # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                nn.BatchNorm1d(pred_dim),
                nn.ReLU(inplace=True),  # hidden layer
                nn.Linear(pred_dim, dim))  # output layer


    def forward(self, x1, x2=None, is_feat=False, is_eval=False):
        # compute features for one view
        if is_feat:
            z1 = self.encoder(x1)  # NxC
            feats, repre, _ = self.encoder(x1, is_feat=is_feat) 
            p1 = self.predictor(z1)
            return feats, z1, p1
        if is_eval:
            output = self.encoder(x1)
            return output
        else:
            z1 = self.encoder(x1)  # NxC
            z2 = self.encoder(x2)  # NxC
            p1 = self.predictor(z1)  # NxC
            p2 = self.predictor(z2)  # NxC
            return p1, p2, z1.detach(), z2.detach()


    def load_encoder(self, encoder):
        checkpoint = torch.load("/root/ug_code/data_check/checkpoint/resnet-50_checkpoint_0099.pth.tar")
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('module.encoder') and not k.startswith('module.encoder.fc'):
                # remove prefix
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = self.encoder.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

