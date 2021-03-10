import torch
import torch.nn as nn
import torchvision.models as models


class BilinearModel(nn.Module):
    def __init__(self, num_classes=20):
        super(BilinearModel, self).__init__()
        self.features = nn.Sequential(
            *list(models.vgg16(pretrained=True).features)[:-1]
        )
        self.classifier = nn.Linear(512 ** 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

        self.d1 = nn.Dropout(p=0.5)
        self.d2 = nn.Dropout(p=0.5)

    def forward(self, x):
        outputs: torch.Tensor = self.features(
            x
        )  # extract features from pretrained base
        outputs = outputs.view(
            x.shape[0], 512, 15 ** 2
        )  # reshape to batchsize * 512 * 15 ** 2

        # Dropout 0.5
        out1 = self.d1(outputs)
        out2 = self.d2(outputs.permute(0, 2, 1))

        outputs = torch.bmm(out1, out2)  # bilinear product
        outputs = torch.div(outputs, 15 ** 2)  # divide by 225 to normalize
        outputs = outputs.view(-1, 512 ** 2)  # reshape to batchsize * 512 * 512
        outputs = torch.sign(outputs) * torch.sqrt(
            outputs + 1e-5
        )  # signed square root normalization
        outputs = nn.functional.normalize(outputs, p=2, dim=1)  # l2 normalization
        outputs = self.classifier(outputs)  # linear projection
        return outputs

