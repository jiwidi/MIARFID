import torch
import torch.nn as nn
import torchvision.models as models


class BilinearModel(nn.Module):
    """Load model with pretrained weights and initialise new layers."""

    def __init__(self, num_classes: int = 20) -> None:
        """Load pretrained model, set new layers with specified number of layers."""
        super(BilinearModel, self).__init__()
        model: nn.Module = models.vgg16(pretrained=True)
        self.features: nn.Module = nn.Sequential(*list(model.features)[:-1])
        self.classifier: nn.Module = nn.Linear(512 ** 2, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight.data)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias.data, val=0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract input features, perform bilinear transform, project to # of classes and return."""
        outputs: torch.Tensor = self.features(
            inputs
        )  # extract features from pretrained base
        outputs = outputs.view(
            inputs.shape[0], 512, 15 ** 2
        )  # reshape to batchsize * 512 * 28 ** 2
        outputs = torch.bmm(outputs, outputs.permute(0, 2, 1))  # bilinear product
        outputs = torch.div(outputs, 15 ** 2)  # divide by 196 to normalize
        outputs = outputs.view(-1, 512 ** 2)  # reshape to batchsize * 512 * 512
        outputs = torch.sign(outputs) * torch.sqrt(
            outputs + 1e-5
        )  # signed square root normalization
        outputs = nn.functional.normalize(outputs, p=2, dim=1)  # l2 normalization
        outputs = self.classifier(outputs)  # linear projection
        return outputs


# class BilinearModel(nn.Module):
#     def __init__(self, num_classes=20, pretrained=True):
#         super(BilinearModel, self).__init__()
#         features = models.vgg16(pretrained=pretrained)
#         # Remove the pooling layer and full connection layer
#         self.conv = nn.Sequential(*list(features.children())[:-1])
#         self.fc = nn.Linear(512 * 512, num_classes)

#         if pretrained:
#             for parameter in self.conv.parameters():
#                 parameter.requires_grad = False
#             nn.init.kaiming_normal_(self.fc.weight.data)
#             nn.init.constant_(self.fc.bias, val=0)

#     def forward(self, input):
#         features = self.conv(input)
#         # Cross product operation

#         features = features.view(features.size(0), 512, -1)
#         features_T = torch.transpose(features, 1, 2)
#         features = torch.bmm(features, features_T) / 49
#         features = features.view(features.size(0), 512 * 512)
#         # The signed square root
#         features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
#         # L2 regularization
#         features = torch.nn.functional.normalize(features)

#         out = self.fc(features)
#         return out
