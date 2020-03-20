import argparse
import cv2
import json
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, pre_features, features, target_layers):
        self.model = model
        self.pre_features = pre_features
        self.features = features
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        for pref in self.pre_features:
            x = getattr(self.model, pref)(x)

        submodel = getattr(self.model, self.features)
        # go through the feature extractor's forward pass
        for name, module in submodel._modules.items():
            # print(name, module)
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model,
                 pre_feature_block=[],
                 feature_block='features',
                 target_layers='35',
                 classifier_block=['classifier']):
        self.model = model
        self.classifier_block = classifier_block
        # assume the model has a module named `feature`      ⬇⬇⬇⬇⬇⬇⬇⬇
        self.feature_extractor = FeatureExtractor(self.model,
                                                  pre_feature_block,
                                                  feature_block,
                                                  target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        # ⬇ target layer    ⬇ final layer's output
        target_activations, output = self.feature_extractor(x)
        print('target_activations[0].size: {}'.format(target_activations[0].size()))# for vgg'35 ([1, 512, 14, 14])
        print('output.size: {}'.format(output.size()))                              # for vgg'36 ([1, 512, 7, 7])

        for i, classifier in enumerate(self.classifier_block):
            if i == len(self.classifier_block) - 1:
                output = output.view(output.size(0), -1)
                print('output.view.size: {}'.format(output.size()))                 # for vgg'36 ([1, 25088])

            output = getattr(self.model, classifier)(output)
            print('output.size: {}'.format(output.size()))                          # for vgg'36 ([1, 1000])

        return target_activations, output


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite("cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, pre_feature_block, feature_block, target_layer_names, classifier_block, use_cuda):
        self.model = model
        self.model.eval()
        self.pre_feature_block = pre_feature_block
        self.feature_block = feature_block
        self.classifier_block = classifier_block
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model,
                                      pre_feature_block,
                                      feature_block,
                                      target_layer_names,
                                      classifier_block)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        with open('imagenet_class_index.json') as f:
            labels = json.load(f)
        
        print('prediction[{}]: {}'.format(index, labels[str(index)][1]))
        print('output.size: {}'.format(output.size()))                            # for vgg'36 ([1, 1000])
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        #print('output: {}'.format(output))
        #print('one_hot: {}'.format(one_hot))                                      # 
        getattr(self.model, self.feature_block).zero_grad()
        for classifier in self.classifier_block:
            getattr(self.model, classifier).zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = self.extractor.get_gradients()
        #print('len(gradients): {}'.format(len(gradients)))
        print('gradients[0].size(): {}'.format(gradients[0].size()))
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        print('target.size(): {}'.format(target.size()))
        target = target.cpu().data.numpy()[0, :]
        print('target.shape: {}'.format(target.shape))

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        print('weights.shape: {}'.format(weights.shape))
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        print('cam.shape: {}'.format(cam.shape))            # (14, 14)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        #print('cam: {}'.format(cam))
        print('cam.shape: {}'.format(cam.shape))
        cam = np.maximum(cam, 0)                            # remove negative numbers
        cam = cv2.resize(cam, (224, 224))
        print('cam.shape: {}'.format(cam.shape))
        #print('cam: {}'.format(cam))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    image_path = './examples/wolf.png'
    use_cuda = False

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    config = {
        'vgg19':    {
            'pre_feature': [],
            'features': 'features',
            'target': ['35'],
            'classifier': ['classifier']
        }, 
        'resnet50': {
            'pre_feature': ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3'],
            'features': 'layer4',
            'target': ['2'],
            'classifier': ['avgpool', 'fc']
        }
    }

    model_name = 'resnet50'
    config = config[model_name]
    model = getattr(models, model_name)(pretrained=True)
    grad_cam = GradCam(model,
                       pre_feature_block=config['pre_feature'],
                       feature_block=config['features'], # features
                       target_layer_names=config['target'],
                       classifier_block=config['classifier'],  # classifier
                       use_cuda=use_cuda)

    img = cv2.imread(image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255

    #print('img.size(): {}'.format(img.size))
    input = preprocess_image(img)
    #print('input.size(): {}'.format(input.size()))

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)
    
    show_cam_on_image(img, mask)
    '''
    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=use_cuda)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)
    '''