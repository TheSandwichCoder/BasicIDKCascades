
import torch
import torchvision.models
from cascade import *
from ImageNetReader import *

from globals import device, DESIRED_PRECISION, M1_P95_CONF_THRESHOLD, M2_P95_CONF_THRESHOLD, M3_P95_CONF_THRESHOLD
import torchvision.transforms as transforms

ImageNetV2Transforms = preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])
dataset = ImageNetV2Dataset(location="data", transform=ImageNetV2Transforms)
classifier_loader = DataLoader(dataset, batch_size=16, num_workers=4)
cascade_loader = DataLoader(dataset, batch_size=1, num_workers=4)

def get_confidence_thresholds():

    m1 = get_resnet_18()
    cc1 = CascadeClassifier(m1)

    print(get_model_conf_threshold(cc1, DESIRED_PRECISION, classifier_loader))


    m2 = get_resnet_34()
    cc2 = CascadeClassifier(m2)

    print(get_model_conf_threshold(cc2, DESIRED_PRECISION, classifier_loader))

    m3 = get_resnet_152()
    cc3 = CascadeClassifier(m3)

    print(get_model_conf_threshold(cc3, DESIRED_PRECISION, classifier_loader))

def benchmark():
    m1 = get_resnet_18()
    cc1 = CascadeClassifier(m1, M1_P95_CONF_THRESHOLD)

    m2 = get_resnet_34()
    cc2 = CascadeClassifier(m2, M2_P95_CONF_THRESHOLD)

    # for i in range(5,10):
    #     conf_threshold = i * 0.1
    #     print(conf_threshold)

    #     m3 = get_resnet_152()
    #     cc3 = CascadeClassifier(m3, conf_threshold)
    #     bench_classifier(cc3, classifier_loader)

    cc3 = CascadeClassifier(get_resnet_152(), M3_P95_CONF_THRESHOLD, deterministic=True)

    cascade = IDKCascade([cc1, cc2, cc3])

    bench_classifier(cc1, cascade_loader)
    bench_classifier(cc2, cascade_loader)
    bench_classifier(cascade, cascade_loader)
    bench_classifier(cc3, cascade_loader)


# m3 = get_resnet_152()
# cc3 = CascadeClassifier(m3, M3_P95_CONF_THRESHOLD)
# bench_classifier(cc3, loader)

# get_confidence_thresholds()
benchmark()