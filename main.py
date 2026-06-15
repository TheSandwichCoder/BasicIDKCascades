
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
classifier_loader = DataLoader(dataset, batch_size=16, num_workers=0)
cascade_loader = DataLoader(dataset, batch_size=1, num_workers=0)

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

    cc3 = CascadeClassifier(get_resnet_152(), M3_P95_CONF_THRESHOLD, deterministic=True)

    cascade_norm = IDKCascade([cc1, cc2, cc3])
    cascade_threshold = IDKCascadeThresholdSkip([cc1, cc2, cc3])
    cascade_rf = IDKCascadeRFSkip([cc1, cc2, cc3])
    cascade_mlp = IDKCascadeMLPSkip([cc1, cc2, cc3])

    bench_classifier(cascade_rf, cascade_loader)
    bench_classifier(cascade_mlp, cascade_loader)
    bench_classifier(cascade_threshold, cascade_loader)
    bench_classifier(cascade_norm, cascade_loader)


def get_datas():
    m1 = get_resnet_18()
    cc1 = CascadeClassifier(m1, M1_P95_CONF_THRESHOLD)

    m2 = get_resnet_34()
    cc2 = CascadeClassifier(m2, M2_P95_CONF_THRESHOLD)

    m3 = get_resnet_152()
    cc3 = CascadeClassifier(m3, M3_P95_CONF_THRESHOLD, deterministic=True)

    data1 = get_data(cc1, cascade_loader)
    data2 = get_data(cc2, cascade_loader)
    data3 = get_data(cc3, cascade_loader)

    np.save('m1_output.npy', data1)
    np.save('m2_output.npy', data2)
    np.save('m3_output.npy', data3)

def get_datas2():
    m1 = get_resnet_18()

    data1 = get_data2(m1, cascade_loader)

    np.save('m1_features.npy', data1)


# m3 = get_resnet_152()
# cc3 = CascadeClassifier(m3, M3_P95_CONF_THRESHOLD)
# bench_classifier(cc3, loader)

# get_confidence_thresholds()
benchmark()

# get_data()