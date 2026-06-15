import joblib
import torch
import torchvision
from globals import device
from tqdm import tqdm
import numpy as np
from train_mlp import MLP
import time

class CascadeClassifier:
    def __init__(self, model, confidence_threshold=0.8, probability = 0, exec_time = 0, deterministic = False):
        self.model = model
        self.confidence_threshold = confidence_threshold

        self.deterministic = deterministic
        # TODO
        self.probability = 0.0
        self.exec_time = 0.0
        self.feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    def run(self, x):
        with torch.no_grad():
            return self.model(x)

    def predict_for_mlp(self, x):
        with torch.no_grad():
            o = self.feature_extractor(x)
            features = torch.flatten(o, 1)

            logits = self.model.fc(features)

            probabilities = torch.nn.functional.softmax(logits, dim=1)

            top_prob, top_cat_id = torch.topk(probabilities, 1, dim=1)

            return features, top_prob, torch.Tensor([top_cat_id[0]])

    def predict_for_rf(self, x):
        with torch.no_grad():
            output = self.model(x)

            probabilities = torch.nn.functional.softmax(output, dim=1)

            prob_np = probabilities.numpy()

            top_prob, top_cat_id = torch.topk(probabilities, 2, dim=1)
            top_prob, top_cat_id = top_prob.squeeze(), top_cat_id.squeeze()

            eps = 1e-12    

            conf = top_prob[0]
            entropy = -np.sum(prob_np * np.log2(prob_np + eps))
            margin = top_prob[0] - top_prob[1]

            return torch.Tensor([top_cat_id[0]]), conf.item(), entropy.item(), margin.item()
    
    def predict_raw(self, x):
        with torch.no_grad():
            output = self.model(x)

            probabilities = torch.nn.functional.softmax(output, dim=1)

            top_prob, top_cat_id = torch.topk(probabilities, 1, dim=1)
            # print(probabilities)
            # print(top_cat_id)

            return top_prob, top_cat_id

    def predict(self, x):
        with torch.no_grad():
            output = self.model(x)

            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, top_cat_id = torch.topk(probabilities, 1, dim=1)

            if self.deterministic:
                return top_cat_id

            # -1 is chosen as the id for IDK
            idk_key = top_prob < self.confidence_threshold
            top_cat_id[idk_key] = -1

            return top_cat_id
        
def get_model_conf_threshold(cc, precision, validation_loader, eps = 0.0001):
    all_predictions = torch.Tensor([])
    all_confidences = torch.Tensor([])
    all_labels = torch.Tensor([])

    with torch.no_grad():
        for batch_features, batch_labels in tqdm(validation_loader):
            
            probs, ids = cc.predict_raw(batch_features.to(device))
            # print(all_predictions.shape, ids.cpu().shape, all_confidences.shape, probs.cpu().shape, all_labels.shape, batch_labels.cpu().shape)
            
            all_predictions = torch.cat((all_predictions, ids.cpu()), dim=0)
            all_confidences = torch.cat((all_confidences, probs.cpu()), dim=0)
            all_labels = torch.cat((all_labels, batch_labels.cpu()), dim=0)

    all_predictions = np.array(all_predictions).flatten()
    all_confidences = np.array(all_confidences).flatten()
    all_labels = np.array(all_labels)

    low = 0
    high = 1
    mid = (low + high) / 2
    last_successful = mid

    # binary search for confidence threshold for precision
    while low < high:
        mid = (low + high) / 2
        # pred correct
        
        num_correct = np.sum(((all_confidences >= mid) & (all_predictions == all_labels)))

        # pred made
        num_total = np.sum(all_confidences >= mid)

        p = num_correct / num_total
        print(p, mid)

        if p >= precision:
            high = mid - eps
            last_successful = mid
        else:
            low = mid + eps


    return last_successful


def get_resnet_18():
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights).to(device)
    model.eval()

    return model
    
def get_resnet_34():
    weights = torchvision.models.ResNet34_Weights.DEFAULT
    model = torchvision.models.resnet34(weights=weights).to(device)
    model.eval()

    return model
    
def get_resnet_152():
    weights = torchvision.models.ResNet152_Weights.DEFAULT
    model = torchvision.models.resnet152(weights=weights).to(device)
    model.eval()

    return model
    

# no optimization. Run classifiers in order
class IDKCascade:
    def __init__(self, classifiers, dependent = True):
        self.n_classifiers = len(classifiers)
        self.classifiers = classifiers
        self.is_dependent = dependent

        self.rf_skipper = joblib.load("saved_models/random_forest.pkl")['model']


    # last classifier assumsed as deterministic
    def predict(self, x):
        cc_i = 0

        while cc_i < self.n_classifiers:
            cc = self.classifiers[cc_i]
            pred, conf, entropy, margin = cc.predict_for_rf(x)

            if not cc.deterministic and conf < cc.confidence_threshold:
                pred = torch.Tensor([-1])

            if not torch.equal(pred.squeeze(), torch.tensor([-1]).squeeze().to(device)) or cc_i == self.n_classifiers - 1:
                # print("returned")
                return pred

            if cc_i == 0:
                if self.skip_type == "threshold":
                    if conf < 0.3:
                        cc_i += 1

                elif self.skip_type == "rf":
                    data = np.expand_dims(np.array([conf, entropy, margin]), axis = 0)

                    can_skip = self.rf_skipper.predict(data)

                    if can_skip[0]:
                        cc_i += 1
            
            cc_i += 1
            

class IDKCascadeThresholdSkip(IDKCascade):
    def __init__(self, classifiers, dependent = True):
        super().__init__(classifiers, dependent)

    def predict(self, x):
        cc_i = 0

        while cc_i < self.n_classifiers:
            cc = self.classifiers[cc_i]

            conf, pred = cc.predict_raw(x)

            if not cc.deterministic and conf < cc.confidence_threshold:
                pred = torch.Tensor([-1])

            if pred.item() != -1 or cc_i == self.n_classifiers - 1:
                return pred

            if cc_i == 0 and conf < 0.3:
                cc_i += 1
            
            cc_i += 1

class IDKCascadeRFSkip(IDKCascade):
    def __init__(self, classifiers, dependent = True):
        super().__init__(classifiers, dependent)
        self.rf_skipper = joblib.load("saved_models/random_forest.pkl")['model']

    def predict(self, x):
        cc_i = 0

        while cc_i < self.n_classifiers:
            cc = self.classifiers[cc_i]

            pred, conf, entropy, margin = cc.predict_for_rf(x)

            if not cc.deterministic and conf < cc.confidence_threshold:
                pred = torch.Tensor([-1])

            if pred.item() != -1 or cc_i == self.n_classifiers - 1:
                return pred

            if cc_i == 0:
                data = np.expand_dims(np.array([conf, entropy, margin]), axis = 0)

                can_skip = self.rf_skipper.predict(data)

                if can_skip[0]:
                    cc_i += 1
            
            cc_i += 1

class IDKCascadeMLPSkip(IDKCascade):
    def __init__(self, classifiers, dependent = True):
        super().__init__(classifiers, dependent)
        state_dict = torch.load("saved_models/mlp.pth", weights_only=True)
        self.mlp_skipper = MLP()

        self.mlp_skipper.load_state_dict(state_dict)

    def predict(self, x):
        cc_i = 0

        while cc_i < self.n_classifiers:
            cc = self.classifiers[cc_i]

            features, conf, pred = cc.predict_for_mlp(x)

            if not cc.deterministic and conf < cc.confidence_threshold:
                pred = torch.Tensor([-1])

            if pred.item() != -1 or cc_i == self.n_classifiers - 1:
                return pred

            if cc_i == 0:
                o = self.mlp_skipper(features)

                if o[0, 1] > o[0, 0]:
                    cc_i += 1
            
            cc_i += 1

def get_data(cc, data_loader):
    arr = []

    with torch.no_grad():
        for batch_features, batch_label in tqdm(data_loader):
            o = cc.run(batch_features.to(device))
            arr.append(o.numpy())

    return np.array(arr)



def get_data2(m, data_loader):
    arr = []

    feature_extractor = torch.nn.Sequential(*list(m.children())[:-1])

    with torch.no_grad():
        for batch_features, batch_label in tqdm(data_loader):
            o = feature_extractor(batch_features)
            o = o.flatten()
            arr.append(o.numpy())

    return np.array(arr)
            
def bench_classifier(cc, data_loader):
    correct = 0
    total_guessed = 0
    total = 0
    time_taken = 0

    with torch.no_grad():
        for batch_features, batch_label in tqdm(data_loader):
            st = time.perf_counter()
            pred = cc.predict(batch_features.to(device))
            et = time.perf_counter()

            correct += sum(pred.flatten() == batch_label.flatten())

            total_guessed += sum(pred.flatten() != -1)
            
            # print(pred.flatten(), batch_label.flatten())
            total += len(pred)
            time_taken += et - st

    acc = correct / total
    prec = correct / total_guessed
    prob = total_guessed / total
    avg_time = time_taken / total 

    print(f"acc:{acc} prec:{prec} prob:{prob} avg time:{avg_time*1000:.1f}ms")

    return acc, prec, prob, avg_time


