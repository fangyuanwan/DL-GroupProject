import numpy as np
import torch
from sklearn.metrics import accuracy_score
from small_text import TransformersDataset, TransformerModelArguments, TransformerBasedClassificationFactory, \
    GreedyCoreset
from small_text.data.datasets import SklearnDataset
from small_text.active_learner import PoolBasedActiveLearner
from small_text.initialization import random_initialization_balanced
from small_text.query_strategies import PredictionEntropy
from small_text.classifiers import ConfidenceEnhancedLinearSVC
from small_text.classifiers.factories import SklearnClassifierFactory
from transformers import AutoTokenizer


class ActiveLearner:
    def __init__(self, train_set, step_count):
        self.train_set_raw = train_set
        self.train_set_sentence = None
        tmp = []
        for item in train_set:
            tmp.append(' '.join(item[0:-1].astype(str)))
        self.train_set_sentence = np.array(tmp)
        self.train_set = TransformersDataset.from_arrays(self.train_set_sentence, train_set[:, -1].astype(np.int64),
                                                         AutoTokenizer.from_pretrained(
                                                             'bert-base-uncased'
                                                         ))
        self.active_learner = None
        self.current_step_labeled = None
        self.indices_labeled = np.array([], dtype=np.int64)
        self.init()
        self.selected = np.array([], dtype=np.int64)
        self.step_count = step_count

    def init(self):
        self.make_active_learner()
        # self.initialize_active_learner()

    def initialize_active_learner(self):
        indices_initial = random_initialization_balanced(self.train_set.y, n_samples=self.step_count)
        self.active_learner.initialize_data(indices_initial, self.train_set.y[indices_initial])
        self.indices_labeled = np.append(self.indices_labeled, indices_initial)
        self.current_step_labeled = self.indices_labeled
        return indices_initial

    def generate_train_data(self):
        to_return = np.array([], dtype=np.int64)
        to_return = np.array(self.train_set_raw)[self.current_step_labeled]
        # print(self.current_step_labeled)
        # print(type(self.current_step_labeled))
        # new_return = np.array(self.train_set.x)[self.current_step_labeled]
        # new_y = self.train_set.y[self.current_step_labeled].reshape((self.step_count, -1))
        # print(new_return)
        # print(new_y)
        # for i in range(0, self.step_count):
        #     to_return = np.append(to_return, new_return[i])
        #     to_return = np.append(to_return, new_y[i])
        # #     print(to_return.shape)
        # to_return = to_return.reshape((self.step_count, -1))
        return to_return

    def query(self):
        if self.current_step_labeled is None:
            self.initialize_active_learner()
        elif len(self.selected) >= len(self.train_set):
            return self.selected
        else:
            self.current_step_labeled = self.active_learner.query(num_samples=self.step_count)
            self.update()
        data = self.generate_train_data()
        self.selected = np.append(self.selected, data).reshape((-1, 103))
        return self.selected

    def update(self):
        self.active_learner.update(self.train_set.y[self.current_step_labeled])

    def make_active_learner(self):
        model = ConfidenceEnhancedLinearSVC()
        num_classes = 7
        transformer_model = TransformerModelArguments('bert-base-uncased')
        clf_factory = TransformerBasedClassificationFactory(transformer_model,
                                                            num_classes,
                                                            kwargs=dict({'device': 'cuda',
                                                                         'mini_batch_size': 32,
                                                                         'class_weight': 'balanced'
                                                                         }))
        query_strategy = GreedyCoreset()  # changeable
        active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, self.train_set)
        self.active_learner = active_learner

        return active_learner


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    test_acc = accuracy_score(y_pred_test, test.y)

    print('Train accuracy: {:.2f}'.format(accuracy_score(y_pred, train.y)))
    print('Test accuracy: {:.2f}'.format(test_acc))

    return test_acc


if __name__ == "__main__":

    torch.cuda.empty_cache()
    import data_loader_recsys_transfer_finetune_ as data_loader_recsys

    print(torch.cuda.is_available())

    dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': 'Data/Session/LFDshort1w.csv'})

    all_samples = dl.example
    print(("len(all_samples)", len(all_samples)))
    items = dl.item_dict
    items_len = len(items)
    print(("len(items)", len(items)))
    targets = dl.target_dict
    targets_len = len(targets)
    print(("len(targets)", len(targets)))

    negtive_samples = 5
    top_k = 1

    if 0 in items:
        padtoken = items[0]  # is the padding token in the beggining of the sentence
    else:
        padtoken = len(items) + 1

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    dev_sample_index = 1 * int(0.9 * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    act = ActiveLearner(train_set, 200)

    query = act.query()
    indices_labeled = act.indices_labeled
    valid_set_skLearn = SklearnDataset(valid_set[:, 0:-1], valid_set[:, -1], [1, 2, 3, 4, 5, 6])
    results = []
    # results.append(evaluate(act.active_learner, act.train_set[indices_labeled], valid_set_skLearn))

    for i in range(20):
        print(i)
        # ...where each iteration consists of labelling 20 samples
        indices_queried = act.query()
        print(indices_queried[:, -1])
        print(act.train_set.y[act.current_step_labeled])

        # Simulate user interaction here. Replace this for real-world usage.
        # y = act.train_set.y[indices_queried]

        # Return the labels for the current query to the active learner.
        # act.active_learner.update(y)
        act.update(act.current_step_labeled)

        print(indices_queried)

        # indices_labeled = np.concatenate([indices_queried, indices_labeled])

        # print('---------------')
        # print(f'Iteration #{i} ({len(indices_labeled)} samples)')
        # results.append(evaluate(act.active_learner, act.train_set[indices_labeled], valid_set_skLearn))

    # while True:
    #     query = act.query()
    #     print(query)
    #     print(query.shape)
