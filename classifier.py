from ast import literal_eval
import Classifier

class Clf:
    config = None
    classifier = None

    def __init__(self, config):
        self.config = self._Parameters(config)
        if self.config.classifier_method == 'svc':
            self.classifier = Classifier.SVC(config)
        elif self.config.classifier_method == 'sgd':
            self.classifier = Classifier.SGD(config)
        elif self.config.classifier_method == 'random forest':
            self.classifier = Classifier.RandomForest(config)
        elif self.config.classifier_method == 'logistic regression':
            self.classifier = Classifier.LogisticRegression(config)
        elif self.config.classifier_method == 'dl':
            self.classifier = Classifier.DL(config)

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def setArgs(self, **kwargs):

        if self.config.classifier_method == 'dl':
            self.classifier.fit(kwargs['1'], kwargs['2'], kwargs['3'], kwargs['4'])
        else:
            self.classifier.setArgs(**kwargs)

    class _Parameters:
        classifier_method = ''
        allowed = []
        kwargs = dict()

        def __init__(self, config):
            self.classifier_method = config.get('CLASSIFIER', 'method')
            self.allowed = literal_eval(config.get('CLASSIFIER', 'allowed'))
            if self.classifier_method not in self.allowed:
                raise ValueError('Classifier method not allowed by'
                    + ' config.CLASSIFIER.allowed.')
