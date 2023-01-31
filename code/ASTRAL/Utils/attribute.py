from abc import ABCMeta, abstractmethod
import math
import pandas
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Variable(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name
        self.domain_size = 0

    @abstractmethod
    def normalize(self, x):
        logging.error('This line must not appear in console')
        pass


class CategoricalVariable(Variable):
    def __init__(self, name):
        super().__init__(name)

    def normalize(self, x):
        # dummy encoding or one-hot encoding
        x_ = pandas.get_dummies(x, prefix=self.name, drop_first=True)
        return x_


class BinaryVariable(CategoricalVariable):
    """
    BinaryVariable is a special case which designed
    for the sensitive attribute and the decision
    attributes. Other attributes with two domain
    values are all considered as Nominal_var.
    """

    def __init__(self, name, pos, neg):
        super().__init__(name)
        self.pos = pos
        self.neg = neg
        self.values = {self.pos, self.neg}
        self.domain_size = 2

    def normalize(self, x, mapping=None):
        values = set(x.values)
        if values.issubset(self.values):
            x_ = x.map({self.pos: 1, self.neg: -1})
            self.pos, self.neg = 1, -1
            return x_
        else:
            logging.error('Domain values are incompatible')

    def count(self, df):
        num_total = df.__len__()

        if isinstance(df, pandas.DataFrame):
            counts = df[self.name].value_counts()
        else:
            counts = df.value_counts()

        try:
            num_pos = counts.loc[self.pos]
        except KeyError:
            num_pos = 0

        try:
            num_neg = counts.loc[self.neg]
        except KeyError:
            num_neg = 0
        return num_total, num_pos, num_neg


class OrdinalVariable(Variable):
    def __init__(self, name):
        super().__init__(name)

    def normalize(self, x, mapping=None):
        values = set(x.values)
        # customized mapping, then scale to [-1, 1]
        if set(mapping.keys()).issubset(values):
            x_ = x.map(mapping)
            x_ = (x_ - x_.min()) / (x_.max() - x_.min()) * 2 - 1
            return x_
        else:
            logging.error('Mapping function is incompatible')


class QuantitativeVariable(Variable):
    def __init__(self, name):
        super().__init__(name)
        self.domain_size = math.inf

    def normalize(self, x, mapping=None):
        # scale to [-1, 1]
        x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
        return x_
