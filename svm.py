from hinge import hinge

import chainer
import chainer.links as L
from chainer import cuda, Variable, Chain


class SVM(Chain):

    def __init__(self, c, penalty='L1'):
        super(SVM, self).__init__(
            fc=L.Linear(2, 2),
        )
        self.c = c
        self.penalty = penalty

    def forward(self, x, t, train=True):
        chainer.config.train = not train
        xp = cuda.get_array_module(*x)

        x = Variable(x)
        t = Variable(t)
        h = self.fc(x)
        loss = hinge(h, t, self.penalty)

        if self.penalty == 'l1':
            loss += self.c * L.sum(Variable(abs(self.fc.W)))

        elif self.penalty == 'l2':
            n = Variable(self.fc.W.dot(self.fc.W.T))
            loss += self.c * L.reshape(n, ())

        return loss
