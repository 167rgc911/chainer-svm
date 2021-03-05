from hinge import hinge

import chainer
import chainer.links as L
import chainer.functions as F
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

        if self.penalty == 'L1':
            loss += self.c * F.sum(F.absolute(self.fc.W))

        elif self.penalty == 'L2':
            n = F.matmul(self.fc.W, self.fc.W.T)
            loss += self.c * F.reshape(n, ())

        return loss
