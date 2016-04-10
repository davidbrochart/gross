import math
import numpy as np
from numba import jit
from pandas import Series

@jit
def run_gr4j(x, p, e, q, s, uh1_tab, uh2_tab, l, m):
    for t in range(p.size):
        if p[t] > e[t]:
            pn = p[t] - e[t]
            en = 0.
            tmp = s[0] / x[0]
            ps = x[0] * (1. - tmp * tmp) * math.tanh(pn / x[0]) / (1. + tmp * math.tanh(pn / x[0]))
            s[0] += ps
        elif p[t] < e[t]:
            ps = 0.
            pn = 0.
            en = e[t] - p[t]
            tmp = s[0] / x[0]
            es = s[0] * (2. - tmp) * np.tanh(en / x[0]) / (1. + (1. - tmp) * np.tanh(en / x[0]))
            tmp = s[0] - es
            if tmp > 0.:
                s[0] = tmp
            else:
                s[0] = 0.
        else:
            pn = 0.
            en = 0.
            ps = 0.
        tmp = (4. * s[0] / (9. * x[0]))
        perc = s[0] * (1. - (1. + tmp * tmp * tmp * tmp) ** (-1. / 4.))
        s[0] -= perc
        pr_0 = perc + pn - ps
        q9 = 0.
        q1 = 0.
        for i in range(m):
            if i == 0:
                pr_i = pr_0
            else:
                pr_i = s[2 + i - 1]
            if i < l:
                q9 += uh1_tab[i] * pr_i;
            q1 += uh2_tab[i] * pr_i;
        q9 *= 0.9
        q1 *= 0.1
        f = x[1] * ((s[1] / x[2]) ** (7. / 2.))
        tmp = s[1] + q9 + f
        if tmp > 0.:
            s[1] = tmp
        else:
            s[1] = 0.
        tmp = s[1] / x[2]
        qr = s[1] * (1. - ((1. + tmp * tmp * tmp * tmp) ** (-1. / 4.)))
        s[1] -= qr
        tmp = q1 + f
        if tmp > 0.:
            qd = tmp
        else:
            qd = 0.
        q[t] = qr + qd
        for i in range(s.size - 2 - 2, -1, -1):
            s[2 + i + 1] = s[2 + i]
        if s.size > 2:
            s[2] = pr_0

class gr4j:
    """
    GR4J model class.
    """
    def sh1(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = (float(t) / self.x[3]) ** (5. / 2.)
        else:
            res = 1.
        return res
    def sh2(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = 0.5 * ((float(t) / self.x[3]) ** (5. / 2.))
        elif t < 2. * self.x[3]:
            res = 1. - 0.5 * ((2. - float(t) / self.x[3]) ** (5. / 2.))
        else:
            res = 1.
        return res
    def uh1(self, j):
        return self.sh1(j) - self.sh1(j - 1)
    def uh2(self, j):
        return self.sh2(j) - self.sh2(j - 1)
    def __init__(self, x, s = None):
        self.x = x
        if s == None:
            self.s = np.array([0., 0.] + [0.] * int(2. * self.x[3]))
        else:
            self.s = np.array(s)
        self.l = int(self.x[3]) + 1
        self.m = int(2. * self.x[3]) + 1
        self.uh1_tab = np.empty(self.l)
        self.uh2_tab = np.empty(self.m)
        for i in range(self.m):
            if i < self.l:
                self.uh1_tab[i] = self.uh1(i + 1)
            self.uh2_tab[i] = self.uh2(i + 1)
    def run(self, pe):
        q = Series(np.empty_like(pe.p), index = pe.index)
        run_gr4j(self.x, pe.p.values, pe.e.values, q.values, self.s, self.uh1_tab, self.uh2_tab, self.l, self.m)
        return q
    def run_raw(self, p, e):
        q = np.empty_like(p)
        run_gr4j(self.x, p, e, q, self.s, self.uh1_tab, self.uh2_tab, self.l, self.m)
        return q
    def set_conf(self, conf):
        if conf == 0:
            for i in range(self.s.size):
                self.s[i] = 0.
        elif conf == 1:
            self.s[0] = self.x[0] / 2.
            self.s[1] = self.x[2] / 2.
            for i in range(self.s.size - 2):
                self.s[i + 2] = 0.



@jit
def run_gr4h(x, p, e, q, s, uh1_tab, uh2_tab, l, m):
    for t in range(p.size):
        if p[t] > e[t]:
            pn = p[t] - e[t]
            en = 0.
            tmp = s[0] / x[0]
            ps = x[0] * (1. - tmp * tmp) * math.tanh(pn / x[0]) / (1. + tmp * math.tanh(pn / x[0]))
            s[0] += ps
        elif p[t] < e[t]:
            ps = 0.
            pn = 0.
            en = e[t] - p[t]
            tmp = s[0] / x[0]
            es = s[0] * (2. - tmp) * np.tanh(en / x[0]) / (1. + (1. - tmp) * np.tanh(en / x[0]))
            tmp = s[0] - es
            if tmp > 0.:
                s[0] = tmp
            else:
                s[0] = 0.
        else:
            pn = 0.
            en = 0.
            ps = 0.
        tmp = s[0] / (5.25 * x[0])
        perc = s[0] * (1. - (1. + tmp * tmp * tmp * tmp) ** (-1. / 4.))
        s[0] -= perc
        pr_0 = perc + pn - ps
        q9 = 0.
        q1 = 0.
        for i in range(m):
            if i == 0:
                pr_i = pr_0
            else:
                pr_i = s[2 + i - 1]
            if i < l:
                q9 += uh1_tab[i] * pr_i;
            q1 += uh2_tab[i] * pr_i;
        q9 *= 0.9
        q1 *= 0.1
        f = x[1] * ((s[1] / x[2]) ** (7. / 2.))
        tmp = s[1] + q9 + f
        if tmp > 0.:
            s[1] = tmp
        else:
            s[1] = 0.
        tmp = s[1] / x[2]
        qr = s[1] * (1. - ((1. + tmp * tmp * tmp * tmp) ** (-1. / 4.)))
        s[1] -= qr
        tmp = q1 + f
        if tmp > 0.:
            qd = tmp
        else:
            qd = 0.
        q[t] = qr + qd
        for i in range(s.size - 2 - 2, -1, -1):
            s[2 + i + 1] = s[2 + i]
        if s.size > 2:
            s[2] = pr_0

class gr4h:
    """
    GR4H model class.
    """
    def sh1(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = (float(t) / self.x[3]) ** 1.25
        else:
            res = 1.
        return res
    def sh2(self, t):
        if t == 0:
            res = 0.
        elif t < self.x[3]:
            res = 0.5 * ((float(t) / self.x[3]) ** 1.25)
        elif t < 2. * self.x[3]:
            res = 1. - 0.5 * ((2. - float(t) / self.x[3]) ** 1.25)
        else:
            res = 1.
        return res
    def uh1(self, j):
        return self.sh1(j) - self.sh1(j - 1)
    def uh2(self, j):
        return self.sh2(j) - self.sh2(j - 1)
    def __init__(self, x, s = None):
        self.x = x
        if s == None:
            self.s = np.array([0., 0.] + [0.] * int(2. * self.x[3]))
        else:
            self.s = np.array(s)
        self.l = int(self.x[3]) + 1
        self.m = int(2. * self.x[3]) + 1
        self.uh1_tab = np.empty(self.l)
        self.uh2_tab = np.empty(self.m)
        for i in range(self.m):
            if i < self.l:
                self.uh1_tab[i] = self.uh1(i + 1)
            self.uh2_tab[i] = self.uh2(i + 1)
    def run(self, pe):
        q = Series(np.empty_like(pe.p), index = pe.index)
        run_gr4h(self.x, pe.p.values, pe.e.values, q.values, self.s, self.uh1_tab, self.uh2_tab, self.l, self.m)
        return q
    def run_raw(self, p, e):
        q = np.empty_like(p)
        run_gr4h(self.x, p, e, q, self.s, self.uh1_tab, self.uh2_tab, self.l, self.m)
        return q
    def set_conf(self, conf):
        if conf == 0:
            for i in range(self.s.size):
                self.s[i] = 0.
        elif conf == 1:
            self.s[0] = self.x[0] / 2.
            self.s[1] = self.x[2] / 2.
            for i in range(self.s.size - 2):
                self.s[i + 2] = 0.



@jit
def run_delay(dc, n, q_in, q_out, s):
    for t in range(q_in.size):
        for i in range(n - 1, 0, -1):
            s[i] = s[i - 1];
        s[0] = (1. - dc) * q_in[t];
        if n > 1:
            s[1] += dc * q_in[t];
        q_out[t] = s[n - 1];

class delay:
    """
    Delay model class.
    """
    def __init__(self, d, s = None):
        self.dc = d - float(int(d));
        self.n = int(math.ceil(d)) + 1;
        if s == None:
            self.s = np.array([0.] * self.n)
        else:
            self.s = np.array(s)
    def run(self, q_in):
        try:
            index = q_in.index
            q_in2 = qin.values
        except:
            index = np.arange(len(q_in))
            q_in2 = q_in
        q_out = Series(np.empty_like(q_in2), index = index)
        run_delay(self.dc, self.n, q_in2, q_out.values, self.s)
        return q_out
    def run_raw(self, q_in):
        q_out = np.empty_like(q_in)
        run_delay(self.dc, self.n, q_in, q_out, self.s)
        return q_out
