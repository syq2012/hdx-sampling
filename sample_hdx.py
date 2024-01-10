import numpy as np
from itertools import combinations
from numpy import random
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle as pk


def get_coord(N, k):
    temp = [i +1 for i in range(N)]
    return list(combinations(temp, k))

# generate cov matrix 
# n is number of vertices 
# vals[i] is the covariance of S, S' when intersection is of size i (e.g. we should have vals 0, 1, 2, 3)
def cov_mat(V, variance):
    m = len(V)
    res = np.zeros((m, m))
    for i in range(m):
        res[i][i] = variance[-1]
        for j in range(i):
            int_size = sum([p in V[j] for p in V[i]])
#             print(V[i], V[j], [p in V[j] for p in V[i]])
            res[i][j] = variance[int_size]
            res[j][i] = variance[int_size]
    return res

def check_psd(M):
    try:
        res = np.linalg.cholesky(M)
        return res
    except np.linalg.LinAlgError:
            return None

def sample_gaussain(mean, M):
    return random.multivariate_normal(mean, M)

# variance size 3 subsets
def shift_3(moments):
    mean = 2 * moments[-1] - 1
    res = []
    for i in range(3):
        res.append(4 * moments[i] - 1 - 2 * mean - mean ** 2)
    res.append(1 - mean ** 2)
    return res

def vertex_loc(V):
    res = {}
    for (i, v) in enumerate(V):
        res[v] = i
    return res

def link_dist(samples, V_loc, v_list):
    res = {}
    order = {}
    for i in v_list:
        res[i] = []
        order[i] = []
    for v in V_loc.keys():
        for i in v_list:
            if i in v:
                res[i].append(samples[V_loc[v]])
                order[i].append(v)
    return res, order

# get moments from KO construction
def get_moments(p, s):
    m = p**s
    N = int(3 * m **3 *(m**3 - 1) * (m ** 2 - 1) / p**7)

    # prob of intersection size 3 -> triangle
    p3 = 2 * p **7 /((N - 1)* (N - 2)) 
    p2 = p ** 2 * (p ** 2 - 1) * p ** 5/ math.comb(N - 1, 3)
    p1 = p ** 7 * (p ** 7 - 2 * p **2 + 1)/ (2 * math.comb(N - 1, 4))
    p0 = p3 ** 2
    
    
    print('num triangle', N * (N - 1) * (N - 2) * p3 / 6)
    vals = np.array([p0, p1, p2, p3])
    return vals, N


 # for johnson idempotent 

# spherical functions coefficients  

def fact(n, diff):
    if diff == 0:
        return 1
    res = 1
    for i in range(diff):
        res *= (n + i)
    return res
def phi(n, m, k, l):
    mult = (-1) ** k / math.comb(n - m, k)
    res = 0
    low = max(0, l - m + k)
    high = min(l, k)
#     print(k, l, low, high)
    for i in range(low, high + 1):
        res += math.comb(m - l, k - i) * math.comb(l, i) * fact(n - m - k + 1, k - i) / fact(-1 * m, k - i)
#     print(res, mult)
    return res * mult
# m < n
def PHI(n, m, k):
    return [phi(n, m, k, l) for l in range(m + 1)]

def eigenval(n, m, k):
    return m * (n - m) - k * (n - k + 1)

# size of neighbor i away from m 
# where d(A, B) = A - |A \cap B|
def k_i(n, m, i):
    return math.comb(n - m, i) * math.comb(m, m - i)

# compute values of idempotent matrix entries based on |A \cap B|
def idempotent_vals(n):
    # each row is phi_i(j)
    spherical_fn =  np.array([PHI(n, 3, i) for i in range(0, 4)])
    l = math.comb(n, 3)
    dim = []
    for i in range(0, 4):
        cur = math.comb(n, i) 
        if i > 0:
            cur -= math.comb(n, i - 1)
        dim.append(cur)
    
    return spherical_fn * (np.array(dim)/l)[:, None]

# compute sqrt eigenvalue of covariance matrix
def sqrt_cov(n, vals):
    spherical_fn =  np.array([PHI(n, 3, i) for i in range(0, 4)])
    # p number 
    k_num = [k_i(n, 3, i) for i in range(0, 4)]
    # print(k_num)
    p_num = (spherical_fn * k_num).T
    
    eigen_val = p_num * np.flip(np.array(vals))[:, None]
    total_eig = np.sum(eigen_val, axis = 0)
    sqrt_eig = np.sqrt(total_eig)
    return sqrt_eig

def intersect(A, B):
    return sum([v in A for v in B])

# sample gaussian with give covariance matrix where vals[i] holds value for A, B with intersection of size i
def sample_gaussian(n, vals, gauss):
    l = math.comb(n, 3)
    V = get_coord(n, 3)
    
    idemp_val = idempotent_vals(n)
    sqrt_cov_eig = sqrt_cov(n, vals)
    
#     gauss = np.random.standard_normal(l)
    res = np.zeros(l)
    for i in range(l):
        cur_E = sum([sqrt_cov_eig[k] * idemp_val[k][3 - intersect(V[i], V[i])] for k in range(4)])
        res[i] += gauss[i] * cur_E
        for j in range(i):
#             get sqrt(cov)[i][j]
            cur_E = sum([sqrt_cov_eig[k] * idemp_val[k][3 - intersect(V[i], V[j])] for k in range(4)])
            res[i] += gauss[j] * cur_E
            res[j] += gauss[i] * cur_E
    return res


# for cholesky decomposition of large matrix 
# for cholesky with IO to save memory
# TODO: minimize number of files by batching?

def cholesky_pickle(V, variances, name):
    l = len(V)
    prev = np.zeros(l)
    
    for i in range(l):
        idx = sum([v in V[i] for v in V[0]])
        pk.dump([variances[idx]/variances[-1]], open('cov_' + str(i) + '_' + name + '.p', 'wb'))
    
    for i in range(1, l):
        if i % 100 == 0:
            print('cur index is', i)
        cur = []
#         with open('cov_' + str(i) + '_' + name + '.p', 'rb') as f:
        cur = pk.load(open('cov_' + str(i) + '_' + name + '.p', 'rb'))
        cur_val = np.sqrt(variances[-1] - np.linalg.norm(cur) ** 2)
#         print('cur_val', cur_val)
        for j in range(i, l):
            temp = []
#             with open( 'cov_' + str(j) + '_' + name + '.p', 'rb'):
            temp = pk.load(open( 'cov_' + str(j) + '_' + name + '.p', 'rb'))
            idx = sum([v in V[i] for v in V[j]])
            temp.append((variances[idx] - np.dot(cur, temp))/cur_val)
#             with open( 'cov_' + str(j) + '_' + name + '.p', 'wb'):
            pk.dump(temp, open('cov_' + str(j) + '_' + name + '.p', 'wb'))
                        
def load_cholesky_decomp(name, l):
    res = []
    for i in range(l):
        res.append(pk.load(open( 'cov_' + str(i) + '_' + name + '.p', 'rb')))
    return res

    