import numpy as np

__all__ = ['bimatch', 'fast_bimatch']


def search_path(u, n, weight, lx, ly, sx, sy, match):

    sx[u] = True
    for v in range(n):
        if ((not sy[v]) and abs(lx[u] + ly[v] - weight[u][v]) < 0.000001):
            sy[v] = True
            if (match[v] == -1 or search_path(match[v], n, weight, lx, ly, sx,
                                              sy, match)):
                match[v] = u
                return True

    return False


def Kuhn_Munkras(weight):

    n = weight.shape[0]
    ly = [0 for i in range(n)]
    lx = [weight[i].max() for i in range(n)]
    match = [-1 for i in range(n)]

    for u in range(n):
        while True:
            sx = [False for i in range(n)]
            sy = [False for i in range(n)]
            if search_path(u, n, weight, lx, ly, sx, sy, match):
                break
            inc = 999999
            for i in range(n):
                if sx[i]:
                    for j in range(n):
                        if ((not sy[j])
                                and ((lx[i] + ly[j] - weight[i][j]) < inc)):
                            inc = lx[i] + ly[j] - weight[i][j]
            if inc == 0:
                print('inc==0!')
            for i in range(n):
                if sx[i]:
                    lx[i] -= inc
                if sy[i]:
                    ly[i] += inc

    sum = 0
    for i in range(n):
        if match[i] >= 0:
            sum += weight[match[i]][i]

    return match, sum


def bimatch(weight_input, thr=0):
    """ Base bipartite match.
    ----------
    Arguments:
    weight_input: weight matrix. Should be numpy format.
    ----------
    Keyword Arguments:
    thr (default=0): mininum weight threshold.
    ----------
    Return:
    result: the matched indexes. (-1 denotes for not matched)
    sum: sum of the total match.
    """

    nx = weight_input.shape[0]
    ny = weight_input.shape[1]
    if nx == ny:
        weight = weight_input.copy()
    else:
        dim = max(nx, ny)
        weight = np.zeros([dim, dim])
        weight[:nx, :ny] = weight_input.copy()
    match, sum = Kuhn_Munkras(weight)
    result = [-1 for i in range(nx)]
    for i, j in enumerate(match):
        if weight[j, i] > thr:
            result[j] = i
    return result, sum


def fast_bimatch(weight_input, thr=0):
    """ Fast bipartite match when the weight matrix is much wider compared to its height.
    
    ----------
    Arguments:
    weight_input: weight matrix. Should be numpy format.
    ----------
    Keyword Arguments:
    thr (default=0): mininum weight threshold.
    ----------
    Return:
    result: the matched indexes. (-1 denotes for not matched)
    sum: sum of the total match.
    ----------
    Note:
    The result matches may be diffrent from bimatch because there may
    be some wights with the same value, and only some of them be
    kept. But the output sum will be the same.
    Speed test show that 100 faster when weight shape = 5:200
    Equal when weigth shape = 50:200
    """

    height = weight_input.shape[0]
    idx = weight_input.argsort()[:, -1:-1 - height:-1]
    idx = list(set(idx.flatten().tolist()))
    weight = weight_input[:, idx]

    result, sum = bimatch(weight, thr=thr)
    result = map(lambda x: idx[x], result)

    return result, sum
    