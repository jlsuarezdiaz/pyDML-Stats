from dml import (kNN, PCA, LDA, ANMM, LMNN, NCA, NCMML, NCMC, ITML, DMLMJ, MCML,
                 LSI, DML_eig, LDML, KLMNN, KANMM, KDMLMJ, KDA, Euclidean, NCMC_Classifier)

from collections import defaultdict
from utils import datasets as ds
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import NearestCentroid

import numpy as np
import pandas as pd

import time
import traceback


def test_datasets():
    return [('iris', 1),
            ('balance', 1),
            ('sonar', 1)
            ]


def small_datasets():
    return [('appendicitis', 1),
            ('balance', 1),
            ('bupa', 1),
            ('cleveland', 1),
            ('glass', 1),
            ('hepatitis', 1),
            ('ionosphere', 1),
            ('iris', 1),
            ('monk-2', 1),
            ('newthyroid', 1),
            ('sonar', 1),
            ('wine', 1)
            ]


def medium_datasets():
    return [('movement_libras', 1),
            ('pima', 1),
            ('vehicle', 1),
            ('vowel', 1),
            ('wdbc', 1),
            ('wisconsin', 1)
            ]


def large_datasets1():
    return [('segment', 5),
            ('satimage', 5),
            ('winequality-red', 1),
            ('digits', 1)
            ]


def large_datasets2():
    return [('spambase', 1),
            ('optdigits', 5),
            ('twonorm', 5),
            ('titanic', 1)
            ]


def large_datasets3():
    return [('banana', 5),
            ('texture', 5),
            ('ring', 5),
            ('letter', 10)]


def large_datasets4():
    return [('phoneme', 5),
            ('page-blocks', 5),
            ('thyroid', 5),
            ('magic', 10)
            ]


def test_dataset_names():
    return ['iris',
            'balance',
            'sonar'
            ]


def dataset_names():
    return ['appendicitis',
            'balance',
            'banana',
            'bupa',
            'cleveland',
            'glass',
            'hepatitis',
            'ionosphere',
            'iris',
            'letter',
            'magic',
            'monk-2',
            'movement_libras',
            'newthyroid',
            'optdigits',
            'page-blocks',
            'phoneme',
            'ring',
            'segment',
            'sonar',
            'spambase',
            'texture',
            'thyroid',
            'titanic',
            'twonorm',
            'vehicle',
            'vowel',
            'wdbc',
            'wine',
            'winequality-red',
            'wisconsin']


def small_datasets_names():
    return ['appendicitis',
            'balance',
            'bupa',
            'cleveland',
            'glass',
            'hepatitis',
            'ionosphere',
            'iris',
            'monk-2',
            'newthyroid',
            'sonar',
            'wine',
            ]


def medium_datasets_names():
    return ['movement_libras',
            'pima',
            'vehicle',
            'vowel',
            'wdbc',
            'wisconsin',
            ]


def large_datasets_names1():
    return ['segment',
            'satimage',
            'winequality-red',
            'digits',
            ]


def large_datasets_names2():
    return ['spambase',
            'optdigits',
            'twonorm',
            'titanic',
            ]


def large_datasets_names3():
    return ['banana',
            'texture',
            'ring',
            'letter',
            ]


def large_datasets_names4():
    return ['phoneme',
            'page-blocks',
            'thyroid',
            'magic',
            ]


def medium_datasets_ker():
    return [('movement_libras', 1),
            ('pima', 1),
            ('wdbc', 1),
            ]


def large_datasets_ker1():
    return [('segment', 5),
            ('satimage', 5),
            ]


def large_datasets_ker2():
    return [('spambase', 1),
            ('optdigits', 5),
            ('twonorm', 5),
            ]


def large_datasets_ker3():
    return [('banana', 5),
            ]


def large_datasets_ker4():
    return [('phoneme', 5),
            ]


def medium_datasets_ker_names():
    return ['movement_libras',
            'pima',
            'wdbc',
            ]


def large_datasets_ker_names1():
    return ['segment',
            'satimage',
            ]


def large_datasets_ker_names2():
    return ['spambase',
            'optdigits',
            'twonorm'
            ]


def large_datasets_ker_names3():
    return ['banana']


def large_datasets_ker_names4():
    return ['phoneme']


def ker_datasets_names():
    return ['appendicitis',
            'balance',
            'banana',
            'bupa',
            'cleveland',
            'glass',
            'hepatitis',
            'ionosphere',
            'iris',
            'monk-2',
            'movement_libras',
            'newthyroid',
            'optdigits',
            'phoneme',
            'pima'
            'satimage',
            'segment',
            'sonar',
            'spambase',
            'twonorm',
            'wdbc',
            'wine',
            ]


def dim_datasets():
    return [('sonar', 1),
            ('movement_libras', 1),
            ('spambase', 1),
            ]


def dim_datasets_names():
    return ['sonar', 'movement_libras', 'spambase']


def test_dim_datasets():
    return [('sonar', 1)]


def test_dim_datasets_names():
    return ['sonar']


def dim_dimensionalities():
    return [1, 2, 3, 5, 10, 20, 30, 40, 50]


def table_css():
    return "table table-striped table-hover table-bordered table-condensed table-responsive"


def test_knn_algs():
    euclidean = Euclidean()
    lda = LDA()
    dmlmj_3 = DMLMJ(n_neighbors=3)
    dmlmj_5 = DMLMJ(n_neighbors=5)
    dmlmj_7 = DMLMJ(n_neighbors=7)

    return [(euclidean, 'Euclidean', 'euclidean', [3, 5, 7], 'Euclidean()'),
            (lda, 'LDA', 'lda', [3, 5, 7], 'LDA()'),
            (dmlmj_3, 'DMLMJ', 'dmlmj', [3], 'DMLMJ(n_neighbors=3)'),
            (dmlmj_5, 'DMLMJ', 'dmlmj', [5], 'DMLMJ(n_neighbors=5)'),
            (dmlmj_7, 'DMLMJ', 'dmlmj', [7], 'DMLMJ(n_neighbors=7)')
            ]


def basic_knn_algs():
    euclidean = Euclidean()
    lda = LDA()
    nca = NCA()
    lmnn_3 = LMNN(k=3)
    lmnn_5 = LMNN(k=5)
    lmnn_7 = LMNN(k=7)
    lmnn_sgd_3 = LMNN(k=3, solver='SGD', eta0=0.01)
    lmnn_sgd_5 = LMNN(k=5, solver='SGD', eta0=0.01)
    lmnn_sgd_7 = LMNN(k=7, solver='SGD', eta0=0.01)
    itml = ITML()
    dmlmj_3 = DMLMJ(n_neighbors=3)
    dmlmj_5 = DMLMJ(n_neighbors=5)
    dmlmj_7 = DMLMJ(n_neighbors=7)
    mcml = MCML()
    lsi = LSI(supervised=True)
    dml_eig = DML_eig()
    ldml = LDML()

    return [(euclidean, 'Euclidean', 'euclidean', [3, 5, 7], 'Euclidean()'),
            (lda, 'LDA', 'lda', [3, 5, 7], 'LDA()'),
            (nca, 'NCA', 'nca', [3, 5, 7], 'NCA()'),
            (lmnn_3, 'LMNN (SDP)', 'lmnn-sdp', [3], 'LMNN(k=3)'),
            (lmnn_5, 'LMNN (SDP)', 'lmnn-sdp', [5], 'LMNN(k=5)'),
            (lmnn_7, 'LMNN (SDP)', 'lmnn-sdp', [7], 'LMNN(k=7)'),
            (lmnn_sgd_3, 'LMNN (SGD)', 'lmnn-sgd', [3], "LMNN(k=3, solver='SGD', eta0=0.01)"),
            (lmnn_sgd_5, 'LMNN (SGD)', 'lmnn-sgd', [5], "LMNN(k=5, solver='SGD', eta0=0.01)"),
            (lmnn_sgd_7, 'LMNN (SGD)', 'lmnn-sgd', [7], "LMNN(k=7, solver='SGD', eta0=0.01)"),
            (itml, 'ITML', 'itml', [3, 5, 7], "ITML()"),
            (dmlmj_3, 'DMLMJ', 'dmlmj', [3], "DMLMJ(n_neighbors=3)"),
            (dmlmj_5, 'DMLMJ', 'dmlmj', [5], "DMLMJ(n_neighbors=5)"),
            (dmlmj_7, 'DMLMJ', 'dmlmj', [7], "DMLMJ(n_neighbors=7)"),
            (mcml, 'MCML', 'mcml', [3, 5, 7], "MCML()"),
            (lsi, 'LSI', 'lsi', [3, 5, 7], "LSI(supervised=True)"),
            (dml_eig, 'DML-eig', 'dml-eig', [3, 5, 7], "DML_eig()"),
            (ldml, 'LDML', 'ldml', [3, 5, 7], "LDML()")
            ]


def test_kernel_knn_algs():
    euclidean = Euclidean()

    kpca_linear = KernelPCA(kernel="linear")
    kpca_poly2 = KernelPCA(kernel="polynomial", degree=2)
    kpca_poly3 = KernelPCA(kernel="polynomial", degree=3)

    kda_linear = KDA(kernel="linear")
    kda_poly2 = KDA(kernel="polynomial", degree=2)
    kda_poly3 = KDA(kernel="polynomial", degree=3)

    return [(euclidean, 'Euclidean', 'euclidean', [3], 'Euclidean()'),

            (kpca_linear, 'KPCA [Linear]', 'kpca-linear', [3], 'sklearn.decomposition.KernelPCA(kernel="linear")'),
            (kpca_poly2, 'KPCA [Poly-2]', 'kpca-poly-2', [3], 'sklearn.decomposition.KernelPCA(kernel="polynomial", degree=2)'),
            (kpca_poly3, 'KPCA [Poly-3]', 'kpca-poly-3', [3], 'sklearn.decomposition.KernelPCA(kernel="polynomial", degree=3)'),

            (kda_linear, 'KDA [Linear]', 'kda-linear', [3], 'KDA(kernel="linear")'),
            (kda_poly2, 'KDA [Poly-2]', 'kda-poly-2', [3], 'KDA(kernel="polynomial", degree=2)'),
            (kda_poly3, 'KDA [Poly-3]', 'kda-poly-3', [3], 'KDA(kernel="polynomial", degree=3)'),
            ]


def kernel_knn_algs():
    euclidean = Euclidean()

    kpca_linear = KernelPCA(kernel="linear")
    kpca_poly2 = KernelPCA(kernel="polynomial", degree=2)
    kpca_poly3 = KernelPCA(kernel="polynomial", degree=3)
    kpca_rbf = KernelPCA(kernel="rbf")
    kpca_lapl = KernelPCA(kernel="laplacian")

    kda_linear = KDA(kernel="linear")
    kda_poly2 = KDA(kernel="polynomial", degree=2)
    kda_poly3 = KDA(kernel="polynomial", degree=3)
    kda_rbf = KDA(kernel="rbf")
    kda_lapl = KDA(kernel="laplacian")

    kanmm_linear = KANMM(kernel="linear")
    kanmm_poly2 = KANMM(kernel="polynomial", degree=2)
    kanmm_poly3 = KANMM(kernel="polynomial", degree=3)
    kanmm_rbf = KANMM(kernel="rbf")
    kanmm_lapl = KANMM(kernel="laplacian")

    kdmlmj_linear = KDMLMJ(kernel="linear")
    kdmlmj_poly2 = KDMLMJ(kernel="polynomial", degree=2)
    kdmlmj_poly3 = KDMLMJ(kernel="polynomial", degree=3)
    kdmlmj_rbf = KDMLMJ(kernel="rbf")
    kdmlmj_lapl = KDMLMJ(kernel="laplacian")

    klmnn_linear = KLMNN(k=3, kernel="linear")
    klmnn_poly2 = KLMNN(k=3, kernel="polynomial", degree=2)
    klmnn_poly3 = KLMNN(k=3, kernel="polynomimal", degree=3)
    klmnn_rbf = KLMNN(k=3, kernel="rbf")
    klmnn_lapl = KLMNN(k=3, kernel="laplacian")

    return [(euclidean, 'Euclidean', 'euclidean', [3], 'Euclidean()'),

            (kpca_linear, 'KPCA [Linear]', 'kpca-linear', [3], 'sklearn.decomposition.KernelPCA(kernel="linear")'),
            (kpca_poly2, 'KPCA [Poly-2]', 'kpca-poly-2', [3], 'sklearn.decomposition.KernelPCA(kernel="polynomial", degree=2)'),
            (kpca_poly3, 'KPCA [Poly-3]', 'kpca-poly-3', [3], 'sklearn.decomposition.KernelPCA(kernel="polynomial", degree=3)'),
            (kpca_rbf, 'KPCA [RBF]', 'kpca-rbf', [3], 'sklearn.decomposition.KernelPCA(kernel="rbf")'),
            (kpca_lapl, 'KPCA [Laplacian]', 'kpca-laplacian', [3], 'sklearn.decomposition.KernelPCA(kernel="laplacian")'),

            (kda_linear, 'KDA [Linear]', 'kda-linear', [3], 'KDA(kernel="linear")'),
            (kda_poly2, 'KDA [Poly-2]', 'kda-poly-2', [3], 'KDA(kernel="polynomial", degree=2)'),
            (kda_poly3, 'KDA [Poly-3]', 'kda-poly-3', [3], 'KDA(kernel="polynomial", degree=3)'),
            (kda_rbf, 'KDA [RBF]', 'kda-rbf', [3], 'KDA(kernel="rbf")'),
            (kda_lapl, 'KDA [Laplacian]', 'kda-laplacian', [3], 'KDA(kernel="laplacian")'),

            (kanmm_linear, 'KANMM [Linear]', 'kanmm-linear', [3], 'KANMM(kernel="linear")'),
            (kanmm_poly2, 'KANMM [Poly-2]', 'kanmm-poly-2', [3], 'KANMM(kernel="polynomial"), degree=2'),
            (kanmm_poly3, 'KANMM [Poly-3]', 'kanmm-poly-3', [3], 'KANMM(kernel="polynomial"), degree=3'),
            (kanmm_rbf, 'KANMM [RBF]', 'kanmm-rbf', [3], 'KANMM(kernel="rbf")'),
            (kanmm_lapl, 'KANMM [Laplacian]', 'kanmm-laplacian', [3], 'KANMM(kernel="laplacian")'),

            (kdmlmj_linear, 'KDMLMJ [Linear]', 'kdmlmj-linear', [3], 'KDMLMJ(kernel="linear")'),
            (kdmlmj_poly2, 'KDMLMJ [Poly-2]', 'kdmlmj-poly-2', [3], 'KDMLMJ(kernel="poly-2")'),
            (kdmlmj_poly3, 'KDMLMJ [Poly-3]', 'kdmlmj-poly-3', [3], 'KDMLMJ(kernel="poly-3")'),
            (kdmlmj_rbf, 'KDMLMJ [RBF]', 'kdmlmj-rbf', [3], 'KDMLMJ(kernel="rbf")'),
            (kdmlmj_lapl, 'KDMLMJ [Laplacian]', 'kdmlmj-laplacian', [3], 'KDMLMJ(kernel="laplacian")'),

            (klmnn_linear, 'KLMNN [Linear]', 'klmnn-linear', [3], 'KLMNN(k=3, kernel="linear")'),
            (klmnn_poly2, 'KLMNN [Poly-2]', 'klmnn-poly-2', [3], 'KLMNN(k=3, kernel="polynomial", degree=2)'),
            (klmnn_poly3, 'KLMNN [Poly-3]', 'klmnn-poly-3', [3], 'KLMNN(k=3, kernel="polynomial", degree=3)'),
            (klmnn_rbf, 'KLMNN [RBF]', 'klmnn-rbf', [3], 'KLMNN(k=3, kernel="rbf")'),
            (klmnn_lapl, 'KLMNN [Laplacian]', 'klmnn-laplacian', [3], 'KLMNN(k=3, kernel="laplacian")'),
            ]


def ncm_algs():
    euclidean = Euclidean()

    ncmml = NCMML()
    ncmc2 = NCMC(centroids_num=2)
    ncmc3 = NCMC(centroids_num=3)

    return [(euclidean, ['Euclidean + NCM', 'Euclidean + NCMC (2 ctrd)', 'Euclidean + NCMC (3 ctrd)'], ['euclidean-ncm', 'euclidean-ncmc-2', 'euclidean-ncmc-3'], [1, 2, 3], 'Euclidean()'),
            (ncmml, ['NCMML'], ['ncmml'], [1], 'NCMML()'),
            (ncmc2, ['NCMC (2 ctrd)'], ['ncmc-2'], [2], 'NCMC(centroids_num=2)'),
            (ncmc3, ['NCMC (3 ctrd)'], ['ncmc-3'], [3], 'NCMC(centroids_num=3)'),
            ]


def test_dim_algs(dim):
    pca = PCA(num_dims=dim)
    lda = LDA(num_dims=dim)
    anmm_3 = ANMM(num_dims=dim, n_friends=3, n_enemies=3)
    anmm_5 = ANMM(num_dims=dim, n_friends=5, n_enemies=5)
    anmm_7 = ANMM(num_dims=dim, n_friends=7, n_enemies=7)

    return [(pca, 'PCA', 'pca', [3, 5, 7], 'PCA(num_dims=dim)'),
            (lda, 'LDA', 'lda', [3, 5, 7], 'LDA(num_dims=dim)'),
            (anmm_3, 'ANMM', 'anmm', [3], 'ANMM(num_dims=dim, n_friends=3, n_enemies=3)'),
            (anmm_5, 'ANMM', 'anmm', [5], 'ANMM(num_dims=dim, n_friends=5, n_enemies=5)'),
            (anmm_7, 'ANMM', 'anmm', [7], 'ANMM(num_dims=dim, n_friends=7, n_enemies=7)'),
            ]


def dim_algs(dim):
    pca = PCA(num_dims=dim)
    lda = LDA(num_dims=dim)

    anmm_3 = ANMM(num_dims=dim, n_friends=3, n_enemies=3)
    anmm_5 = ANMM(num_dims=dim, n_friends=5, n_enemies=5)
    anmm_7 = ANMM(num_dims=dim, n_friends=7, n_enemies=7)

    lmnn_3 = LMNN(num_dims=dim, k=3, solver='SGD', eta0=0.01)
    lmnn_5 = LMNN(num_dims=dim, k=5, solver='SGD', eta0=0.01)
    lmnn_7 = LMNN(num_dims=dim, k=7, solver='SGD', eta0=0.01)

    nca = NCA(num_dims=dim)

    dmlmj_3 = DMLMJ(num_dims=dim, n_neighbors=3)
    dmlmj_5 = DMLMJ(num_dims=dim, n_neighbors=5)
    dmlmj_7 = DMLMJ(num_dims=dim, n_neighbors=7)

    return [(pca, 'PCA', 'pca', [3, 5, 7], 'PCA(num_dims=dim)'),
            (lda, 'LDA', 'lda', [3, 5, 7], 'LDA(num_dims=dim)'),
            (anmm_3, 'ANMM', 'anmm', [3], 'ANMM(num_dims=dim, n_friends=3, n_enemies=3)'),
            (anmm_5, 'ANMM', 'anmm', [5], 'ANMM(num_dims=dim, n_friends=5, n_enemies=5)'),
            (anmm_7, 'ANMM', 'anmm', [7], 'ANMM(num_dims=dim, n_friends=7, n_enemies=7)'),
            (lmnn_3, 'LMNN', 'lmnn', [3], "LMNN(num_dims=dim, k=3, solver='SGD', eta0=0.01)"),
            (lmnn_5, 'LMNN', 'lmnn', [5], "LMNN(num_dims=dim, k=5, solver='SGD', eta0=0.01)"),
            (lmnn_7, 'LMNN', 'lmnn', [7], "LMNN(num_dims=dim, k=7, solver='SGD', eta0=0.01)"),
            (nca, 'NCA', 'nca', [3, 5, 7], 'NCA(num_dims=dim)'),
            (dmlmj_3, 'DMLMJ', 'dmlmj', [3], 'DMLMJ(num_dims=dim, n_neighbors=3)'),
            (dmlmj_5, 'DMLMJ', 'dmlmj', [5], 'DMLMJ(num_dims=dim, n_neighbors=5)'),
            (dmlmj_7, 'DMLMJ', 'dmlmj', [7], 'DMLMJ(num_dims=dim, n_neighbors=7)'),
            ]


def test_basic_knn(alg_list, datasets, textkey="cv-basic", testname="BASIC", seed=28):
    """
    Evaluates the algorithms specified in the datasets provided.

    Parameters
    ----------

    alg_list : list
        The list of algorithms. Each item must be a quadruple (alg, name, key, ks, cons), where 'alg' is the algorithm, 'name'
        is the string name, 'key' is a key-name for the alg, 'ks' is the list of neighbors to consider in k-NN, and cons
        is the initialization code of the algorithm.

    datasets : list
        The list of datasets to use. Each item must be a pair (str, frac), where 'str' is the name of the dataset
        and 'frac' is the fraction of the dataset to take (for big datasets).

    """
    print("* " + testname + " TEST STARTED")
    mms = MinMaxScaler()
    rownames = ["FOLD " + str(i + 1) for i in range(10)]

    results = {}

    for dset, f in datasets:
        print("** DATASET ", dset)

        folds, [n, d, c] = ds.reduced_dobscv10(dset, f)

        print("** SIZE ", n, " x ", d, " [", c, " classes]")

        results[dset] = {}

        norm_folds = []

        for i, (xtr, ytr, xtst, ytst) in enumerate(folds):
            print("*** NORMALIZING FOLD ", i + 1)
            # Normalizing
            xtr = mms.fit_transform(xtr)
            xtst = mms.transform(xtst)
            norm_folds.append((xtr, ytr, xtst, ytst))

        for j, (dml, dml_name, dml_key, ks, cons) in enumerate(alg_list):
            print("*** EVALUATING DML ", dml_name)

            results[dset][dml_key] = defaultdict(lambda: np.zeros([12, 3]))

            for i, (xtr, ytr, xtst, ytst) in enumerate(norm_folds):
                print("**** FOLD ", i + 1)
                np.random.seed(seed)

                try:
                    print("***** TRAINING")
                    start = time.time()     # Start timer
                    dml.fit(xtr, ytr)       # Fitting distance
                    end = time.time()       # Stop timer
                    elapsed = end - start   # Timer measurement

                    for k in ks:
                        print("****** TEST K = ", k)
                        knn = kNN(k, dml)
                        knn.fit(xtr, ytr)

                        results[dset][dml_key][k][i, 0] = knn.score()            # Train score
                        results[dset][dml_key][k][i, 1] = knn.score(xtst, ytst)  # Test score
                        results[dset][dml_key][k][i, 2] = elapsed                # Time score
                except:
                    print("--- ERROR IN DML ", dml_name)
                    for k in ks:
                        results[dset][dml_key][k][i, 0] = np.nan          # Train score
                        results[dset][dml_key][k][i, 1] = np.nan          # Test score
                        results[dset][dml_key][k][i, 2] = np.nan          # Time score

                    traceback.print_exc()

            for k in ks:
                results[dset][dml_key][k][10, :] = np.mean(results[dset][dml_key][k][:10, :], axis=0)
                results[dset][dml_key][k][11, :] = np.std(results[dset][dml_key][k][:10, :], axis=0)

                # Saving results
                r = pd.DataFrame(results[dset][dml_key][k], columns=['TRAIN', 'TEST', 'TIME'], index=rownames + ["MEAN", "STD"])

                r.to_csv("../results/" + textkey + "-" + dml_key + "-" + str(k) + "nn-" + dset + ".csv")
                r.to_html("../results/" + textkey + "-" + dml_key + "-" + str(k) + "nn-" + dset + ".html", classes=[table_css(), "kfoldtable meanstd"])

                print("RESULTS: ", dset, ", dml = ", dml_name, ", k = ", k)
                print(r)


def test_ker_knn(alg_list, datasets, seed=28):
    test_basic_knn(alg_list, datasets, "cv-ker", "KERNEL", seed)


def test_ncm(alg_list, datasets, seed=28):
    """
    Evaluates the algorithms specified in the datasets provided.

    Parameters
    ----------

    alg_list : list
        The list of algorithms. Each item must be a quadruple (alg, name, key, ks, cons), where 'alg' is the algorithm, 'name'
        is the string name, 'key' is a key-name for the alg, 'ks' is the list of neighbors to consider in k-NN, and cons
        is the initialization code of the algorithm.

    datasets : list
        The list of datasets to use. Each item must be a pair (str, frac), where 'str' is the name of the dataset
        and 'frac' is the fraction of the dataset to take (for big datasets).

    """
    print("* NEAREST CENTROIDS TEST STARTED")
    mms = MinMaxScaler()
    rownames = ["FOLD " + str(i + 1) for i in range(10)]

    results = {}

    for dset, f in datasets:
        print("** DATASET ", dset)

        folds, [n, d, c] = ds.reduced_dobscv10(dset, f)

        print("** SIZE ", n, " x ", d, " [", c, " classes]")

        results[dset] = {}

        norm_folds = []

        for i, (xtr, ytr, xtst, ytst) in enumerate(folds):
            print("*** NORMALIZING FOLD ", i + 1)
            # Normalizing
            xtr = mms.fit_transform(xtr)
            xtst = mms.transform(xtst)
            norm_folds.append((xtr, ytr, xtst, ytst))

        for j, (dml, dml_name, dml_key, ks, cons) in enumerate(alg_list):
            print("*** EVALUATING DML ", dml_name)

            results[dset] = defaultdict(lambda: np.zeros([12, 3]))

            for i, (xtr, ytr, xtst, ytst) in enumerate(norm_folds):
                print("**** FOLD ", i + 1)
                np.random.seed(seed)

                try:
                    print("***** TRAINING")
                    start = time.time()     # Start timer
                    dml.fit(xtr, ytr)       # Fitting distance
                    end = time.time()       # Stop timer
                    elapsed = end - start   # Timer measurement

                    xtr2 = dml.transform()
                    xtst2 = dml.transform(xtst)

                    for namek, keyk, k in zip(dml_name, dml_key, ks):
                        print("****** TEST NCM [", k, " CTRD]")

                        if k == 1:
                            ncm = NearestCentroid()
                        else:
                            ncm = NCMC_Classifier(k)

                        ncm.fit(xtr2, ytr)

                        results[dset][keyk][i, 0] = ncm.score(xtr2, ytr)            # Train score
                        results[dset][keyk][i, 1] = ncm.score(xtst2, ytst)  # Test score
                        results[dset][keyk][i, 2] = elapsed                # Time score
                except:
                    print("--- ERROR IN DML ", dml_name)
                    for keyk in dml_key:
                        results[dset][keyk][i, 0] = np.nan          # Train score
                        results[dset][keyk][i, 1] = np.nan          # Test score
                        results[dset][keyk][i, 2] = np.nan          # Time score

                    traceback.print_exc()

            for keyk, namek in zip(dml_key, dml_name):
                results[dset][keyk][10, :] = np.mean(results[dset][keyk][:10, :], axis=0)
                results[dset][keyk][11, :] = np.std(results[dset][keyk][:10, :], axis=0)

                # Saving results
                r = pd.DataFrame(results[dset][keyk], columns=['TRAIN', 'TEST', 'TIME'], index=rownames + ["MEAN", "STD"])

                r.to_csv("../results/cv-ncm-" + keyk + "-" + dset + ".csv")
                r.to_html("../results/cv-ncm-" + keyk + "-" + dset + ".html", classes=[table_css(), "kfoldtable meanstd"])

                print("RESULTS: ", dset, ", dml = ", namek)
                print(r)


def test_dim_knn(datasets, dimensions, dim_alg_function=dim_algs, add_nclass1=True, add_maxdim=True, seed=28):
    for dset, f in datasets:
        folds, [n, d, c] = ds.reduced_dobscv10(dset, f)
        dset_dims = dimensions
        if add_nclass1:
            dset_dims.append(c - 1)
        if add_maxdim:
            dset_dims.append(d)
        dset_dims = np.unique(dset_dims)

        for dim in dset_dims:
            test_basic_knn(dim_alg_function(dim), [(dset, f)], "cv-dim-" + str(dim), "DIM " + str(dim), seed)


def recopilate_test_algs():
    return [('Euclidean', 'euclidean'),
            ('LDA', 'lda'),
            ('DMLMJ', 'dmlmj'),
            ]


def recopilate_basic_algs():
    return [('Euclidean', 'euclidean'),
            ('LDA', 'lda'),
            ('ITML', 'itml'),
            ('DMLMJ', 'dmlmj'),
            ('NCA', 'nca'),
            ('LMNN [SDP]', 'lmnn-sdp'),
            ('LMNN [SGD]', 'lmnn-sgd'),
            ('LSI', 'lsi'),
            ('DML-eig', 'dml-eig'),
            ('MCML', 'mcml'),
            ('LDML', 'ldml'),
            ]


def recopilate_ncm_algs():
    return [('Euclidean + NCM', 'euclidean-ncm'),
            ('NCMML', 'ncmml'),
            ('Euclidean + NCMC (2 ctrd)', 'euclidean-ncmc-2'),
            ('NCMC (2 ctrd)', 'ncmc-2'),
            ('Euclidean + NCMC (3 ctrd)', 'euclidean-ncmc-3'),
            ('NCMC (3 ctrd)', 'ncmc-3'),
            ]


def recopilate_test_kernel_algs():
    return [('Euclidean', 'euclidean'),

            ('KPCA [Linear]', 'kpca-linear'),
            ('KPCA [Poly-2]', 'kpca-poly-2'),
            ('KPCA [Poly-3]', 'kpca-poly-3'),

            ('KDA [Linear]', 'kda-linear'),
            ('KDA [Poly-2]', 'kda-poly-2'),
            ('KDA [Poly-3]', 'kda-poly-3'),
            ]


def recopilate_kernel_algs():
    return [('Euclidean', 'euclidean'),

            ('KPCA [Linear]', 'kpca-linear'),
            ('KPCA [Poly-2]', 'kpca-poly-2'),
            ('KPCA [Poly-3]', 'kpca-poly-3'),
            ('KPCA [RBF]', 'kpca-rbf'),
            ('KPCA [Laplacian]', 'kpca-laplacian'),

            ('KDA [Linear]', 'kda-linear'),
            ('KDA [Poly-2]', 'kda-poly-2'),
            ('KDA [Poly-3]', 'kda-poly-3'),
            ('KDA [RBF]', 'kda-rbf'),
            ('KDA [Laplacian]', 'kda-laplacian'),

            ('KANMM [Linear]', 'kanmm-linear'),
            ('KANMM [Poly-2]', 'kanmm-poly-2'),
            ('KANMM [Poly-3]', 'kanmm-poly-3'),
            ('KANMM [RBF]', 'kanmm-rbf'),
            ('KANMM [Laplacian]', 'kanmm-laplacian'),

            ('KDMLMJ [Linear]', 'kdmlmj-linear'),
            ('KDMLMJ [Poly-2]', 'kdmlmj-poly-2'),
            ('KDMLMJ [Poly-3]', 'kdmlmj-poly-3'),
            ('KDMLMJ [RBF]', 'kdmlmj-rbf'),
            ('KDMLMJ [Laplacian]', 'kdmlmj-laplacian'),

            ('KLMNN [Linear]', 'klmnn-linear'),
            ('KLMNN [Poly-2]', 'klmnn-poly-2'),
            ('KLMNN [Poly-3]', 'klmnn-poly-3'),
            ('KLMNN [RBF]', 'klmnn-rbf'),
            ('KLMNN [Laplacian]', 'klmnn-laplacian'),
            ]


def recopilate_test_dim_algs():
    return [('PCA', 'pca'),
            ('LDA', 'lda'),
            ('ANMM', 'anmm')
            ]


def recopilate_dim_algs():
    return [('PCA', 'pca'),
            ('LDA', 'lda'),
            ('ANMM', 'anmm'),
            ('DMLMJ', 'dmlmj'),
            ('NCA', 'nca'),
            ('LMNN', 'lmnn')
            ]


def add_avg(df):
    '''
    Adds avg ranking and score to a dataframe.
    '''
    avgmean = pd.Series(df.mean(), name="AVG SCORE")
    avgrank = pd.Series(df.rank(axis=1, ascending=False).fillna(df.columns.size).mean(), name="AVG RANKING")
    return df.append([avgmean, avgrank])


def recopilate_basic(alg_list, dataset_names, textkey="cv-basic", outkey="basic", ks=[3, 5, 7]):
    """
        Recopilate results

        Parameters
        ----------

        alg_list : a list with the algorithms. Each item must be a pair (name, key).

        dataset_names : the names of the datasets
    """
    print("* RECOPILATING: ", textkey)

    alg_names = list(map(lambda x: x[0], alg_list))
    alg_keys = list(map(lambda x: x[1], alg_list))

    final_results = {}
    for k in ks:
        print("** K = ", k)
        final_results[k] = {}
        final_results[k]['train'] = np.empty([len(dataset_names), len(alg_list)])
        final_results[k]['test'] = np.empty([len(dataset_names), len(alg_list)])
        final_results[k]['time'] = np.empty([len(dataset_names), len(alg_list)])

        final_results[k]['train'] = pd.DataFrame(final_results[k]['train'], index=dataset_names, columns=alg_names)
        final_results[k]['test'] = pd.DataFrame(final_results[k]['test'], index=dataset_names, columns=alg_names)
        final_results[k]['time'] = pd.DataFrame(final_results[k]['time'], index=dataset_names, columns=alg_names)

        for dts in dataset_names:
            for algname, algkey in alg_list:
                filename = "../results/" + textkey + "-" + algkey + "-" + str(k) + "nn-" + dts + ".csv"
                print("**** READING ", filename)
                partial_results = pd.read_csv(filename, index_col=0)
                final_results[k]['train'][algname][dts] = partial_results['TRAIN']['MEAN']
                final_results[k]['test'][algname][dts] = partial_results['TEST']['MEAN']
                final_results[k]['time'][algname][dts] = partial_results['TIME']['MEAN']

        final_results[k]['train'] = add_avg(final_results[k]['train'])
        final_results[k]['test'] = add_avg(final_results[k]['test'])
        final_results[k]['time'] = add_avg(final_results[k]['time'])

        print("*** SAVING")

        final_results[k]['train'].to_csv('../results/' + outkey + '-' + str(k) + 'nn-train.csv')
        final_results[k]['test'].to_csv('../results/' + outkey + '-' + str(k) + 'nn-test.csv')
        final_results[k]['time'].to_csv('../results/' + outkey + '-' + str(k) + 'nn-time.csv')
        final_results[k]['train'].to_latex('../results/' + outkey + '-' + str(k) + 'nn-train.tex')
        final_results[k]['test'].to_latex('../results/' + outkey + '-' + str(k) + 'nn-test.tex')
        final_results[k]['time'].to_latex('../results/' + outkey + '-' + str(k) + 'nn-time.tex')
        final_results[k]['train'].to_html('../results/' + outkey + '-' + str(k) + 'nn-train.html', classes=[table_css(), "maxhighlightable withavg"])
        final_results[k]['test'].to_html('../results/' + outkey + '-' + str(k) + 'nn-test.html', classes=[table_css(), "maxhighlightable withavg"])
        final_results[k]['time'].to_html('../results/' + outkey + '-' + str(k) + 'nn-time.html', classes=[table_css(), "maxhighlightable withavg"])


def recopilate_kernel(alg_list, dataset_names, ks=[3]):
    recopilate_basic(alg_list, dataset_names, "cv-ker", "ker", ks)


def recopilate_ncm(alg_list, dataset_names):
    print("* RECOPILATING: ncm")

    alg_names = list(map(lambda x: x[0], alg_list))
    alg_keys = list(map(lambda x: x[1], alg_list))

    final_results = {}

    final_results['train'] = np.empty([len(dataset_names), len(alg_list)])
    final_results['test'] = np.empty([len(dataset_names), len(alg_list)])
    final_results['time'] = np.empty([len(dataset_names), len(alg_list)])

    final_results['train'] = pd.DataFrame(final_results['train'], index=dataset_names, columns=alg_names)
    final_results['test'] = pd.DataFrame(final_results['test'], index=dataset_names, columns=alg_names)
    final_results['time'] = pd.DataFrame(final_results['time'], index=dataset_names, columns=alg_names)

    for dts in dataset_names:
        for algname, algkey in alg_list:
            filename = "../results/cv-ncm-" + algkey + "-" + dts + ".csv"
            print("*** READING ", filename)
            partial_results = pd.read_csv(filename, index_col=0)

            final_results['train'][algname][dts] = partial_results['TRAIN']['MEAN']
            final_results['test'][algname][dts] = partial_results['TEST']['MEAN']
            final_results['time'][algname][dts] = partial_results['TIME']['MEAN']

    final_results['train'] = add_avg(final_results['train'])
    final_results['test'] = add_avg(final_results['test'])
    final_results['time'] = add_avg(final_results['time'])

    print("** SAVING")

    final_results['train'].to_csv("../results/ncm-train.csv")
    final_results['test'].to_csv("../results/ncm-test.csv")
    final_results['time'].to_csv("../results/ncm-time.csv")
    final_results['train'].to_latex("../results/ncm-train.tex")
    final_results['test'].to_latex("../results/ncm-test.tex")
    final_results['time'].to_latex("../results/ncm-time.tex")
    final_results['train'].to_html("../results/ncm-train.html", classes=[table_css(), "maxhighlightable withavg"])
    final_results['test'].to_html("../results/ncm-test.html", classes=[table_css(), "maxhighlightable withavg"])
    final_results['time'].to_html("../results/ncm-time.html", classes=[table_css(), "maxhighlightable withavg"])


def recopilate_dim(alg_list, dataset_names, dimensions, ks=[3, 5, 7], add_nclass1=True, add_maxdim=True):
    print("* RECOPILATING: dim")
    dimension_names = dimensions.copy()
    if add_nclass1:
        dimension_names.append("N. Classes - 1")
    if add_maxdim:
        dimension_names.append("Max. Dimension")

    alg_names = list(map(lambda x: x[0], alg_list))
    alg_keys = list(map(lambda x: x[1], alg_list))

    final_results = {}

    for dst in dataset_names:
        print("** DATASET ", dst)
        final_results[dst] = {}
        folds, [n, d, c] = ds.dobscv10(dst)

        ds_dimensions = dimensions.copy()
        if add_nclass1:
            ds_dimensions.append(c - 1)
        if add_maxdim:
            ds_dimensions.append(d)

        for k in ks:
            print("*** K = ", k)

            final_results[dst][k] = {}

            final_results[dst][k]['train'] = np.empty([len(dimension_names), len(alg_list)])
            final_results[dst][k]['test'] = np.empty([len(dimension_names), len(alg_list)])
            final_results[dst][k]['time'] = np.empty([len(dimension_names), len(alg_list)])

            final_results[dst][k]['train'] = pd.DataFrame(final_results[dst][k]['train'], index=dimension_names, columns=alg_names)
            final_results[dst][k]['test'] = pd.DataFrame(final_results[dst][k]['test'], index=dimension_names, columns=alg_names)
            final_results[dst][k]['time'] = pd.DataFrame(final_results[dst][k]['time'], index=dimension_names, columns=alg_names)

            for algname, algkey in alg_list:
                for dimname, dim in zip(dimension_names, ds_dimensions):
                    filename = "../results/cv-dim-" + str(dim) + "-" + algkey + "-" + str(k) + "nn-" + dst + ".csv"
                    print("***** READING ", filename)

                    if algkey == "lda" and dim > c - 1:  # Ignore LDA at dimension higher than c-1
                        print("****** IGNORED LDA AT DIM ", dim)
                        final_results[dst][k]['train'][algname][dimname] = np.nan
                        final_results[dst][k]['test'][algname][dimname] = np.nan
                        final_results[dst][k]['time'][algname][dimname] = np.nan

                    else:
                        partial_results = pd.read_csv(filename, index_col=0)

                        final_results[dst][k]['train'][algname][dimname] = partial_results['TRAIN']['MEAN']
                        final_results[dst][k]['test'][algname][dimname] = partial_results['TEST']['MEAN']
                        final_results[dst][k]['time'][algname][dimname] = partial_results['TIME']['MEAN']

            print("**** SAVING")
            final_results[dst][k]['train'].to_csv("../results/dim-" + str(k) + "nn-" + dst + "-train.csv")
            final_results[dst][k]['test'].to_csv("../results/dim-" + str(k) + "nn-" + dst + "-test.csv")
            final_results[dst][k]['time'].to_csv("../results/dim-" + str(k) + "nn-" + dst + "-time.csv")
            final_results[dst][k]['train'].to_latex("../results/dim-" + str(k) + "nn-" + dst + "-train.tex")
            final_results[dst][k]['test'].to_latex("../results/dim-" + str(k) + "nn-" + dst + "-test.tex")
            final_results[dst][k]['time'].to_latex("../results/dim-" + str(k) + "nn-" + dst + "-time.tex")
            final_results[dst][k]['train'].to_html("../results/dim-" + str(k) + "nn-" + dst + "-train.html", classes=[table_css(), "maxhighlightable dimchartable"])
            final_results[dst][k]['test'].to_html("../results/dim-" + str(k) + "nn-" + dst + "-test.html", classes=[table_css(), "maxhighlightable dimchartable"])
            final_results[dst][k]['time'].to_html("../results/dim-" + str(k) + "nn-" + dst + "-time.html", classes=[table_css(), "maxhighlightable dimchartable"])
