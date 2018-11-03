import sys
import test_framework as tf

error = False
if len(sys.argv) > 1:
    mod = sys.argv[1]

    if mod == "test":
        alg_list = tf.recopilate_test_algs()
        dataset_names = tf.test_dataset_names()

        alg_list_ker = tf.recopilate_test_kernel_algs()
        dataset_names_ker = tf.test_dataset_names()

        alg_list_ncm = tf.recopilate_ncm_algs()

        alg_list_dim = tf.recopilate_test_dim_algs()
        dataset_names_dim = tf.test_dim_datasets_names()

    elif mod == "all":
        alg_list = tf.recopilate_basic_algs()
        dataset_names = tf.dataset_names()

        alg_list_ker = tf.recopilate_kernel_algs()
        dataset_names_ker = tf.ker_datasets_names()

        alg_list_ncm = tf.recopilate_ncm_algs()

        alg_list_dim = tf.recopilate_dim_algs()
        dataset_names_dim = tf.dim_datasets_names()

    elif mod == "basic":
        if len(sys.argv) <= 2:
            error = True
        elif sys.argv[2] not in ["small", "medium", "large1", "large2", "large3", "large4", "all"]:
            print("Invalid option: ", sys.argv[2])
            error = True

    elif mod == "ncm":
        if len(sys.argv) <= 2:
            error = True
        elif sys.argv[2] not in ["small", "medium", "large1", "large2", "large3", "large4", "all"]:
            print("Invalid option: ", sys.argv[2])
            error = True

    elif mod == "ker":
        if len(sys.argv) <= 2:
            error = True
        elif sys.argv[2] not in ["small", "medium", "large1", "large2", "large3", "large4", "all"]:
            print("Invalid option: ", sys.argv[2])
            error = True

    elif mod == "dim":
        if len(sys.argv) <= 2:
            error = True
        elif sys.argv[2] not in ["0", "1", "2", "all"]:
            print("Invalid option: ", sys.argv[2])
            error = True

    else:
        print("Invalid option: ", mod)
        error = True

else:
    error = True

if error:
    print("Please run this script with one of the following arguments:")
    print("- test: run the test for the recopilation.")
    print("- all: run the recopilation of all results.")
    print("- basic [small|medium|large1|large2|large3|large4|all]: run the recopilation of the basic experiments in the specified datasets.")
    print("- ncm [small|medium|large1|large2|large3|large4|all]: run the recopilation of the ncm experiments in the specified datasets.")
    print("- ker [small|medium|large1|large2|large3|large4|all]: run the recopilation of the kernel experiments in the specified datasets.")
    print("- dim [0|1|2|all]: run the recopilation of the dimensionality experiments in the specified datasets.")
else:

    if mod in ["test", "all"]:
        tf.recopilate_basic(alg_list, dataset_names)
        tf.recopilate_kernel(alg_list_ker, dataset_names_ker)
        tf.recopilate_ncm(alg_list_ncm, dataset_names)
        tf.recopilate_dim(alg_list_dim, dataset_names_dim, tf.dim_dimensionalities())

    elif mod == "basic":
        if sys.argv[2] == "small":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.small_datasets_names())
        elif sys.argv[2] == "medium":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.medium_datasets_names())
        elif sys.argv[2] == "large1":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.large_datasets_names1())
        elif sys.argv[2] == "large2":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.large_datasets_names2())
        elif sys.argv[2] == "large3":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.large_datasets_names3())
        elif sys.argv[2] == "large4":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.large_datasets_names4())
        elif sys.argv[2] == "all":
            tf.recopilate_basic(tf.recopilate_basic_algs(), tf.dataset_names())

    elif mod == "ncm":
        if sys.argv[2] == "small":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.small_datasets_names())
        elif sys.argv[2] == "medium":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.medium_datasets_names())
        elif sys.argv[2] == "large1":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.large_datasets_names1())
        elif sys.argv[2] == "large2":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.large_datasets_names2())
        elif sys.argv[2] == "large3":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.large_datasets_names3())
        elif sys.argv[2] == "large4":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.large_datasets_names4())
        elif sys.argv[2] == "all":
            tf.recopilate_ncm(tf.recopilate_ncm_algs(), tf.dataset_names())

    elif mod == "ker":
        if sys.argv[2] == "small":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.small_datasets_names())
        elif sys.argv[2] == "medium":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.medium_datasets_ker_names())
        elif sys.argv[2] == "large1":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.large_datasets_ker_names1())
        elif sys.argv[2] == "large2":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.large_datasets_ker_names2())
        elif sys.argv[2] == "large3":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.large_datasets_ker_names3())
        elif sys.argv[2] == "large4":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.large_datasets_ker_names4())
        elif sys.argv[2] == "all":
            tf.recopilate_kernel(tf.recopilate_kernel_algs(), tf.ker_datasets_names())

    elif mod == "dim":
        if sys.argv[2].isdigit():
            tf.recopilate_dim(tf.recopilate_dim_algs(), [tf.dim_datasets_names()[int(sys.argv[2])]], tf.dim_dimensionalities())
        elif sys.argv[2] == "all":
            tf.recopilate_dim(tf.recopilate_dim_algs(), tf.dim_datasets_names(), tf.dim_dimensionalities())
