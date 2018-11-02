import sys
import test_framework as tf

error = False
if len(sys.argv) > 1:
    mod = sys.argv[1]

    if mod == "test":
        alg_list = tf.test_kernel_knn_algs()
        datasets = tf.test_datasets()

    elif mod == "small":
        alg_list = tf.kernel_knn_algs()
        datasets = tf.small_datasets()

    elif mod == "medium":
        alg_list = tf.ker_knn_algs()
        datasets = tf.medium_datasets_ker()

    elif mod == "large1":
        alg_list = tf.ker_knn_algs()
        datasets = tf.large_datasets_ker1()

    elif mod == "large2":
        alg_list = tf.ker_knn_algs()
        datasets = tf.large_datasets_ker2()

    elif mod == "large3":
        alg_list = tf.ker_knn_algs()
        datasets = tf.large_datasets_ker3()

    elif mod == "large4":
        alg_list = tf.ker_knn_algs()
        datasets = tf.large_datasets_ker4()

    else:
        print("Invalid option: ", mod)
        error = True

else:
    error = True

if error:
    print("Please run this script with one of the following arguments:")
    print("- test: run the test for the kernel experiment.")
    print("- small: run the kernel experiment with the small datasets.")
    print("- medium: run the kernel experiment with the medium datasets.")
    print("- large1: run the kernel experiment with the large datasets (part 1).")
    print("- large2: run the kernel experiment with the large datasets (part 2).")
    print("- large3: run the kernel experiment with the large datasets (part 3).")
    print("- large4: run the kernel experiment with the large datasets (part 4).")

else:
    tf.test_ker_knn(alg_list, datasets)
