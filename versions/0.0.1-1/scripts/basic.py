import sys
import test_framework as tf

error = False
if len(sys.argv) > 1:
    mod = sys.argv[1]

    if mod == "test":
        alg_list = tf.test_knn_algs()
        datasets = tf.test_datasets()

    elif mod == "small":
        alg_list = tf.basic_knn_algs()
        datasets = tf.small_datasets()

    elif mod == "medium":
        alg_list = tf.basic_knn_algs()
        datasets = tf.medium_datasets()

    elif mod == "large1":
        alg_list = tf.basic_knn_algs()
        datasets = tf.large_datasets1()

    elif mod == "large2":
        alg_list = tf.basic_knn_algs()
        datasets = tf.large_datasets2()

    elif mod == "large3":
        alg_list = tf.basic_knn_algs()
        datasets = tf.large_datasets3()

    elif mod == "large4":
        alg_list = tf.basic_knn_algs()
        datasets = tf.large_datasets4()

    else:
        print("Invalid option: ", mod)
        error = True

else:
    error = True

if error:
    print("Please run this script with one of the following arguments:")
    print("- test: run the test for the basic experiment.")
    print("- small: run the basic experiment with the small datasets.")
    print("- medium: run the basic experiment with the medium datasets.")
    print("- large1: run the basic experiment with the large datasets (part 1).")
    print("- large2: run the basic experiment with the large datasets (part 2).")
    print("- large3: run the basic experiment with the large datasets (part 3).")
    print("- large4: run the basic experiment with the large datasets (part 4).")

else:
    tf.test_basic_knn(alg_list, datasets)
