import sys
import test_framework as tf

error = False
if len(sys.argv) > 1:
    mod = sys.argv[1]

    if mod == "test":
        alg_function = tf.test_dim_algs
        datasets = tf.test_dim_datasets()

    elif mod.isdigit():
        alg_function = tf.dim_algs
        datasets = [tf.dim_datasets()[int(mod)]]

    else:
        print("Invalid option: ", mod)
        error = True

else:
    error = True

if error:
    print("Please run this script with one of the following arguments:")
    print("- test: run the test for the dimensionality experiment.")
    print("- i, where i is an integer between 0 and ", len(tf.dim_datasets()) - 1, ": run the dimensionality experiment with the i-th dataset.")

else:
    tf.test_dim_knn(datasets, tf.dim_dimensionalities(), alg_function)
