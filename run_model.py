import numpy as np
import gpflow
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt


def run_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validate: np.ndarray,
    y_validate: np.ndarray,
    n_basis: int,
    kernel: str,
):

    if n_basis != np.shape(y_train)[-1]:
        print("basis function expectation doesn't match data\n")
        exit()

    kernel_shape = np.shape(x_train)[-1]
    lengthscales = np.ones(kernel_shape)
    kernels = {
        "RBF": gpflow.kernels.RBF(kernel_shape, lengthscales=lengthscales)
        + gpflow.kernels.Linear(kernel_shape),
        "Matern52": gpflow.kernels.Matern52(kernel_shape, lengthscales=lengthscales)
        + gpflow.kernels.Linear(kernel_shape),
        "Matern32": gpflow.kernels.Matern32(kernel_shape, lengthscales=lengthscales)
        + gpflow.kernels.Linear(kernel_shape),
        "Matern12": gpflow.kernels.Matern12(kernel_shape, lengthscales=lengthscales)
        + gpflow.kernels.Linear(kernel_shape),
    }
    if kernel in ["RBF", "Matern52", "Matern32", "Matern12"]:
        model = gpflow.models.GPR((x_train, y_train), kernel=kernels[kernel])
        print(model)
    elif kernel == "Auto":
        score = 1e20
        for n, k in enumerate(kernels):
            print(kernels[k])
            model = gpflow.models.GPR((x_train, y_train), kernel=kernels[k])
            opt = gpflow.optimizers.Scipy()
            result = opt.minimize(model.training_loss, model.trainable_variables)
            outs = model.posterior().predict_f(x_validate)
            new_score = sum(sum((((outs[0] - y_validate) ** 2) / outs[0])))
            print(f"GP model with kernel {n} had a score of {new_score}")
            if result.success and new_score < score:
                new_model = model
                score = new_score
                print("successful model:", result)
                print("model score:", score)
        print("final model:", new_model)
        model = new_model
    else:
        print("Check kernel specification\n")
        exit()

    opt = gpflow.optimizers.Scipy()
    result = opt.minimize(model.training_loss, model.trainable_variables)
    print(result)

    inps = np.random.rand(1000, np.shape(x_train)[-1])
    samps = model.posterior().predict_f(inps)

    with open("gp_model.pkl", "wb") as outp:
        pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
    outp.close()

    model.compiled_predict_f = tf.function(
        lambda Xnew: model.predict_f(Xnew, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[1, np.shape(x_train)[-1]], dtype=tf.float64)],
    )
    model.compiled_predict_y = tf.function(
        lambda Xnew: model.predict_y(Xnew, full_cov=False),
        input_signature=[tf.TensorSpec(shape=[1, np.shape(x_train)[-1]], dtype=tf.float64)],
    )

    tf.saved_model.save(model, "gp_model")

    outs = model.posterior().predict_f(x_validate)

    for i in range(n_basis):
        plt.figure(i)
        plt.title(f"Basis function {i}")
        plt.xlabel("Abaqus Output")
        plt.ylabel("GP Predicted Output")
        plt.plot(y_validate[:, i], outs[0][:, i], "o")
        plt.plot(np.array([0, 1]), np.array([0, 1]), "k")
        plt.legend()
        plt.savefig(f"Basis_func_{i}.png")

    return {"dependency_hack": "variable used to force dependency between tasks"}
