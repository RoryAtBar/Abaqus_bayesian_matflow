#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 11:38:17 2024

@author: w10944rb
"""

import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from sys import argv
import pickle
import matplotlib.pyplot as plt

normed_X_training_data = argv[1]
normed_Y_training_data = argv[2]
normed_X_validation_data = argv[3]
normed_Y_validation_data = argv[4]
X_normaliser_obj = argv[5]
Y_normaliser_obj = argv[6]
n_basis = int(argv[7])
kern = argv[8]
model_name = argv[9]
print("Xtraining data:", normed_X_training_data)
print("Ytraining data:", normed_Y_training_data)
print("Xvalidation:", normed_X_validation_data)
print("Yvalidation:", normed_Y_validation_data)
print("X normaliser:", X_normaliser_obj)
print("Y normaliser:", Y_normaliser_obj)
print("number of basis functions:", n_basis)
print("kernel:", kern)
print("model name:", model_name)


X_train = np.array(pd.read_pickle(normed_X_training_data))
Y_train = np.array(pd.read_pickle(normed_Y_training_data))

X_validate = np.array(pd.read_pickle(normed_X_validation_data))
Y_validate = np.array(pd.read_pickle(normed_Y_validation_data))

if n_basis != np.shape(Y_train)[-1]:
    print("basis function expectation doesn't match data\n")
    exit()


if kern == "RBF":
    model = gpflow.models.GPR(
        (X_train, Y_train),
        kernel=gpflow.kernels.RBF(
            np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
        )
        + gpflow.kernels.Linear(np.shape(X_train)[-1]),
    )
    print(model)
elif kern == "Matern52":
    model = gpflow.models.GPR(
        (X_train, Y_train),
        kernel=gpflow.kernels.Matern52(
            np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
        )
        + gpflow.kernels.Linear(np.shape(X_train)[-1]),
    )
    print(model)
elif kern == "Matern32":
    model = gpflow.models.GPR(
        (X_train, Y_train),
        kernel=gpflow.kernels.Matern32(
            np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
        )
        + gpflow.kernels.Linear(np.shape(X_train)[-1]),
    )
    print(model)
elif kern == "Matern12":
    model = gpflow.models.GPR(
        (X_train, Y_train),
        kernel=gpflow.kernels.Matern32(
            np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
        )
        + gpflow.kernels.Linear(np.shape(X_train)[-1]),
    )
    print(model)
elif kern == "Auto":
    score = 1e20
    kernels = [
        [
            gpflow.kernels.RBF(np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1]))
            + gpflow.kernels.Linear(np.shape(X_train)[-1])
        ],
        [
            gpflow.kernels.Matern52(
                np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
            )
            + gpflow.kernels.Linear(np.shape(X_train)[-1])
        ],
        [
            gpflow.kernels.Matern32(
                np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
            )
            + gpflow.kernels.Linear(np.shape(X_train)[-1])
        ],
        [
            gpflow.kernels.Matern12(
                np.shape(X_train)[-1], lengthscales=np.ones(np.shape(X_train)[-1])
            )
            + gpflow.kernels.Linear(np.shape(X_train)[-1])
        ],
    ]

    for n, k in enumerate(kernels):
        print(k[0])
        model = gpflow.models.GPR((X_train, Y_train), kernel=k[0])
        opt = gpflow.optimizers.Scipy()
        result = opt.minimize(model.training_loss, model.trainable_variables)
        outs = model.posterior().predict_f(X_validate)
        new_score = sum(sum((((outs[0] - Y_validate) ** 2) / outs[0])))
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

inps = np.random.rand(1000, np.shape(X_train)[-1])
samps = model.posterior().predict_f(inps)


with open(f"{model_name}.pkl", "wb") as outp:
    pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
outp.close()

model.compiled_predict_f = tf.function(
    lambda Xnew: model.predict_f(Xnew, full_cov=False),
    input_signature=[tf.TensorSpec(shape=[1, np.shape(X_train)[-1]], dtype=tf.float64)],
)
model.compiled_predict_y = tf.function(
    lambda Xnew: model.predict_y(Xnew, full_cov=False),
    input_signature=[tf.TensorSpec(shape=[1, np.shape(X_train)[-1]], dtype=tf.float64)],
)

tf.saved_model.save(model, model_name)

outs = model.posterior().predict_f(X_validate)

for i in range(n_basis):
    plt.figure(i)
    plt.title(f"Basis function {i}")
    plt.xlabel("Abaqus Output")
    plt.ylabel("GP Predicted Output")
    plt.plot(Y_validate[:, i], outs[0][:, i], "o")
    plt.plot(np.array([0, 1]), np.array([0, 1]), "k")
    plt.legend()
    plt.savefig(f"Basis_func_{i}.png")
