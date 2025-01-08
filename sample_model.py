import pymc as pm
import numpy as np
import skfda
from glob import glob
import pickle
import arviz as az
import pytensor.tensor as pt
from pathlib import Path


def sample_model(
    x_pickle: str,
    y_pickle: str,
    n_basis: int,
    plastic_start_step: int,
    plastic_end_step: int,
    burn_in: int,
    sample_size: int,
    sythetic_mean_basis_coeffs: list,
    variance_dir: str,
    dependency_hack: str,
):

    # The Bayesian Inference is uncertainty propagated backwards from either experiental or
    # sythetic data. The test curve represents the basis functions from skfda that represent the mean
    # basis function coefficients of the curve that will be used to calculate likelihood function values
    test_curve = np.array(sythetic_mean_basis_coeffs)

    print("test curve coeffs:", test_curve, "Array length:", len(test_curve))

    if n_basis != len(test_curve):
        # The reason this is here rather than simply setting Synthetic_mean_basis_coeffs
        # to the number of coefficient arguments is to help avoid user errors. If the basis coefficients are not
        # read correctly and there are not the expected number of coefficients, the script ends here.
        print("Incorrect number of basis functions")
        print(f"Stated number of functions is {n_basis}, but {len(test_curve)} provided")
        exit()

    # import the already created GP model saved as a pickle. It is important to run this script in the same
    # environment as that used to create the saved model because this method of saving objects is very sensitive to
    # module versions.
    gp_model_path = (
        Path.cwd().parents[3] / "artifacts/tasks/task_3_run_model/e_0/r_0/gp_model.pkl"
    )
    with open(gp_model_path, "rb") as mod:
        model = pickle.load(mod)

    # retrieve data normalisers
    with open(x_pickle, "rb") as unpick:
        x = pickle.load(unpick)

    with open(y_pickle, "rb") as unpick:
        y = pickle.load(unpick)

    # Imports actual stress strain curves to generate a covariance matrix representing the uncertainty
    variance_dat_list = glob(f"{variance_dir}/*.csv")

    print("Curves found:", variance_dat_list)

    variance_dat_dict = {
        csv_file: np.loadtxt(csv_file, delimiter=",", skiprows=3) for csv_file in variance_dat_list
    }
    keys = list(variance_dat_dict.keys())

    # Check data is as expected
    print("Dict of variance data keys:", variance_dat_dict.keys())
    print("Dict of variance data:", variance_dat_dict)
    print("Example data:", variance_dat_dict[keys[-1]])

    # The true_strain numpy array reflects the strain at each timestep in the Abaqus model. Some of the timesteps
    # have been cropped so that the Gaussian Process can be fit only to the plastic deformation behaviour.
    # This section finds the basis function coefficients that fit to the experimetnal data in the same plastic strain range
    # as the cropped Abaqus output data used to calculate the functional PCA that trained the GP.
    # The basis functions are found, and the covariance matrix calculated. The covariance matrix is used in the Bayesian inference.
    start = plastic_start_step * 0.025
    true_strain = np.linspace(0, 0.475, 20)
    stresses_for_fPCA = []
    strains_for_fPCA = []
    for key in keys:
        stress, strain, load, displacement = get_stress_strain(variance_dat_dict[key])
        strain_filter = [s > start and s < true_strain[plastic_end_step] for s in strain]
        stresses_for_fPCA.append(stress[strain_filter] * 1e6)
        strains_for_fPCA.append(strain[strain_filter])

    basis = skfda.representation.basis.MonomialBasis(n_basis=n_basis)
    basis_rep_vals = []
    for n, i in enumerate(stresses_for_fPCA):
        fixed_data_i = skfda.FDataGrid(i, grid_points=strains_for_fPCA[n])
        basis_rep_i = fixed_data_i.to_basis(basis)
        basis_rep_vals.append(basis_rep_i.coefficients[0])

    cov_mat = np.cov(np.array(basis_rep_vals).T)

    print("Covariance matrix: ", cov_mat)

    sigma = cov_mat
    logl = LogLikeWithGrad(loglike_combined, test_curve, sigma, model.posterior(), x, y)
    chain_no = 1

    # Uniform distributions are being used as priors as they restrict the MCMC samples to the range of parameters
    # Used to train the gaussian process surrogate model
    minimums = x.recover(np.zeros(2))
    maximums = x.recover(np.ones(2))

    # Check the maximum and minimum values are in line with the data
    print("About to start MCMC")
    print("Maximums:", maximums)
    print("Minimiums:", minimums)

    # PyMC context manager for the model
    with pm.Model():
        # uniform priors on f and c
        f = pm.Uniform("Friction", lower=minimums[0], upper=maximums[0])
        c = pm.Uniform("Conductance", lower=minimums[1], upper=maximums[1])

        theta = pt.as_tensor_variable([f, c])
        pm.Potential("likelihood", logl(theta))

        # sigma = pm.Uniform("sigma", lower=0,upper=1)

        # use a Normal distribution
        # step=pm.NUTS(step_scale=0.005, adapt_step_size=False)
        # step=pm.Metropolis(step_scale=0.01, adapt_step_size=False)
        step = pm.Metropolis()

        idata = pm.sample(tune=burn_in, draws=sample_size, step=step, cores=1, chains=chain_no)

    # Save the chain as a netcdf
    az.to_netcdf(idata, f"Idata_chain.nc")


# Variance values are needed to perform the Bayesian inference. This is based on experimental data.
# This function extracts the raw stress-strain curves.
def get_stress_strain(data, interval=50, offset=200):
    points = data[:, 0]
    force = data[:, 4]
    strain = data[:, 5]
    TrueStress = data[:, 6]
    TrueStrain = data[:, 7]
    deltaL = data[:, 3]

    #     grad_force=(np.gradient(force[::interval]))
    #     test_start=np.where(grad_force==np.max(grad_force))
    #     test_end=np.where(grad_force==np.min(grad_force))
    #     point_start=int(points[::interval][test_start])-offset
    #     point_end=int(points[::interval][test_end])
    #     stress= TrueStress[point_start:point_end]
    #     strain= TrueStrain[point_start:point_end]

    stress = TrueStress
    strain = TrueStrain
    load = force
    displacement = deltaL

    return stress, strain, load, displacement


# set up the pymc model and functions


# This class is from the PyMC website and is to wrap the samples for the MCMC since PyMC is designed to used tensor variables,
# and the GP takes a numpy array as input and output.
# Further classes are to calculate the gradient in posterior space using central difference.
# This may allow for more efficient sampling during the MCMC step.
class LogLikeWithGrad(pt.Op):
    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    # my_loglike(theta,  flow_curve_coeffs, cov_mat_full_curve,gp_model,strain_vals,X_norm_obj, Y_norm_obj,final_step_kernel)
    def __init__(self, loglike, curve_coeffs, sigma, GPmodel, X_norm_obj, Y_norm_obj):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.curve_coeffs = curve_coeffs

        self.sigma = sigma
        self.GPmodel = GPmodel

        self.X_norm_obj = X_norm_obj
        self.Y_norm_obj = Y_norm_obj

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.curve_coeffs, self.sigma, self.GPmodel)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(
            theta, self.curve_coeffs, self.sigma, self.GPmodel, self.X_norm_obj, self.Y_norm_obj
        )

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(pt.Op):
    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, curve_coeffs, sigma, GPmodel):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.curve_coeffs = curve_coeffs

        self.sigma = sigma
        self.GPmodel = GPmodel

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = normal_gradients(theta, self.curve_coeffs, self.sigma, self.GPmodel)

        outputs[0][0] = grads


def normal_gradients(theta, curve_coeffs, sigma, GPmodel):
    """
    Calculate the partial derivatives of a function at a set of values. The
    derivatives are calculated using the central difference, using an iterative
    method to check that the values converge as step size decreases.

    Parameters
    ----------
    theta: array_like
        A set of values, that are passed to a function, at which to calculate
        the gradient of that function
    x, data, sigma:
        Observed variables as we have been using so far


    Returns
    -------
    grads: array_like
        An array of gradients for each non-fixed value.
    """

    grads = np.empty(2)
    aux_vect = data - my_model(theta, GPmodel)  # /(2*sigma**2)
    grads[0] = np.sum(aux_vect * np.linspace(1, len(aux_vect), len(aux_vect) + 1))
    grads[1] = np.sum(aux_vect)

    return grads


def my_model(theta, gp_model, Y_norm_obj):
    predicted_output = gp_model.predict_f(np.array([theta]))
    return predicted_output


def loglike_combined(theta, data, sigma, gp_model, X_norm_obj, Y_norm_obj):
    theta_inp = X_norm_obj.normalise(theta)
    # print(theta_inp)
    # print(sigma, type(sigma),np.shape(sigma))
    # inv_cov = np.linalg.inv(sigma)
    normed_out, normed_var = my_model(theta_inp, gp_model, Y_norm_obj)
    # print("normalised out:", normalised_out)
    # print("data",data)
    new_covar = sigma + np.eye(len(normed_var[0])) * Y_norm_obj.recover(normed_var[0])
    princo = len(theta)
    inv_cov = np.linalg.inv(new_covar[:princo, :princo])
    p_model = Y_norm_obj.recover(normed_out[0])
    # print("p",p_model)
    diff = np.array((data - p_model))
    # print("diff: ",diff)
    return (-0.5) * (np.matmul(np.matmul(diff[:princo], inv_cov), diff[:princo, None]))[0]


# The normaliser class was used to put all of the data into values between 0 and 1. The normalised data
# Was used to train the GP surrogate model, so the normaliser is re-implemented here so that the data
# stored in the previous nomaliser object may simply be saved in a pickle file in the previous step and recovered
class Normaliser():
    def __init__(self, data=None):
        self.min = None
        self.max = None
        if data is not None:
            self.add_data(data)

    @property
    def ready(self):
        return self.min is not None and self.max is not None

    @property
    def scale(self):
        self.check_ready()
        return np.abs(self.max - self.min)

    @property
    def offset(self):
        self.check_ready()
        return self.min

    def check_ready(self):
        if not self.ready:
            raise ValueError('No data set')

    def add_data(self, data):
        new_min = data.min(axis=0)
        new_max = data.max(axis=0)
        if self.ready:
            mask = new_min < self.min
            self.min[mask] = new_min[mask]

            mask = new_max > self.max
            self.max[mask] = new_max[mask]
        else:
            self.min = new_min
            self.max = new_max

    def normalise(self, data, i=None):
        self.check_ready()
        if i is None:
            return (data - self.offset) / self.scale
        else:
            assert isinstance(i, int)
        
    def recover(self, data, i=None):
        self.check_ready()
        if i is None:
            return data * self.scale + self.offset
        else:
            assert isinstance(i, int)
            return data * self.scale[i] + self.offset[i]

