import numpy as np
import skfda
import pickle


def clean_profile_data(
    odb_results_table: dict,
    n_basis: int,
    plastic_start_step: int,
    plastic_end_step: int,
    train_to_validate: float,
):
    temp_profile = clean_data(odb_results_table["nt11"], 4, 44)
    peeq_profile = clean_data(odb_results_table["peeq_vals"], 3, -5)
    barelling_profile = clean_data(odb_results_table["barrelling_profile"], 3, -5)
    force_profile1 = clean_data(odb_results_table["force_vals1"], 3, -5, null_value=[0])
    force_profile2 = clean_data(odb_results_table["force_vals2"], 3, -5, null_value=[0])
    force_profile = [fp1[:, 1] + fp2[:, 1] for fp1, fp2 in zip(force_profile1, force_profile2)]

    # Convert force-displacement into stress and strain values
    true_strain = np.linspace(0, 0.475, 20)
    sample_volume = np.pi * (5e-3**2) * 10e-3
    sample_area = sample_volume / (np.exp(-true_strain) / 100)

    # Create a boolean vector containing all the simulations that successfully
    # completed
    truth_list = []
    for i in force_profile:
        truth_list.append(len(i) == 20)

    force_filtered = np.zeros((sum(truth_list), 20))
    count = 0
    for n, values in enumerate(force_profile):
        if truth_list[n]:
            force_filtered[count, :] = values
            count += 1

    true_stress_filtered = np.zeros(np.shape(force_filtered))
    for n, force in enumerate(force_filtered):
        true_stress_filtered[n, :] = force / sample_area

    basis = skfda.representation.basis.MonomialBasis(n_basis=n_basis)

    # filter out failed simulations
    valid_friction = [i for (i, v) in zip(odb_results_table["friction"], truth_list) if v]
    valid_conductance = [i for (i, v) in zip(odb_results_table["conductance"], truth_list) if v]
    valid_sims = np.array((valid_friction, np.array(valid_conductance))).T

    # Normalise the data, then conduct fPCA on the normalised data
    x = Normaliser(valid_sims)
    x_normalised = x.normalise(valid_sims)
    fixed_data = skfda.FDataGrid(
        true_stress_filtered[:, plastic_start_step:plastic_end_step],
        grid_points=true_strain[plastic_start_step:plastic_end_step],
    )
    basis_rep = fixed_data.to_basis(basis)

    y = Normaliser(basis_rep.coefficients)
    y_normalised = y.normalise(basis_rep.coefficients)

    # Separate the training and validation data
    data_randomiser = np.random.choice(
        np.shape(y_normalised)[0], np.shape(y_normalised)[0], replace=False
    )
    y_randomised = np.zeros(np.shape(y_normalised))
    x_randomised = np.zeros(np.shape(x_normalised))
    for n, i in enumerate(data_randomiser):
        y_randomised[n, :] = y_normalised[i, :]
        x_randomised[n, :] = x_normalised[i, :]
    cut_off = int(np.shape(y_randomised)[0] * train_to_validate)
    x_train = x_randomised[:cut_off, :]
    x_validate = x_randomised[cut_off:, :]
    y_train = y_randomised[:cut_off, :]
    y_validate = y_randomised[cut_off:, :]

    model_data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_validate": x_validate,
        "y_validate": y_validate,
    }

    # Save x and y as pickle files and load in next task.
    # MatFlow can't handle passing this data format as an output parameter.
    with open("x.pkl", "wb") as outp:
        pickle.dump(x, outp, pickle.HIGHEST_PROTOCOL)
    with open("y.pkl", "wb") as outp:
        pickle.dump(y, outp, pickle.HIGHEST_PROTOCOL)

    return model_data


def clean_data(raw_data: list[str], first_index: int, last_index: int, null_value=None):
    """Remove junk data points and convert json string data into numpy array"""
    # There are some lines of junk/empty data, which varies depending on the data variables
    # The lines of interest are specified by first_index and last_index
    if null_value is None:
        null_value = []
    clean_data = []
    if last_index < 0:
        last_index = len(raw_data[0]) + last_index
    raw_data_last_index = len(raw_data[0]) - 1
    n_junk_rows = raw_data_last_index - last_index + first_index
    for n, row in enumerate(raw_data):
        if not (isinstance(row, float)):
            prof = np.zeros((len(row) - n_junk_rows, 2))
            for k, string in enumerate(row[first_index : last_index + 1]):
                nums = string.split()
                prof[k, 0] = float(nums[0])
                prof[k, 1] = float(nums[1])
            clean_data.append(prof)
        else:
            clean_data.append(null_value)
    return clean_data


class Normaliser:
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
            raise ValueError("No data set")

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
