import gptools
import numpy as np


def gp_profile(
    data_X: np.ndarray,
    data_y: np.ndarray,
    err_y: np.ndarray,
    X_star: np.ndarray,
    calc_gradient: bool = False,
):
    hp = gptools.UniformJointPrior([[0.0, 20.0]]) * gptools.GammaJointPriorAlt(
        [1.0, 0.5, 0.0, 1.0], [0.3, 0.25, 0.1, 0.1]
    )
    k_gibbs = gptools.GibbsKernel1dTanh(hyperprior=hp)
    gp = gptools.GaussianProcess(k_gibbs)

    valid_mask = ~np.isnan(data_y) & ~np.isnan(data_X) & ~np.isnan(err_y)
    if np.sum(valid_mask) == 0:
        return None, None, None, None
    data_X = data_X[valid_mask]
    data_y = data_y[valid_mask]
    err_y = err_y[valid_mask]

    gp.add_data(data_X, data_y, err_y)
    gp.remove_outliers(sigma=2)

    # Boundary conditions, informed by Chilenski 2016
    val_bc = np.array([[1.1, 0, 0.01], [1.2, 0, 0.01], [1.3, 0, 0.01], [1.4, 0, 0.01]])
    grad_bc = np.array(
        [[0, 0, 0], [1.1, 0, 0.1], [1.2, 0, 0.1], [1.3, 0, 0.1], [1.4, 0, 0.1]]
    )
    gp.add_data(val_bc[:, 0], val_bc[:, 1], err_y=val_bc[:, 2], n=0)
    gp.add_data(grad_bc[:, 0], grad_bc[:, 1], err_y=grad_bc[:, 2], n=1)

    gp.optimize_hyperparameters(verbose=False, random_starts=8, max_tries=4, num_proc=4)
    y_star, std_y_star = gp.predict(X_star)

    if not calc_gradient:
        return y_star, std_y_star, None, None
    else:
        grad_y_star, std_grad_y_star = gp.predict(X_star, n=1)
        return y_star, std_y_star, grad_y_star, std_grad_y_star
