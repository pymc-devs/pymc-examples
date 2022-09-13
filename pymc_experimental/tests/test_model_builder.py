import numpy as np
import pandas as pd
import pymc as pm

from pymc_experimental.model_builder import ModelBuilder


class test_ModelBuilder(ModelBuilder):
    _model_type = "LinearModel"
    version = "0.1"

    def build_model(self, model_config, data=None):
        if data is not None:
            x = pm.MutableData("x", data["input"].values)
            y_data = pm.MutableData("y_data", data["output"].values)

        # prior parameters
        a_loc = model_config["a_loc"]
        a_scale = model_config["a_scale"]
        b_loc = model_config["b_loc"]
        b_scale = model_config["b_scale"]
        obs_error = model_config["obs_error"]

        # priors
        a = pm.Normal("a", a_loc, sigma=a_scale)
        b = pm.Normal("b", b_loc, sigma=b_scale)
        obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

        # observed data
        if data is not None:
            y_model = pm.Normal("y_model", a + b * x, obs_error, shape=x.shape, observed=y_data)

    def _data_setter(self, data: pd.DataFrame):
        with self.model:
            pm.set_data({"x": data["input"].values})
            if "output" in data.columns:
                pm.set_data({"y_data": data["output"].values})

    @classmethod
    def create_sample_input(cls):
        x = np.linspace(start=0, stop=1, num=100)
        y = 5 * x + 3
        y = y + np.random.normal(0, 1, len(x))
        data = pd.DataFrame({"input": x, "output": y})

        model_config = {
            "a_loc": 0,
            "a_scale": 10,
            "b_loc": 0,
            "b_scale": 10,
            "obs_error": 2,
        }

        sampler_config = {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

        return data, model_config, sampler_config


def test_fit():
    with pm.Model() as model:
        x = np.linspace(start=0, stop=1, num=100)
        y = 5 * x + 3
        x = pm.MutableData("x", x)
        y_data = pm.MutableData("y_data", y)

        a_loc = 7
        a_scale = 3
        b_loc = 5
        b_scale = 3
        obs_error = 2

        a = pm.Normal("a", a_loc, sigma=a_scale)
        b = pm.Normal("b", b_loc, sigma=b_scale)
        obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

        y_model = pm.Normal("y_model", a + b * x, obs_error, observed=y_data)

        idata = pm.sample(tune=100, draws=200, chains=1, cores=1, target_accept=0.5)
        idata.extend(pm.sample_prior_predictive())
        idata.extend(pm.sample_posterior_predictive(idata))

    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model_2 = test_ModelBuilder(model_config, sampler_config, data)
    model_2.idata = model_2.fit()
    assert str(model_2.idata.groups) == str(idata.groups)


def test_predict():
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    data, model_config, sampler_config = test_ModelBuilder.create_sample_input()
    model_2 = test_ModelBuilder(model_config, sampler_config, data)
    model_2.idata = model_2.fit()
    model_2.predict(prediction_data)
    with pm.Model() as model:
        x = np.linspace(start=0, stop=1, num=100)
        y = 5 * x + 3
        x = pm.MutableData("x", x)
        y_data = pm.MutableData("y_data", y)
        a_loc = 7
        a_scale = 3
        b_loc = 5
        b_scale = 3
        obs_error = 2

        a = pm.Normal("a", a_loc, sigma=a_scale)
        b = pm.Normal("b", b_loc, sigma=b_scale)
        obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

        y_model = pm.Normal("y_model", a + b * x, obs_error, observed=y_data)

        idata = pm.sample(tune=10, draws=20, chains=3, cores=1)
        idata.extend(pm.sample_prior_predictive())
        idata.extend(pm.sample_posterior_predictive(idata))
        y_test = pm.sample_posterior_predictive(idata)

        assert str(model_2.idata.groups) == str(idata.groups)
