---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc-ex
  language: python
  name: pymc-ex
---

+++ {"papermill": {"duration": 0.043172, "end_time": "2021-02-23T11:26:55.064791", "exception": false, "start_time": "2021-02-23T11:26:55.021619", "status": "completed"}, "tags": []}

(GLM-poisson-regression)=
# GLM: Poisson Regression

:::{post} November 30, 2022
:tags: regression, poisson
:category: intermediate
:author: Jonathan Sedar, Benjamin Vincent
:::

+++ {"papermill": {"duration": 0.069202, "end_time": "2021-02-23T11:27:01.489628", "exception": false, "start_time": "2021-02-23T11:27:01.420426", "status": "completed"}, "tags": []}

This is a minimal reproducible example of Poisson regression to predict counts using dummy data.

This Notebook is basically an excuse to demo Poisson regression using PyMC, both manually and using `bambi` to demo interactions using the `formulae` library. We will create some dummy data, Poisson distributed according to a linear model, and try to recover the coefficients of that linear model through inference.

For more statistical detail see:

+ Basic info on [Wikipedia](https://en.wikipedia.org/wiki/Poisson_regression)
+ GLMs: Poisson regression, exposure, and overdispersion in Chapter 6.2 of [ARM, Gelmann & Hill 2006](http://www.stat.columbia.edu/%7Egelman/arm/)
+ This worked example from ARM 6.2 by [Clay Ford](http://www.clayford.net/statistics/poisson-regression-ch-6-of-gelman-and-hill/)

This very basic model is inspired by [a project by Ian Osvald](http://ianozsvald.com/2016/05/07/statistically-solving-sneezes-and-sniffles-a-work-in-progress-report-at-pydatalondon-2016/), which is concerned with understanding the various effects of external environmental factors upon the allergic sneezing of a test subject.

```{code-cell} ipython3
#!pip install seaborn
```

```{code-cell} ipython3
---
papermill:
  duration: 6.051698
  end_time: '2021-02-23T11:27:01.160546'
  exception: false
  start_time: '2021-02-23T11:26:55.108848'
  status: completed
tags: []
---
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns

from formulae import design_matrices
```

```{code-cell} ipython3
---
papermill:
  duration: 0.111837
  end_time: '2021-02-23T11:27:01.349763'
  exception: false
  start_time: '2021-02-23T11:27:01.237926'
  status: completed
tags: []
---
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)

%config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")
```

+++ {"papermill": {"duration": 0.06268, "end_time": "2021-02-23T11:27:01.615645", "exception": false, "start_time": "2021-02-23T11:27:01.552965", "status": "completed"}, "tags": []}

## Local Functions

+++ {"papermill": {"duration": 0.073451, "end_time": "2021-02-23T11:27:01.763249", "exception": false, "start_time": "2021-02-23T11:27:01.689798", "status": "completed"}, "tags": []}

## Generate Data

+++ {"papermill": {"duration": 0.060542, "end_time": "2021-02-23T11:27:01.884617", "exception": false, "start_time": "2021-02-23T11:27:01.824075", "status": "completed"}, "tags": []}

This dummy dataset is created to emulate some data created as part of a study into quantified self, and the real data is more complicated than this. Ask Ian Osvald if you'd like to know more [@ianozvald](https://twitter.com/ianozsvald).


### Assumptions:

+ The subject sneezes N times per day, recorded as `nsneeze (int)`
+ The subject may or may not drink alcohol during that day, recorded as `alcohol (boolean)`
+ The subject may or may not take an antihistamine medication during that day, recorded as the negative action `nomeds (boolean)`
+ We postulate (probably incorrectly) that sneezing occurs at some baseline rate, which increases if an antihistamine is not taken, and further increased after alcohol is consumed.
+ The data is aggregated per day, to yield a total count of sneezes on that day, with a boolean flag for alcohol and antihistamine usage, with the big assumption that nsneezes have a direct causal relationship.


Create 4000 days of data: daily counts of sneezes which are Poisson distributed w.r.t alcohol consumption and antihistamine usage

```{code-cell} ipython3
---
papermill:
  duration: 0.07367
  end_time: '2021-02-23T11:27:02.023323'
  exception: false
  start_time: '2021-02-23T11:27:01.949653'
  status: completed
tags: []
---
# decide poisson theta values
theta_noalcohol_meds = 1  # no alcohol, took an antihist
theta_alcohol_meds = 3  # alcohol, took an antihist
theta_noalcohol_nomeds = 6  # no alcohol, no antihist
theta_alcohol_nomeds = 36  # alcohol, no antihist

# create samples
q = 1000
df = pd.DataFrame(
    {
        "nsneeze": np.concatenate(
            (
                rng.poisson(theta_noalcohol_meds, q),
                rng.poisson(theta_alcohol_meds, q),
                rng.poisson(theta_noalcohol_nomeds, q),
                rng.poisson(theta_alcohol_nomeds, q),
            )
        ),
        "alcohol": np.concatenate(
            (
                np.repeat(False, q),
                np.repeat(True, q),
                np.repeat(False, q),
                np.repeat(True, q),
            )
        ),
        "nomeds": np.concatenate(
            (
                np.repeat(False, q),
                np.repeat(False, q),
                np.repeat(True, q),
                np.repeat(True, q),
            )
        ),
    }
)
```

```{code-cell} ipython3
---
papermill:
  duration: 0.093062
  end_time: '2021-02-23T11:27:02.176348'
  exception: false
  start_time: '2021-02-23T11:27:02.083286'
  status: completed
tags: []
---
df.tail()
```

+++ {"papermill": {"duration": 0.071086, "end_time": "2021-02-23T11:27:02.312429", "exception": false, "start_time": "2021-02-23T11:27:02.241343", "status": "completed"}, "tags": []}

##### View means of the various combinations (Poisson mean values)

```{code-cell} ipython3
---
papermill:
  duration: 0.082117
  end_time: '2021-02-23T11:27:02.449759'
  exception: false
  start_time: '2021-02-23T11:27:02.367642'
  status: completed
tags: []
---
df.groupby(["alcohol", "nomeds"]).mean().unstack()
```

+++ {"papermill": {"duration": 0.054583, "end_time": "2021-02-23T11:27:02.561633", "exception": false, "start_time": "2021-02-23T11:27:02.507050", "status": "completed"}, "tags": []}

### Briefly Describe Dataset

```{code-cell} ipython3
---
papermill:
  duration: 2.510687
  end_time: '2021-02-23T11:27:05.124151'
  exception: false
  start_time: '2021-02-23T11:27:02.613464'
  status: completed
tags: []
---
g = sns.catplot(
    x="nsneeze",
    row="nomeds",
    col="alcohol",
    data=df,
    kind="count",
    height=4,
    aspect=1.5,
)
for ax in (g.axes[1, 0], g.axes[1, 1]):
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        label.set_visible(n % 5 == 0)
```

+++ {"papermill": {"duration": 0.049808, "end_time": "2021-02-23T11:27:05.231176", "exception": false, "start_time": "2021-02-23T11:27:05.181368", "status": "completed"}, "tags": []}

**Observe:**

+ This looks a lot like poisson-distributed count data (because it is)
+ With `nomeds == False` and `alcohol == False` (top-left, akak antihistamines WERE used, alcohol was NOT drunk) the mean of the poisson distribution of sneeze counts is low.
+ Changing `alcohol == True` (top-right) increases the sneeze count `nsneeze` slightly
+ Changing `nomeds == True` (lower-left) increases the sneeze count `nsneeze` further
+ Changing both `alcohol == True and nomeds == True` (lower-right) increases the sneeze count `nsneeze` a lot, increasing both the mean and variance.

+++ {"papermill": {"duration": 0.049476, "end_time": "2021-02-23T11:27:05.330914", "exception": false, "start_time": "2021-02-23T11:27:05.281438", "status": "completed"}, "tags": []}

---

+++ {"papermill": {"duration": 0.054536, "end_time": "2021-02-23T11:27:05.438038", "exception": false, "start_time": "2021-02-23T11:27:05.383502", "status": "completed"}, "tags": []}

## Poisson Regression

+++ {"papermill": {"duration": 0.048945, "end_time": "2021-02-23T11:27:05.540630", "exception": false, "start_time": "2021-02-23T11:27:05.491685", "status": "completed"}, "tags": []}

Our model here is a very simple Poisson regression, allowing for interaction of terms:

$$ \theta = exp(\beta X)$$

$$ Y_{sneeze\_count} \sim Poisson(\theta)$$

+++ {"papermill": {"duration": 0.04972, "end_time": "2021-02-23T11:27:05.641588", "exception": false, "start_time": "2021-02-23T11:27:05.591868", "status": "completed"}, "tags": []}

**Create linear model for interaction of terms**

```{code-cell} ipython3
---
papermill:
  duration: 0.056994
  end_time: '2021-02-23T11:27:05.748431'
  exception: false
  start_time: '2021-02-23T11:27:05.691437'
  status: completed
tags: []
---
fml = "nsneeze ~ alcohol + nomeds + alcohol:nomeds"  # full formulae formulation
```

```{code-cell} ipython3
---
papermill:
  duration: 0.058609
  end_time: '2021-02-23T11:27:05.859414'
  exception: false
  start_time: '2021-02-23T11:27:05.800805'
  status: completed
tags: []
---
fml = "nsneeze ~ alcohol * nomeds"  # lazy, alternative formulae formulation
```

+++ {"papermill": {"duration": 0.048682, "end_time": "2021-02-23T11:27:05.958802", "exception": false, "start_time": "2021-02-23T11:27:05.910120", "status": "completed"}, "tags": []}

### 1. Manual method, create design matrices and manually specify model

+++ {"papermill": {"duration": 0.049076, "end_time": "2021-02-23T11:27:06.059305", "exception": false, "start_time": "2021-02-23T11:27:06.010229", "status": "completed"}, "tags": []}

**Create Design Matrices**

```{code-cell} ipython3
dm = design_matrices(fml, df, na_action="error")
```

```{code-cell} ipython3
mx_ex = dm.common.as_dataframe()
mx_en = dm.response.as_dataframe()
```

```{code-cell} ipython3
mx_ex
```

+++ {"papermill": {"duration": 0.062897, "end_time": "2021-02-23T11:27:06.420853", "exception": false, "start_time": "2021-02-23T11:27:06.357956", "status": "completed"}, "tags": []}

**Create Model**

```{code-cell} ipython3
---
papermill:
  duration: 29.137887
  end_time: '2021-02-23T11:27:35.621305'
  exception: false
  start_time: '2021-02-23T11:27:06.483418'
  status: completed
tags: []
---
with pm.Model() as mdl_fish:

    # define priors, weakly informative Normal
    b0 = pm.Normal("Intercept", mu=0, sigma=10)
    b1 = pm.Normal("alcohol", mu=0, sigma=10)
    b2 = pm.Normal("nomeds", mu=0, sigma=10)
    b3 = pm.Normal("alcohol:nomeds", mu=0, sigma=10)

    # define linear model and exp link function
    theta = (
        b0
        + b1 * mx_ex["alcohol"].values
        + b2 * mx_ex["nomeds"].values
        + b3 * mx_ex["alcohol:nomeds"].values
    )

    ## Define Poisson likelihood
    y = pm.Poisson("y", mu=pm.math.exp(theta), observed=mx_en["nsneeze"].values)
```

+++ {"papermill": {"duration": 0.049445, "end_time": "2021-02-23T11:27:35.720870", "exception": false, "start_time": "2021-02-23T11:27:35.671425", "status": "completed"}, "tags": []}

**Sample Model**

```{code-cell} ipython3
---
papermill:
  duration: 108.169723
  end_time: '2021-02-23T11:29:23.939578'
  exception: false
  start_time: '2021-02-23T11:27:35.769855'
  status: completed
tags: []
---
with mdl_fish:
    inf_fish = pm.sample()
    # inf_fish.extend(pm.sample_posterior_predictive(inf_fish))
```

+++ {"papermill": {"duration": 0.118023, "end_time": "2021-02-23T11:29:24.142987", "exception": false, "start_time": "2021-02-23T11:29:24.024964", "status": "completed"}, "tags": []}

**View Diagnostics**

```{code-cell} ipython3
---
papermill:
  duration: 4.374731
  end_time: '2021-02-23T11:29:28.617406'
  exception: false
  start_time: '2021-02-23T11:29:24.242675'
  status: completed
tags: []
---
az.plot_trace(inf_fish);
```

+++ {"papermill": {"duration": 0.076462, "end_time": "2021-02-23T11:29:28.790410", "exception": false, "start_time": "2021-02-23T11:29:28.713948", "status": "completed"}, "tags": []}

**Observe:**

+ The model converges quickly and traceplots looks pretty well mixed

+++ {"papermill": {"duration": 0.07685, "end_time": "2021-02-23T11:29:28.943674", "exception": false, "start_time": "2021-02-23T11:29:28.866824", "status": "completed"}, "tags": []}

### Transform coeffs and recover theta values

```{code-cell} ipython3
az.summary(np.exp(inf_fish.posterior), kind="stats")
```

+++ {"papermill": {"duration": 0.075014, "end_time": "2021-02-23T11:29:29.324266", "exception": false, "start_time": "2021-02-23T11:29:29.249252", "status": "completed"}, "tags": []}

**Observe:**

+ The contributions from each feature as a multiplier of the baseline sneezecount appear to be as per the data generation:
    
    
    1. exp(Intercept): mean=1.05  cr=[0.98, 1.10]        
    
        Roughly linear baseline count when no alcohol and meds, as per the generated data: 

        theta_noalcohol_meds = 1 (as set above)
        theta_noalcohol_meds = exp(Intercept) 
                             = 1


    2. exp(alcohol): mean=2.86  cr=[2.67, 3.07]
    
        non-zero positive effect of adding alcohol, a ~3x multiplier of 
        baseline sneeze count, as per the generated data: 

        theta_alcohol_meds = 3 (as set above)
        theta_alcohol_meds = exp(Intercept + alcohol) 
                           = exp(Intercept) * exp(alcohol) 
                           = 1 * 3 = 3


    3. exp(nomeds): mean=5.73  cr=[5.34, 6.08]    
    
        larger, non-zero positive effect of adding nomeds, a ~6x multiplier of 
        baseline sneeze count, as per the generated data: 

        theta_noalcohol_nomeds = 6 (as set above)
        theta_noalcohol_nomeds = exp(Intercept + nomeds)
                               = exp(Intercept) * exp(nomeds) 
                               = 1 * 6 = 6
    
    
    4. exp(alcohol:nomeds): mean=2.10  cr=[1.96, 2.28]
    
        small, positive interaction effect of alcohol and meds, a ~2x multiplier of 
        baseline sneeze count, as per the generated data: 

        theta_alcohol_nomeds = 36 (as set above)
        theta_alcohol_nomeds = exp(Intercept + alcohol + nomeds + alcohol:nomeds)
                             = exp(Intercept) * exp(alcohol) * exp(nomeds * alcohol:nomeds)
                             = 1 * 3 * 6 * 2 = 36

+++ {"papermill": {"duration": 0.076829, "end_time": "2021-02-23T11:29:29.477240", "exception": false, "start_time": "2021-02-23T11:29:29.400411", "status": "completed"}, "tags": []}

### 2. Alternative method, using `bambi`

+++ {"papermill": {"duration": 0.074408, "end_time": "2021-02-23T11:29:29.628052", "exception": false, "start_time": "2021-02-23T11:29:29.553644", "status": "completed"}, "tags": []}

**Create Model**

+++ {"papermill": {"duration": 0.07467, "end_time": "2021-02-23T11:29:29.778406", "exception": false, "start_time": "2021-02-23T11:29:29.703736", "status": "completed"}, "tags": []}

**Alternative automatic formulation using `bambi`**

```{code-cell} ipython3
---
papermill:
  duration: 4.699873
  end_time: '2021-02-23T11:29:34.554521'
  exception: false
  start_time: '2021-02-23T11:29:29.854648'
  status: completed
tags: []
---
model = bmb.Model(fml, df, family="poisson")
```

+++ {"papermill": {"duration": 0.077285, "end_time": "2021-02-23T11:29:34.719403", "exception": false, "start_time": "2021-02-23T11:29:34.642118", "status": "completed"}, "tags": []}

**Fit Model**

```{code-cell} ipython3
---
papermill:
  duration: 115.426671
  end_time: '2021-02-23T11:31:30.222773'
  exception: false
  start_time: '2021-02-23T11:29:34.796102'
  status: completed
tags: []
---
inf_fish_alt = model.fit()
```

+++ {"papermill": {"duration": 0.075564, "end_time": "2021-02-23T11:31:30.375433", "exception": false, "start_time": "2021-02-23T11:31:30.299869", "status": "completed"}, "tags": []}

**View Traces**

```{code-cell} ipython3
---
papermill:
  duration: 2.970961
  end_time: '2021-02-23T11:31:33.424138'
  exception: false
  start_time: '2021-02-23T11:31:30.453177'
  status: completed
tags: []
---
az.plot_trace(inf_fish_alt);
```

+++ {"papermill": {"duration": 0.10274, "end_time": "2021-02-23T11:31:33.628707", "exception": false, "start_time": "2021-02-23T11:31:33.525967", "status": "completed"}, "tags": []}

### Transform coeffs

```{code-cell} ipython3
az.summary(np.exp(inf_fish_alt.posterior), kind="stats")
```

+++ {"papermill": {"duration": 0.10059, "end_time": "2021-02-23T11:31:34.095731", "exception": false, "start_time": "2021-02-23T11:31:33.995141", "status": "completed"}, "tags": []}

**Observe:**

+ The traceplots look well mixed
+ The transformed model coeffs look moreorless the same as those generated by the manual model
+ Note that the posterior predictive samples have an extreme skew

```{code-cell} ipython3
:tags: []

posterior_predictive = model.predict(inf_fish_alt, kind="pps")
```

We can use `az.plot_ppc()` to check that the posterior predictive samples are similar to the observed data.

For more information on posterior predictive checks, we can refer to {ref}`pymc:posterior_predictive`.

```{code-cell} ipython3
az.plot_ppc(inf_fish_alt);
```

+++ {"papermill": {"duration": 0.106366, "end_time": "2021-02-23T11:31:34.956844", "exception": false, "start_time": "2021-02-23T11:31:34.850478", "status": "completed"}, "tags": []}

## Authors
- Example originally contributed by [Jonathan Sedar](https://github.com/jonsedar) 2016-05-15.
- Updated to PyMC v4 by [Benjamin Vincent](https://github.com/drbenvincent) May 2022.
- Notebook header and footer updated November 2022.

+++

## Watermark

```{code-cell} ipython3
---
papermill:
  duration: 0.16014
  end_time: '2021-02-23T11:31:43.372227'
  exception: false
  start_time: '2021-02-23T11:31:43.212087'
  status: completed
tags: []
---
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor,aeppl
```

:::{include} ../page_footer.md
:::
