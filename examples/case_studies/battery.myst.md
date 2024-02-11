---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: pymc_env
  language: python
  name: python3
---

(battery)=
# Bayesian Decision Theory: should I buy a house battery?

:::{post} Feburary, 2024
:tags: decision making, case study, 
:category: beginner, reference
:author: Benjamin T. Vincent
:::

```{code-cell} ipython3
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
```

```{code-cell} ipython3
%config InlineBackend.figure_format = 'retina'  # high resolution figures
az.style.use("arviz-darkgrid")
rng = np.random.default_rng(42)
```

## Introduction

This case study will focus on the decision of whether to buy a house battery. This will be based on my own personal circumstances, partly because I am currently considering this decision, and partly because I have weekly data on my electricity usage and generation. 

We will use Bayesian Decision Theory to make the decision of whether to buy a house battery. A decision like this is complex and will involve many factors, but here we will only consider the financial aspects of the decision.

We are going to consider the following scenarios:
1. The status quo: We don't buy a house battery and continue to use electricity as we currently do.
2. Buy a house battery: We spend money upfront with the hope that our ongoing costs will be lower.

There are clearly many variations of the second scenario, such as the size of the battery, the way in which we use it and the choice of tariff. The aim here is not to consider all of these scenarios - we will pick a relatively simple scenario and focus on the process of making the decision.

The approach we will take is to project out the costs of each scenario over the next 10 years (see figure below). We can then estimate the time to break even and also get a sense of the rate at which savings will accrue if we buy a house battery.

![](battery_schematic.jpg)

Clearly, buying a house battery will carry upfront costs, but by estimating the time it takes to recoup these costs, we can make a more informed decision. Because energy use and generation is seasonal, the costs incurred under both scenarios will not be linear. But hopefully this schematic figure gives a good sense of how we will go about making the decision.

+++

### House energy consumption and generation
We use electricity for heating, cooking, lighting, and appliances. We currently have a gas combi boiler for heating and hot water, and so we don't use electricity for heating or hot water, via a heat pump for example. Currently we are on a flat rate tariff, where we pay the same amount for electricity (0.2816 p/kWr) at all times of day.

The house has 3 solar panels on the roof, which generate electricity during the day. Without a house battery, any energy created by the solar panels is either used by the house or exported to the grid. We are paid for any electricity exported to the grid, but this is at a relatively low rate. We currently get paid 0.15 p/kWhr for electricity exported to the grid. In the rest of this notebook we will refer to energy generated or exported as "PV" for photovoltaic.

+++

### Calculating costs

In order to calculate the costs of each scenario, we need to know the following:

$$
\text{cost} = (\text{import rate} \cdot \text{total import}) - (\text{export rate} \cdot \text{total export})
$$

where:

* $\text{import rate}$ = 0.2816 p/kWr.
* $\text{total import}$ will be given by the total energy demand minus the PV generation used by the house.
* $\text{export rate}$ = 0.15 p/kWr.
* $\text{total export}$ will be given by the PV generation minus the energy demand of the house.

In the UK we also have a daily standing charge. This is a fixed cost that will remain constant regardless of the strategy we take, so we ignore it in our calculations.

The costs calculation may get a little more complex later on in the notebook where we move away from a flat rate tarrif and consider a time of use tarrif with a cheaper night rate.

+++

### Why buy a house battery?
1. Higher utilisation of solar energy: Rather than exporting any excess electricity to the grid, we could store it in a house battery and use it later. While this would result in less money being made from exporting excess solar energy, it would also mean that we would use less electricity from the grid, which is more expensive.
2. Load shifting: We could use the house battery to store electricity from the grid overnight when it is cheap, and use it when it is more expensive. This would involve moving to a time-of-use tariff, where electricity is more expensive at certain times of day.


We have recently moved into a house which has solar panels on the roof. This means that we generate electricity during the day, and use it at night. We are considering buying a house battery to store the electricity generated during the day, and use it at night.

+++

## Load and process the data

```{code-cell} ipython3
try:
    df = pd.read_csv(os.path.join("..", "data", "energy_use.csv"), parse_dates=["date"])
except FileNotFoundError:
    df = pd.read_csv(pm.get_data("energy_use.csv"), parse_dates=["date"])

# calculate time sinse last reading
df["tdelta"] = df["date"].diff()
# calculate week of year
df["week"] = df["date"].dt.isocalendar().week
df.set_index("date", inplace=True)
df.head()
```

The raw data (columns) we have available are:

* `date`: the date on which the reading was taken.
* `grid_import`: the total energy imported from the grid.
* `grid_export`: the total energy exported to the grid.
* `pv_gen`: the total energy generated by the PV system.

We will need to calculate some quantities from these raw measurements to proceed. First, we'll calculate the amount of solar energy used by the house.

```{code-cell} ipython3
df["pv_used"] = df["pv_gen"] - df["grid_export"]
```

Because these are raw meter readings, let's reset these to be zero at the start of the data. This will make the calculations easier.

```{code-cell} ipython3
for col in ["grid_import", "grid_export", "pv_gen"]:
    df[col] = df[col] - df[col].iloc[0]
```

And we'll calculate the total energy demand of the house as the sum of the energy imported from the grid and the energy generated by the PV system that was used by the house.

```{code-cell} ipython3
df["total_demand"] = df["grid_import"] + df["pv_used"]
```

```{code-cell} ipython3
df.head()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
df[["grid_import", "grid_export", "pv_gen", "pv_used", "total_demand"]].plot(ax=ax)
ax.set_ylabel("Energy (kWh)")
```

```{code-cell} ipython3
IMPORT_RATE = 0.2816  # £/kWh
EXPORT_RATE = 0.15  # £/kWh
```

```{code-cell} ipython3
df["cost"] = IMPORT_RATE * df["total_demand"] - EXPORT_RATE * df["grid_export"]
```

```{code-cell} ipython3
fig, ax = plt.subplots()
df["cost"].plot(ax=ax)
ax.set_ylabel("Cost (£)");
```

So far we've imported our raw data, calculated a number important quantities from that, and calculated the financial costs of the current strategy. Importantly, this data is from the past and we have less than a year's worth of data.

+++

## Forecasting cost of the status quo into the future

Let's use our historical data to forecast both the energy demand and the PV generation into the future. We can then use these forecasts to calculate the costs of the status quo into the future.

+++

**TODO: normalise the data to get in units of kWhr/day.**

```{code-cell} ipython3
fig, ax = plt.subplots()

ax.step(df.index, df["total_demand"].diff(), label="Total energy demand")
ax.step(df.index, df["pv_gen"].diff(), label="PV generation")
plt.legend()
plt.xticks(rotation=45)
ax.set(title="Historical energy use and generation", ylabel="Energy (kWh)");
```

### Forecasting PV generation

We will use a simple parametric equation to fit the PV generation data. This will incorporate our knowledge of the physical system and allow us to forecast the PV generation into the future.

TODO: FIND PARAMETRIC EQUATION

```{code-cell} ipython3

```

### Forecasting the energy demand

```{code-cell} ipython3

```

### The forecasted costs

Let's put that together to generate a forecast of the costs of the status quo scenario into the future.

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## Forecasting cost of the house battery scenario into the future

+++

### The house battery scenario
As we've said, there are many different plausible scenarios here depending on the capacity of the battery in kWh, the way in which it is used, and the choice of tariff. In this section we will outline a simple but realistic scenario.

#### Initial costs

#### Battery capacity

10 kWh usable capacity

#### Round trip efficiency


#### Tariff


#### Battery use strategy

We will focus on load shifting where we charge the battery at the cheaper overnight rate with the aim of using that energy during the day. In an ideal day, we would be able to completely avoid importing from the grid during the regular day rate. Though this is likely to be unrealistic for two reasons. 

Firstly, the battery may not always be able to match the peak instantaneous demand of the house. For example, if we use our electric cooker, hob, have the TV and lighting as well as computer, we may exceed the power output of the battery. Although without historical data on the instantaneous power usage of the house on a miniute by minuite basis, it will be very hard to estimate how much grid import may happen at peak times.

Secondly, the battery may not always have sufficient charge to meet the daily demand of the house. This could happen if:
* the battery were not fully charged overnight for example
* if the total demand of the house was uncharachteristically high

```{code-cell} ipython3

```

```{code-cell} ipython3

```

## Estimating the time to break even

Now we can compare the costs of the two scenarios and estimate the time to break even. This is the time at which the costs of the two scenarios are equal. After this time, the house battery scenario will be cheaper.

```{code-cell} ipython3

```

## Authors
- Authored by [Benjamin T. Vincent](https://github.com/drbenvincent) in February 2024

+++

## References
:::{bibliography}
:filter: docname in docnames
:::

+++

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p pytensor
```

:::{include} ../page_footer.md
:::
