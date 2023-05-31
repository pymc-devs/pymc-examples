---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"id": "Pq7u0kdRwDje"}

(reinforcement_learning)=
# Fitting a Reinforcement Learning Model to Behavioral Data with PyMC

:::{post} Aug 5, 2022
:tags: PyTensor, Reinforcement Learning
:category: advanced, how-to
:author: Ricardo Vieira
:::


Reinforcement Learning models are commonly used in behavioral research to model how animals and humans learn, in situtions where they get to make repeated choices that are followed by some form of feedback, such as a reward or a punishment.

In this notebook we will consider the simplest learning scenario, where there are only two possible actions. When an action is taken, it is always followed by an immediate reward. Finally, the outcome of each action is independent from the previous actions taken. This scenario is sometimes referred to as the [multi-armed bandit problem](https://en.wikipedia.org/wiki/Multi-armed_bandit).


Let's say that the two actions (e.g., left and right buttons) are associated with a unit reward 40% and 60% of the time, respectively. At the beginning the learning agent does not know which action $a$ is better, so they may start by assuming both actions have a mean value of 50%. We can store these values in a table, which is usually referred to as a $Q$ table:

$$ Q = \begin{cases}
      .5, a = \text{left}\\
      .5, a = \text{right}
    \end{cases}
$$

When an action is chosen and a reward $r = \{0,1\}$ is observed, the estimated value of that action is updated as follows:

$$Q_{a} = Q_{a} + \alpha (r - Q_{a})$$

where $\alpha \in [0, 1]$ is a learning parameter that influences how much the value of an action is shifted towards the observed reward in each trial. Finally, the $Q$ table values are converted into action probabilities via the softmax transformation:

$$ P(a = \text{right}) = \frac{\exp(\beta Q_{\text{right}})}{\exp(\beta Q_{\text{right}}) + \exp(\beta Q_{\text{left}})}$$

where the $\beta \in (0, +\infty)$ parameter determines the level of noise in the agent choices. Larger values will be associated with more deterministic choices and smaller values with increasingly random choices.

```{code-cell} ipython3
:id: QTq-0HMw7dBK

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy

from matplotlib.lines import Line2D
```

```{code-cell} ipython3
seed = sum(map(ord, "RL_PyMC"))
rng = np.random.default_rng(seed)
az.style.use("arviz-darkgrid")
%config InlineBackend.figure_format = "retina"
```

+++ {"id": "aG_Nxvr5wC4B"}

## Generating fake data

```{code-cell} ipython3
:id: hcPVL7kZ8Zs2

def generate_data(rng, alpha, beta, n=100, p_r=None):
    if p_r is None:
        p_r = [0.4, 0.6]
    actions = np.zeros(n, dtype="int")
    rewards = np.zeros(n, dtype="int")
    Qs = np.zeros((n, 2))

    # Initialize Q table
    Q = np.array([0.5, 0.5])
    for i in range(n):
        # Apply the Softmax transformation
        exp_Q = np.exp(beta * Q)
        prob_a = exp_Q / np.sum(exp_Q)

        # Simulate choice and reward
        a = rng.choice([0, 1], p=prob_a)
        r = rng.random() < p_r[a]

        # Update Q table
        Q[a] = Q[a] + alpha * (r - Q[a])

        # Store values
        actions[i] = a
        rewards[i] = r
        Qs[i] = Q.copy()

    return actions, rewards, Qs
```

```{code-cell} ipython3
:id: ceNagbmsZXW6

true_alpha = 0.5
true_beta = 5
n = 150
actions, rewards, Qs = generate_data(rng, true_alpha, true_beta, n)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 208
id: MDhJI8vOXZeU
outputId: 60f7ee37-2d1f-44ad-afff-b9ba7d82a8d8
tags: [hide-input]
---
_, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(actions))

ax.plot(x, Qs[:, 0] - 0.5 + 0, c="C0", lw=3, alpha=0.3)
ax.plot(x, Qs[:, 1] - 0.5 + 1, c="C1", lw=3, alpha=0.3)

s = 7
lw = 2

cond = (actions == 0) & (rewards == 0)
ax.plot(x[cond], actions[cond], "o", ms=s, mfc="None", mec="C0", mew=lw)

cond = (actions == 0) & (rewards == 1)
ax.plot(x[cond], actions[cond], "o", ms=s, mfc="C0", mec="C0", mew=lw)

cond = (actions == 1) & (rewards == 0)
ax.plot(x[cond], actions[cond], "o", ms=s, mfc="None", mec="C1", mew=lw)

cond = (actions == 1) & (rewards == 1)
ax.plot(x[cond], actions[cond], "o", ms=s, mfc="C1", mec="C1", mew=lw)

ax.set_yticks([0, 1], ["left", "right"])
ax.set_ylim(-1, 2)
ax.set_ylabel("action")
ax.set_xlabel("trial")

reward_artist = Line2D([], [], c="k", ls="none", marker="o", ms=s, mew=lw, label="Reward")
no_reward_artist = Line2D(
    [], [], ls="none", marker="o", mfc="w", mec="k", ms=s, mew=lw, label="No reward"
)
Qvalue_artist = Line2D([], [], c="k", ls="-", lw=3, alpha=0.3, label="Qvalue (centered)")

ax.legend(handles=[no_reward_artist, Qvalue_artist, reward_artist], fontsize=12, loc=(1.01, 0.27));
```

+++ {"id": "6RNLAtqDXgG_"}

The plot above shows a simulated run of 150 trials, with parameters $\alpha = .5$ and $\beta = 5$, and constant reward probabilities of $.4$ and $.6$ for the left (blue) and right (orange) actions, respectively. 

Solid and empty dots indicate actions followed by rewards and no-rewards, respectively. The solid line shows the estimated $Q$ value for each action centered around the respective colored dots (the line is above its dots when the respective $Q$ value is above $.5$, and below otherwise). It can be seen that this value increases with rewards (solid dots) and decreases with non-rewards (empty dots). 

The change in line height following each outcome is directly related to the $\alpha$ parameter. The influence of the $\beta$ parameter is more difficult to grasp, but one way to think about it is that the higher its value, the more an agent will stick to the action that has the highest estimated value, even if the difference between the two is quite small. Conversely, as this value approaches zero, the agent will start picking randomly between the two actions, regardless of their estimated values.

+++ {"id": "LUTfha8Hc1ap"}

## Estimating the learning parameters via Maximum Likelihood

Having generated the data, the goal is to now 'invert the model' to estimate the learning parameters $\alpha$ and $\beta$. I start by doing it via Maximum Likelihood Estimation (MLE). This requires writing a custom function that computes the likelihood of the data given a potential $\alpha$ and $\beta$ and the fixed observed actions and rewards (actually the function computes the negative log likelihood, in order to avoid underflow issues).

I employ the handy scipy.optimize.minimize function, to quickly retrieve the values of $\alpha$ and $\beta$ that maximize the likelihood of the data (or actually, minimize the negative log likelihood).

This was also helpful when I later wrote the PyTensor function that computed the choice probabilities in PyMC. First, the underlying logic is the same, the only thing that changes is the syntax. Second, it provides a way to be confident that I did not mess up, and what I was actually computing was what I intended to.

```{code-cell} ipython3
:id: lWGlRE3BjR0E

def llik_td(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    alpha, beta = x
    actions, rewards = args

    # Initialize values
    Q = np.array([0.5, 0.5])
    logp_actions = np.zeros(len(actions))

    for t, (a, r) in enumerate(zip(actions, rewards)):
        # Apply the softmax transformation
        Q_ = Q * beta
        logp_action = Q_ - scipy.special.logsumexp(Q_)

        # Store the log probability of the observed action
        logp_actions[t] = logp_action[a]

        # Update the Q values for the next trial
        Q[a] = Q[a] + alpha * (r - Q[a])

    # Return the negative log likelihood of all observed actions
    return -np.sum(logp_actions[1:])
```

+++ {"id": "xXZgywFIgz6J"}

The function `llik_td` is strikingly similar to the `generate_data` one, except that instead of simulating an action and reward in each trial, it stores the log-probability of the observed action.

The function `scipy.special.logsumexp` is used to compute the term $\log(\exp(\beta Q_{\text{right}}) + \exp(\beta Q_{\text{left}}))$ in a way that is more numerically stable. 

In the end, the function returns the negative sum of all the log probabilities, which is equivalent to multiplying the probabilities in their original scale.

(The first action is ignored just to make the output comparable to the later PyTensor function. It doesn't actually change any estimation, as the initial probabilities are fixed and do not depend on either the $\alpha$ or $\beta$ parameters.)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
id: -E8B-rrBgy0j
outputId: 7c18b426-8d50-4706-f940-45ec716877f4
---
llik_td([true_alpha, true_beta], *(actions, rewards))
```

+++ {"id": "WT2UwuKWvRCq"}

Above, I computed the negative log likelihood of the data given the true $\alpha$ and $\beta$ parameters.

Below, I let scipy find the MLE values for the two parameters:

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 260
id: W1MOBxvw4Zl9
outputId: 39a73f7a-2362-4ef7-cc03-1e9aeda35ecf
---
x0 = [true_alpha, true_beta]
result = scipy.optimize.minimize(llik_td, x0, args=(actions, rewards), method="BFGS")
print(result)
print("")
print(f"MLE: alpha = {result.x[0]:.2f} (true value = {true_alpha})")
print(f"MLE: beta = {result.x[1]:.2f} (true value = {true_beta})")
```

+++ {"id": "y_cXP93QeVVM"}

The estimated MLE values are relatively close to the true ones. However, this procedure does not give any idea of the plausible uncertainty around these parameter values. To get that, I'll turn to PyMC for a bayesian posterior estimation.

But before that, I will implement a simple vectorization optimization to the log-likelihood function that will be more similar to the PyTensor counterpart. The reason for this is to speed up the slow bayesian inference engine down the road.

```{code-cell} ipython3
:id: 4knb5sKW9V66

def llik_td_vectorized(x, *args):
    # Extract the arguments as they are passed by scipy.optimize.minimize
    alpha, beta = x
    actions, rewards = args

    # Create a list with the Q values of each trial
    Qs = np.ones((n, 2), dtype="float64")
    Qs[0] = 0.5
    for t, (a, r) in enumerate(
        zip(actions[:-1], rewards[:-1])
    ):  # The last Q values were never used, so there is no need to compute them
        Qs[t + 1, a] = Qs[t, a] + alpha * (r - Qs[t, a])
        Qs[t + 1, 1 - a] = Qs[t, 1 - a]

    # Apply the softmax transformation in a vectorized way
    Qs_ = Qs * beta
    logp_actions = Qs_ - scipy.special.logsumexp(Qs_, axis=1)[:, None]

    # Return the logp_actions for the observed actions
    logp_actions = logp_actions[np.arange(len(actions)), actions]
    return -np.sum(logp_actions[1:])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
id: w9Z_Ik7AlBQC
outputId: 445a7838-29d0-4f21-bfd8-5b65606af286
---
llik_td_vectorized([true_alpha, true_beta], *(actions, rewards))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
id: bDPZJe7RqCZX
outputId: a90fbb47-ee9b-4390-87ff-f4b39ece8fca
---
%timeit llik_td([true_alpha, true_beta], *(actions, rewards))
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 34
id: Dvrqf878swBX
outputId: 94bf3268-0eab-4ce9-deb9-5d1527b3c19d
---
%timeit llik_td_vectorized([true_alpha, true_beta], *(actions, rewards))
```

+++ {"id": "YAs_zpPZyopT"}

The vectorized function gives the same results, but runs almost one order of magnitude faster. 

When implemented as an PyTensor function, the difference between the vectorized and standard versions was not this drastic. Still, it ran twice as fast, which meant the model also sampled at twice the speed it would otherwise have!

+++ {"id": "tC7xbCCIL7K4"}

## Estimating the learning parameters via PyMC

The most challenging part was to create an PyTensor function/loop to estimate the Q values when sampling our parameters with PyMC.

```{code-cell} ipython3
:id: u8L_FAB4hle1

def update_Q(action, reward, Qs, alpha):
    """
    This function updates the Q table according to the RL update rule.
    It will be called by pytensor.scan to do so recursevely, given the observed data and the alpha parameter
    This could have been replaced be the following lamba expression in the pytensor.scan fn argument:
        fn=lamba action, reward, Qs, alpha: pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
    """

    Qs = pt.set_subtensor(Qs[action], Qs[action] + alpha * (reward - Qs[action]))
    return Qs
```

```{code-cell} ipython3
:id: dHzhTy20g4vh

# Transform the variables into appropriate PyTensor objects
rewards_ = pt.as_tensor_variable(rewards, dtype="int32")
actions_ = pt.as_tensor_variable(actions, dtype="int32")

alpha = pt.scalar("alpha")
beta = pt.scalar("beta")

# Initialize the Q table
Qs = 0.5 * pt.ones((2,), dtype="float64")

# Compute the Q values for each trial
Qs, _ = pytensor.scan(
    fn=update_Q, sequences=[actions_, rewards_], outputs_info=[Qs], non_sequences=[alpha]
)

# Apply the softmax transformation
Qs = Qs * beta
logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

# Calculate the negative log likelihod of the observed actions
logp_actions = logp_actions[pt.arange(actions_.shape[0] - 1), actions_[1:]]
neg_loglike = -pt.sum(logp_actions)
```

+++ {"id": "C9Ayn6-kzhPN"}

Let's wrap it up in a function to test out if it's working as expected.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 89
id: g1hkTd75xxwo
outputId: a2310fd3-cac2-48c6-9d22-3c3b72410427
---
pytensor_llik_td = pytensor.function(
    inputs=[alpha, beta], outputs=neg_loglike, on_unused_input="ignore"
)
result = pytensor_llik_td(true_alpha, true_beta)
float(result)
```

+++ {"id": "AmcoU1CF5ix-"}

The same result is obtained, so we can be confident that the PyTensor loop is working as expected. We are now ready to implement the PyMC model.

```{code-cell} ipython3
:id: c70L4ZBT7QLr

def pytensor_llik_td(alpha, beta, actions, rewards):
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    # Compute the Qs values
    Qs = 0.5 * pt.ones((2,), dtype="float64")
    Qs, updates = pytensor.scan(
        fn=update_Q, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
    )

    # Apply the sotfmax transformation
    Qs = Qs[:-1] * beta
    logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

    # Calculate the log likelihood of the observed actions
    logp_actions = logp_actions[pt.arange(actions.shape[0] - 1), actions[1:]]
    return pt.sum(logp_actions)  # PyMC expects the standard log-likelihood
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 245
id: XQNBZLMvAdbo
outputId: 65d7a861-476c-4598-985c-e0b0fcd744c4
---
with pm.Model() as m:
    alpha = pm.Beta(name="alpha", alpha=1, beta=1)
    beta = pm.HalfNormal(name="beta", sigma=10)

    like = pm.Potential(name="like", var=pytensor_llik_td(alpha, beta, actions, rewards))

    tr = pm.sample(random_seed=rng)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 539
id: vgSumt-oATfN
outputId: eb3348a4-3092-48c8-d8b4-678af0173079
---
az.plot_trace(data=tr);
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 408
id: BL84iT_RAzEL
outputId: dcd4174b-4148-45cb-f72d-973f1487d8c2
---
az.plot_posterior(data=tr, ref_val=[true_alpha, true_beta]);
```

+++ {"id": "1FtAp76PBLCr"}

In this example, the obtained posteriors are nicely centered around the MLE values. What we have gained is an idea of the plausible uncertainty around these values.

### Alternative model using Bernoulli for the likelihood

In this last section I provide an alternative implementation of the model using a Bernoulli likelihood.

+++

:::{Note}
One reason why it's useful to use the Bernoulli likelihood is that one can then do prior and posterior predictive sampling as well as model comparison. With `pm.Potential` you cannot do it, because PyMC does not know what is likelihood and what is prior nor how to generate random draws. Neither of this is a problem when using a `pm.Bernoulli` likelihood.
:::

```{code-cell} ipython3
:id: pQdszDk_qYCX

def right_action_probs(alpha, beta, actions, rewards):
    rewards = pt.as_tensor_variable(rewards, dtype="int32")
    actions = pt.as_tensor_variable(actions, dtype="int32")

    # Compute the Qs values
    Qs = 0.5 * pt.ones((2,), dtype="float64")
    Qs, updates = pytensor.scan(
        fn=update_Q, sequences=[actions, rewards], outputs_info=[Qs], non_sequences=[alpha]
    )

    # Apply the sotfmax transformation
    Qs = Qs[:-1] * beta
    logp_actions = Qs - pt.logsumexp(Qs, axis=1, keepdims=True)

    # Return the probabilities for the right action, in the original scale
    return pt.exp(logp_actions[:, 1])
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 121
id: S55HgqZiTfpa
outputId: a2db2d68-8bf3-4773-8368-5b6dff310e4b
---
with pm.Model() as m_alt:
    alpha = pm.Beta(name="alpha", alpha=1, beta=1)
    beta = pm.HalfNormal(name="beta", sigma=10)

    action_probs = right_action_probs(alpha, beta, actions, rewards)
    like = pm.Bernoulli(name="like", p=action_probs, observed=actions[1:])

    tr_alt = pm.sample(random_seed=rng)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 452
id: zjXW103JiDRQ
outputId: aafc1b1e-082e-414b-cac7-0ad805097057
---
az.plot_trace(data=tr_alt);
```

```{code-cell} ipython3
:id: SDJN2w117eox

az.plot_posterior(data=tr_alt, ref_val=[true_alpha, true_beta]);
```

## Watermark

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p aeppl,xarray
```

## References

:::{bibliography}
:filter: docname in docnames
:::

+++

## Credits

* Authored by [Ricardo Vieira](https://github.com/ricardov94) in June 2022

  * Adapted PyMC code from Maria Eckstein ([GitHub](https://github.com/MariaEckstein/SLCN), [PyMC Discourse](https://discourse.pymc.io/t/modeling-reinforcement-learning-of-human-participant-using-pymc3/1735))

  * Adapted MLE code from Robert Wilson and Anne Collins {cite:p}`collinswilson2019` ([GitHub](https://github.com/AnneCollins/TenSimpleRulesModeling))

* Re-executed by [Juan Orduz](https://juanitorduz.github.io/) in August 2022 ([pymc-examples#410](https://github.com/pymc-devs/pymc-examples/pull/410))

+++

:::{include} ../page_footer.md
:::
