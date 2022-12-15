import warnings
from typing import Sequence, Tuple, Union

import numpy as np
import pytensor.tensor as pt
from pymc import SymbolicRandomVariable
from pymc.distributions.discrete import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.transforms import Chain
from pymc.logprob.abstract import _get_measurable_outputs, _logprob
from pymc.logprob.joint_logprob import factorized_joint_logprob
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.pytensorf import constant_fold, inputvars
from pytensor import Mode
from pytensor.compile import SharedVariable
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import Constant, FunctionGraph, ancestors, clone_replace
from pytensor.scan import map as scan_map
from pytensor.tensor import TensorVariable
from pytensor.tensor.elemwise import Elemwise

__all__ = ["MarginalModel"]


class MarginalModel(Model):
    """Subclass of PyMC Model that implements functionality for automatic
    marginalization of variables in the logp transformation

    After defining the full Model, the `marginalize` method can be used to indicate a
    subset of variables that should be marginalized

    Notes
    -----
    Marginalization functionality is still very restricted. Only finite discrete
    variables can be marginalized. Deterministics and Potentials cannot be conditionally
    dependent on the marginalized variables.

    Furthermore, not all instances of such variables can be marginalized. If a variable
    has batched dimensions, it is required that any conditionally dependent variables
    use information from an individual batched dimension. In other words, the graph
    connecting the marginalized variable(s) to the dependent variable(s) must be
    composed strictly of Elemwise Operations. This is necessary to ensure an efficient
    logprob graph can be generated. If you want to bypass this restriction you can
    separate each dimension of the marginalized variable into the scalar components
    and then stack them together. Note that such graphs will grow exponentially in the
    number of  marginalized variables.

    For the same reason, it's not possible to marginalize RVs with multivariate
    dependent RVs.

    Examples
    --------

    Marginalize over a single variable

    .. code-block:: python
        import pymc as pm
        from pymc_experimental import MarginalModel

        with MarginalModel() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

            m.marginalize([x])

            idata = pm.sample()

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.marginalized_rvs = []

    def _delete_rv_mappings(self, rv: TensorVariable) -> None:
        """Remove all model mappings referring to rv

        This can be used to "delete" an RV from a model
        """
        assert rv in self.basic_RVs, "rv is not part of the Model"

        name = rv.name
        self.named_vars.pop(name)
        if name in self.named_vars_to_dims:
            self.named_vars_to_dims.pop(name)

        value = self.rvs_to_values.pop(rv)
        self.values_to_rvs.pop(value)

        self.rvs_to_transforms.pop(rv)
        if rv in self.free_RVs:
            self.free_RVs.remove(rv)
            self.rvs_to_initial_values.pop(rv)
        else:
            self.observed_RVs.remove(rv)
        if rv in self.rvs_to_total_sizes:
            self.rvs_to_total_sizes.pop(rv)

    def _transfer_rv_mappings(self, old_rv: TensorVariable, new_rv: TensorVariable) -> None:
        """Transfer model mappings from old_rv to new_rv"""

        assert old_rv in self.basic_RVs, "old_rv is not part of the Model"
        assert new_rv not in self.basic_RVs, "new_rv is already part of the Model"

        self.named_vars.pop(old_rv.name)
        new_rv.name = old_rv.name
        self.named_vars[new_rv.name] = new_rv
        if old_rv in self.named_vars_to_dims:
            self._RV_dims[new_rv] = self._RV_dims.pop(old_rv)

        value = self.rvs_to_values.pop(old_rv)
        self.rvs_to_values[new_rv] = value
        self.values_to_rvs[value] = new_rv

        self.rvs_to_transforms[new_rv] = self.rvs_to_transforms.pop(old_rv)
        if old_rv in self.free_RVs:
            index = self.free_RVs.index(old_rv)
            self.free_RVs.pop(index)
            self.free_RVs.insert(index, new_rv)
            self.rvs_to_initial_values[new_rv] = self.rvs_to_initial_values.pop(old_rv)
        elif old_rv in self.observed_RVs:
            index = self.observed_RVs.index(old_rv)
            self.observed_RVs.pop(index)
            self.observed_RVs.insert(index, new_rv)
            if old_rv in self.rvs_to_total_sizes:
                self.rvs_to_total_sizes[new_rv] = self.rvs_to_total_sizes.pop(old_rv)

    def _marginalize(self, user_warnings=False):
        fg = FunctionGraph(outputs=self.basic_RVs + self.marginalized_rvs, clone=False)

        toposort = fg.toposort()
        rvs_left_to_marginalize = self.marginalized_rvs
        for rv_to_marginalize in sorted(
            self.marginalized_rvs,
            key=lambda rv: toposort.index(rv.owner),
            reverse=True,
        ):
            # Check that no deterministics or potentials dependend on the rv to marginalize
            for det in self.deterministics:
                if is_conditional_dependent(
                    det, rv_to_marginalize, self.basic_RVs + rvs_left_to_marginalize
                ):
                    raise NotImplementedError(
                        f"Cannot marginalize {rv_to_marginalize} due to dependent Deterministic {det}"
                    )
            for pot in self.potentials:
                if is_conditional_dependent(
                    pot, rv_to_marginalize, self.basic_RVs + rvs_left_to_marginalize
                ):
                    raise NotImplementedError(
                        f"Cannot marginalize {rv_to_marginalize} due to dependent Potential {pot}"
                    )

            old_rvs, new_rvs = replace_finite_discrete_marginal_subgraph(
                fg, rv_to_marginalize, self.basic_RVs + rvs_left_to_marginalize
            )

            if user_warnings and len(new_rvs) > 2:
                warnings.warn(
                    "There are multiple dependent variables in a FiniteDiscreteMarginalRV. "
                    f"Their joint logp terms will be assigned to the first RV: {old_rvs[1]}",
                    UserWarning,
                )

            rvs_left_to_marginalize.remove(rv_to_marginalize)

            for old_rv, new_rv in zip(old_rvs, new_rvs):
                new_rv.name = old_rv.name
                if old_rv in self.marginalized_rvs:
                    idx = self.marginalized_rvs.index(old_rv)
                    self.marginalized_rvs.pop(idx)
                    self.marginalized_rvs.insert(idx, new_rv)
                if old_rv in self.basic_RVs:
                    self._transfer_rv_mappings(old_rv, new_rv)
                    if user_warnings:
                        # Interval transforms for dependent variable won't work for non-constant bounds because
                        # the RV inputs are now different and may depend on another RV that also depends on the
                        # same marginalized RV
                        transform = self.rvs_to_transforms[new_rv]
                        if isinstance(transform, IntervalTransform) or (
                            isinstance(transform, Chain)
                            and any(
                                isinstance(tr, IntervalTransform) for tr in transform.transform_list
                            )
                        ):
                            warnings.warn(
                                f"The transform {transform} for the variable {old_rv}, which depends on the "
                                f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                                UserWarning,
                            )
        return self

    def _logp(self, *args, **kwargs):
        return super().logp(*args, **kwargs)

    def logp(self, vars=None, **kwargs):
        m = self.clone()._marginalize()
        if vars is not None:
            if not isinstance(vars, Sequence):
                vars = (vars,)
            vars = [m[var.name] for var in vars]
        return m._logp(vars=vars, **kwargs)

    def clone(self):
        m = MarginalModel()
        vars = self.basic_RVs + self.potentials + self.deterministics + self.marginalized_rvs
        cloned_vars = clone_replace(vars)
        vars_to_clone = {var: cloned_var for var, cloned_var in zip(vars, cloned_vars)}

        m.named_vars = {name: vars_to_clone[var] for name, var in self.named_vars.items()}
        m.named_vars_to_dims = self.named_vars_to_dims
        m.values_to_rvs = {i: vars_to_clone[rv] for i, rv in self.values_to_rvs.items()}
        m.rvs_to_values = {vars_to_clone[rv]: i for rv, i in self.rvs_to_values.items()}
        m.rvs_to_transforms = {vars_to_clone[rv]: i for rv, i in self.rvs_to_transforms.items()}
        # Special logic due to bug in pm.Model
        m.rvs_to_total_sizes = {
            vars_to_clone[rv]: i for rv, i in self.rvs_to_total_sizes.items() if rv in vars_to_clone
        }
        m.rvs_to_initial_values = {
            vars_to_clone[rv]: i for rv, i in self.rvs_to_initial_values.items()
        }
        m.free_RVs = [vars_to_clone[rv] for rv in self.free_RVs]
        m.observed_RVs = [vars_to_clone[rv] for rv in self.observed_RVs]
        m.potentials = [vars_to_clone[pot] for pot in self.potentials]
        m.deterministics = [vars_to_clone[det] for det in self.deterministics]

        m.marginalized_rvs = [vars_to_clone[rv] for rv in self.marginalized_rvs]
        return m

    def marginalize(self, rvs_to_marginalize: Union[TensorVariable, Sequence[TensorVariable]]):
        if not isinstance(rvs_to_marginalize, Sequence):
            rvs_to_marginalize = (rvs_to_marginalize,)

        supported_dists = (Bernoulli, Categorical, DiscreteUniform)
        for rv_to_marginalize in rvs_to_marginalize:
            if rv_to_marginalize not in self.free_RVs:
                raise ValueError(
                    f"Marginalized RV {rv_to_marginalize} is not a free RV in the model"
                )
            if not isinstance(rv_to_marginalize.owner.op, supported_dists):
                raise NotImplementedError(
                    f"RV with distribution {rv_to_marginalize.owner.op} cannot be marginalized. "
                    f"Supported distribution include {supported_dists}"
                )

            self._delete_rv_mappings(rv_to_marginalize)
            self.marginalized_rvs.append(rv_to_marginalize)

        # Raise errors and warnings immediately
        self.clone()._marginalize(user_warnings=True)


class MarginalRV(SymbolicRandomVariable):
    """Base class for Marginalized RVs"""


class FiniteDiscreteMarginalRV(MarginalRV):
    """Base class for Finite Discrete Marginalized RVs"""


def find_conditional_input_rvs(output_rvs, all_rvs):
    """Find conditionally indepedent input RVs"""
    blockers = [other_rv for other_rv in all_rvs if other_rv not in output_rvs]
    return [
        var
        for var in ancestors(output_rvs, blockers=blockers)
        if var in blockers
        or (var.owner is None and not isinstance(var, (Constant, SharedVariable)))
    ]


def is_conditional_dependent(
    dependent_rv: TensorVariable, dependable_rv: TensorVariable, all_rvs
) -> bool:
    """Check if dependent_rv is conditionall dependent on dependable_rv,
    given all conditionally independent all_rvs"""

    return dependable_rv in find_conditional_input_rvs((dependent_rv,), all_rvs)


def find_conditional_dependent_rvs(dependable_rv, all_rvs):
    """Find rvs than depend on dependable"""
    return [
        rv
        for rv in all_rvs
        if (rv is not dependable_rv and is_conditional_dependent(rv, dependable_rv, all_rvs))
    ]


def is_elemwise_subgraph(rv_to_marginalize, other_input_rvs, output_rvs):
    # TODO: No need to consider apply nodes outside the subgraph...
    fg = FunctionGraph(outputs=output_rvs, clone=False)

    non_elemwise_blockers = [
        o for node in fg.apply_nodes if not isinstance(node.op, Elemwise) for o in node.outputs
    ]
    blocker_candidates = [rv_to_marginalize] + other_input_rvs + non_elemwise_blockers
    blockers = [var for var in blocker_candidates if var not in output_rvs]

    truncated_inputs = [
        var
        for var in ancestors(output_rvs, blockers=blockers)
        if (
            var in blockers
            or (var.owner is None and not isinstance(var, (Constant, SharedVariable)))
        )
    ]

    # Check that we reach the marginalized rv following a pure elemwise graph
    if rv_to_marginalize not in truncated_inputs:
        return False

    # Check that none of the truncated inputs depends on the marginalized_rv
    other_truncated_inputs = [inp for inp in truncated_inputs if inp is not rv_to_marginalize]
    # TODO: We don't need to go all the way to the root variables
    if rv_to_marginalize in ancestors(
        other_truncated_inputs, blockers=[rv_to_marginalize, *other_input_rvs]
    ):
        return False
    return True


def replace_finite_discrete_marginal_subgraph(fgraph, rv_to_marginalize, all_rvs):
    # TODO: This should eventually be integrated in a more general routine that can
    #  identify other types of supported marginalization, of which finite discrete
    #  RVs is just one

    dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
    if not dependent_rvs:
        raise ValueError(f"No RVs depend on marginalized RV {rv_to_marginalize}")

    ndim_supp = {rv.owner.op.ndim_supp for rv in dependent_rvs}
    if max(ndim_supp) > 0:
        raise NotImplementedError(
            "Marginalization of withe dependent Multivariate RVs not implemented"
        )

    marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
    dependent_rvs_input_rvs = [
        rv
        for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
        if rv is not rv_to_marginalize
    ]

    # If the marginalized RV has batched dimensions, check that graph between
    # marginalized RV and dependent RVs is composed strictly of Elemwise Operations.
    # This implies (?) that the dimensions are completely independent and a logp graph
    # can ultimately be generated that is proportional to the support domain and not
    # to the variables dimensions
    # We don't need to worry about this if the  RV is scalar.
    if np.prod(constant_fold(tuple(rv_to_marginalize.shape))) > 1:
        if not is_elemwise_subgraph(rv_to_marginalize, dependent_rvs_input_rvs, dependent_rvs):
            raise NotImplementedError(
                "The subgraph between a marginalized RV and its dependents includes non Elemwise operations. "
                "This is currently not supported",
            )

    input_rvs = [*marginalized_rv_input_rvs, *dependent_rvs_input_rvs]
    rvs_to_marginalize = [rv_to_marginalize, *dependent_rvs]

    outputs = rvs_to_marginalize
    # Clone replace inner RV rng inputs so that we can be sure of the update order
    # replace_inputs = {rng: rng.type() for rng in updates_rvs_to_marginalize.keys()}
    # Clone replace outter RV inputs, so that their shared RNGs don't make it into
    # the inner graph of the marginalized RVs
    # FIXME: This shouldn't be needed!
    replace_inputs = {}
    replace_inputs.update({input_rv: input_rv.type() for input_rv in input_rvs})
    cloned_outputs = clone_replace(outputs, replace=replace_inputs)

    marginalization_op = FiniteDiscreteMarginalRV(
        inputs=list(replace_inputs.values()),
        outputs=cloned_outputs,
        ndim_supp=0,
    )
    marginalized_rvs = marginalization_op(*replace_inputs.keys())
    fgraph.replace_all(tuple(zip(rvs_to_marginalize, marginalized_rvs)))
    return rvs_to_marginalize, marginalized_rvs


@_get_measurable_outputs.register(FiniteDiscreteMarginalRV)
def _get_measurable_outputs_finite_discrete_marginal_rv(op, node):
    # Marginalized RVs are not measurable
    return node.outputs[1:]


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> Tuple[int, ...]:
    op = rv.owner.op
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        p_param = rv.owner.inputs[3]
        return tuple(range(pt.get_vector_length(p_param)))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(rv.owner.inputs[3:])
        return tuple(range(lower, upper + 1))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


@_logprob.register(FiniteDiscreteMarginalRV)
def finite_discrete_marginal_rv_logp(op, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    marginalized_rvs_node = op.make_node(*inputs)
    marginalized_rv, *dependent_rvs = clone_replace(
        op.inner_outputs,
        replace={u: v for u, v in zip(op.inner_inputs, marginalized_rvs_node.inputs)},
    )

    # Obtain the joint_logp graph of the inner RV graph
    # Some inputs are not root inputs (such as transformed projections of value variables)
    # Or cannot be used as inputs to an OpFromGraph (shared variables and constants)
    inputs = list(inputvars(inputs))
    rvs_to_values = {}
    dummy_marginalized_value = marginalized_rv.clone()
    rvs_to_values[marginalized_rv] = dummy_marginalized_value
    rvs_to_values.update(zip(dependent_rvs, values))
    logps_dict = factorized_joint_logprob(rv_values=rvs_to_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    values_axis_bcast = []
    for value in values:
        vbcast = value.type.broadcastable
        mbcast = dummy_marginalized_value.type.broadcastable
        mbcast = (True,) * (len(vbcast) - len(mbcast)) + mbcast
        values_axis_bcast.append([i for i, (m, v) in enumerate(zip(mbcast, vbcast)) if m != v])
    joint_logp = logps_dict[dummy_marginalized_value]
    for value, values_axis_bcast in zip(values, values_axis_bcast):
        joint_logp += logps_dict[value].sum(values_axis_bcast, keepdims=True)

    # Wrap the joint_logp graph in an OpFromGrah, so that we can evaluate it at different
    # values of the marginalized RV
    # OpFromGraph does not accept constant inputs
    non_const_values = [
        value
        for value in rvs_to_values.values()
        if not isinstance(value, (Constant, SharedVariable))
    ]
    joint_logp_op = OpFromGraph([*non_const_values, *inputs], [joint_logp], inline=True)

    # Compute the joint_logp for all possible n values of the marginalized RV. We assume
    # each original dimension is independent so that it sufficies to evaluate the graph
    # n times, once with each possible value of the marginalized RV replicated across
    # batched dimensions of the marginalized RV

    # PyMC does not allow RVs in the logp graph, even if we are just using the shape
    marginalized_rv_shape = constant_fold(tuple(marginalized_rv.shape))
    marginalized_rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
    marginalized_rv_domain_tensor = pt.swapaxes(
        pt.full(
            (*marginalized_rv_shape, len(marginalized_rv_domain)),
            marginalized_rv_domain,
            dtype=marginalized_rv.dtype,
        ),
        axis1=0,
        axis2=-1,
    )

    # OpFromGraph does not accept constant inputs
    non_const_values = [
        value for value in values if not isinstance(value, (Constant, SharedVariable))
    ]
    # Arbitrary cutoff to switch to Scan implementation to keep graph size under control
    if len(marginalized_rv_domain) <= 10:
        joint_logps = [
            joint_logp_op(marginalized_rv_domain_tensor[i], *non_const_values, *inputs)
            for i in range(len(marginalized_rv_domain))
        ]
    else:
        # Make sure this is rewrite is registered
        from pymc.pytensorf import local_remove_check_parameter

        def logp_fn(marginalized_rv_const, *non_sequences):
            return joint_logp_op(marginalized_rv_const, *non_sequences)

        joint_logps, _ = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*non_const_values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
        )

    joint_logps = pt.logsumexp(joint_logps, axis=0)

    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    return joint_logps, *(pt.constant(0),) * (len(values) - 1)
