"""
Simple Lot-Sizing Example with Inventory, Backlog and Setup.
The demand of each period can be fulfilled by the production of the same period,
or by inventory (production from previous periods). If the demand of a period is
not fulfilled it is a backlog.

Backlogs can be fulfilled in future periods (Delayed Demands) or can be not fulfilled
at all (Lost Demands = Backlog).

The production only occurs in the period if there is a setup or setup carry-over (setup
of product in the previous periods). The setup time also consumes the production time, so
the production time + setup time shall be less or equal to the period time available.

The setup prodide information about from which product to which product the setup is, this
enable custom setup costs for each pair product transition.

In this model there is only one production line.

The Minimization function will try to minimize the backlog, inventory amount and setup cost. The
backlog amount will have a higher weight.

Formulation:
    Variables:
        X_it = Production amount of product i in period t
        I_it = Inventory amount of product i in period t
        B_it = Backlog amount of product i in period t
        Z_ijt = Setup of product i from product j in period t
        ZA_it = Setup carry-over of product i in period t

    Params:
        D_it = Demand of product i in period t
        C_t = Time available to produce (capacity) at period t
        T_t = Time to produce 1000 units of products at period t
        SC_ij = Setup cost of product i from product j
        ST_ij = Setup time of product i from product j

    Minimize: sum((1.5 * B_it) + I_it), i=0..n, t=0..m) + sum(SC_ij * Z_ijt, i,j=0..n, t=0..m)         (for all products i, j and periods t)

    Subject to:
        1) Demand Fulfillment: Production, Inventory and Backlog Balance:
            The demand of each period shall be fulfilled by production or inventory, if
            not fulfilled it is a backlog.

            X_it + I_i(t-1) - I_it + B_it - B_i(t-1) == D_it        (for all products i and periods t)

        2) Ensure Setup to Produce:
            The production only occurs in the period if there is a setup or setup carry-over.

            X_it <= sum(D_it, t=0..n) * (sum(Z_ijt, j=0..n) + ZA_it)        (for all products i and periods t)

        3) Ensure Setup Only When Produce:
            Ensure that a setup will be made only when a production will happen in the same period or in the next one.
            This avoids useless setups (not used) that can happen in a local optimum.


            sum(Z_ijt, j=0..n) <= X_it + X_i(t+1)        (for all products i and periods t)

        4) Capacity Restriction:
            For all periods t, the time consumed by producing x products + the time to
            setup a new product must be less or equal to the period time available.

            sum(X_it * T_t, i=0..n) + sum(sum(Z_ijt * ST_ji, i=0..n), j=0..n) <= Ct            (for all periods t)

        5) Single Setup per Period:
            Ensure at maximum one setup for each period.

            sum(sum(Z_ijt, i=0..n), j=0..n) <= 1             (for all periods t)

        6) Single Setup Carry-Over per Period:
            Ensure at maximum one setup carry-over for each period.

            sum(ZA_it) <= 1             (for all periods t)

        7) Ensure Setup Carry-Over consistency:
            Only setup carry-over if the product had a setup or a setup carry-over at the previous period.

            ZA_it <= sum(Z_ij(t-1), j=0..n) + ZA_i(t-1)           (for all products i and periods t)

        8) Ensure Setup Carry-Over consistency:
            Only have setup carry-over if had setup (or setup carry-over) in the previous period and do
            not had setup of a different product in the last period.

            ZA_it <= 1 + sum(Z_ij(t-1), j=0..n) - sum(Z_kj(t-1), j=0..n)         (for all products i and k where i != k and for all periods t)

        9) Setup from products connection:
            To setup the product i from the product j at period t, must have a setup carry-over of the product j at t.
            Since we can have a single setup at each period and have a setup carry-over every time a setup occurred in
            the last period, we can use the setup carry over to identify the "product from".


            Z_ijt <= ZA_jt             (for all products i and j, and for all periods t)


        10) No setup between same product
            Ensure that we don't have setup from product i to I, which don't affect the cost but makes the output less clear.

            Z_iit = 0             (for all products i and for all periods t)

        11) Zeroed Production, Inventory, Backlog and Setup  at period T0 (not in the planning horizon):
            X_i0 = 0 (for all products i)
            I_i0 = 0 (for all products i)
            B_i0 = 0 (for all products i)
            Z_ij0 = 0 (for all products i and j)

        12) Initial Setup:
            For the first setup (transition) to happen we need to have a initial setup running on line at period 0. This is done
            setting the setup carry-over of the initial setup to 1.
            ZA_i0 = 0 (for all products i, except for the product initially available in line)

        13) At least one production:
            To ensure the production of small demands, we can ensure that all products has at least one opportunity to run. There
            must be at least one setup (Z) or setup carry-over (ZA) for each product in the demand.

            sum(Z_ijt, j=0..n, t=1..m) + sum(ZA_it, t=1..m) > 0 (for all products i)

        14) Do not produce more then demand
            To ensure the production to be <= then the demand of each product

            sum(X_it, t=0..m) <= sum(D_it, t=0..n)

        14) Non-Negativity for Production, Inventory and Backlog:
            X_it >= 0 (for all products i and periods t)
            I_it >= 0 (for all products i and periods t)
            B_it >= 0 (for all products i and periods t)

        15) Binary Variable for Setup and Setup Carry-Over:
            Z_ijt in {0, 1} (for all products i, j and periods t)
            ZA_it in {0, 1} (for all products i and periods t)
"""

import logging
import subprocess
from datetime import datetime, timedelta
from math import inf
from pathlib import Path

import gurobipy as gurobipy
import pandas as pd
from pulp import (
    LpBinary,
    LpContinuous,
    LpMinimize,
    LpProblem,
    LpSolutionOptimal,
    LpVariable,
    getSolver,
    lpSum,
)

from entities.optimization_result import (
    BacklogItem,
    InventoryItem,
    OptimizationResult,
    ProductionItem,
    SetupCarryOverItem,
    SetupItem,
    SolverStatus,
)
from entities.optimization_run import OptzRun
from entities.production_schedule import DEFAULT_PRECISION
from entities.scenario import Scenario
from use_cases.optimization.models import Model

_DUMP_VARIABLES_FOR_DEBUG = False

_SOLVER_THREADS_TO_USE = 4
_SOLVER_RELATIVE_GAP = 0.01

_SOLVER_PULP = "PULP_CBC_CMD"
_SOLVER_GUROBI = "GUROBI"
_DEFAULT_SOLVER = _SOLVER_GUROBI

GUROBI_STATUS_MAPPING = {
    gurobipy.GRB.OPTIMAL: SolverStatus.SOLVED_OPTIMAL,
    gurobipy.GRB.INFEASIBLE: SolverStatus.NOT_SOLVED,
    gurobipy.GRB.TIME_LIMIT: SolverStatus.SOLVED_NOT_OPTIMAL,
}

# ENSURE_PRODUCTION_OF_ALL_PRODUCTS = True  # TODO - Maybe this don't make much sense
DEFAULT_PRECISION = 3
# _MIN_TOTAL_PROD_BY_PRODUCT = 200
# _MIN_PROD_IN_FIRST_PERIOD = 200

# Select default solver if GUROBI is not available in the environment.
try:
    logging.info("GUROBI License Info:")

    result = subprocess.run(
        ["gurobi_cl", "--license"], capture_output=True, text=True, check=True
    )
    output = result.stdout

    if "Using license file" in output:
        logging.info("\nLicense validated, using GUROBI SOLVER.")
        _DEFAULT_SOLVER = _SOLVER_GUROBI
    else:
        logging.info("No license found, using PULP SOLVER.")
        _DEFAULT_SOLVER = _SOLVER_PULP

except subprocess.CalledProcessError as e:
    logging.error(f"Error with command gurobi_cl: {e}. Using PulP Solver Instead.")
    _DEFAULT_SOLVER = _SOLVER_PULP

except FileNotFoundError:
    logging.warning("GUROBI not found in you system. Using PulP Solver Instead.")
    _DEFAULT_SOLVER = _SOLVER_PULP

# This param is used to balance Backlog cost and Setup cost. It represents
# how many procts need to be postponed/backloged in order to be more "expensive"
# then a setup. This value can be adjusted as the system evolves.
# 1/X, where X is the number of products that can be backloged/postponed before
# becoming more expensive than executing a setup.
# Obs.: The user can define a setup cost weight multiplier, which will impact the
# relation between the both costs, but he idea is the same
_POSTPONING_WEIGHT = round(1 / 100, 6)  # Not meeting in the right month
_UNMET_WEIGHT = _POSTPONING_WEIGHT  # Not meeting at the end of period
_INVENTORY_WEIGHT = (
    _POSTPONING_WEIGHT * 0.01
)  # 1% of the Postpnment (only minimize when not affect much the postnoments)


# Model Variables Name
_PRODUCTION_VAR_NAME = "Production"
_INVENTORY_VAR_NAME = "Inventory"
_BACKLOG_VAR_NAME = "Backlog"
_SETUP_VAR_NAME = "Setup"
_SETUP_CARRY_OVER_NAME = "SetupCarryOver"

__DUMMY_PRODUCTS_MAP_TO = {}
__DUMMY_PRODUCTS_MAP_FROM = {}


class MissingDataError(Exception):
    pass


class MissingProductionRateError(Exception):
    pass


class DumpModelUnknowFormatError(Exception):
    pass


class ReadModelUnknowFormatError(Exception):
    pass


class UnknowVariableNameError(Exception):
    pass


def _gen_products_list(prod_need: pd.DataFrame, initial_setup: str) -> list[str]:
    return list(set(prod_need.sku.unique().tolist() + [initial_setup]))


def _gen_setup_cost_matrix(
    products: list[str],
    setup_matrix: pd.DataFrame,
) -> dict[dict]:
    """
    Generate the setup cost matrix for the model.

    Parameters
        products : list[str]
            List of products to be considered.
        setup_matrix : pd.DataFrame
            Setup matrix with the setup complexity.

    Returns a setup cost dict in the form of:
    {
        <product_from>:
            {
                <product_to_1>: <cost>,
                <product_to_2>: <cost>,
                ...
            }
    }
    """
    setup_matrix_lookup = setup_matrix.set_index(["sku_from", "sku_to"])
    lookup_dict = {}
    for product_from in products:
        tmp_from_dict = lookup_dict.get(product_from, {})
        for product_to in products:
            try:
                tmp_from_dict[product_to] = round(
                    float(
                        setup_matrix_lookup.loc[(product_from, product_to)].setup_cost
                    ),
                    DEFAULT_PRECISION,
                )
            except KeyError:
                logging.warning(
                    f"Not setup_matrix value for skus: {product_from} -> {product_to}"
                )
                tmp_from_dict[product_to] = 0.0

        lookup_dict[product_from] = tmp_from_dict

    return lookup_dict


def _gen_setup_time_matrix(
    products: list[str],
    setup_matrix: pd.DataFrame,
    min_setup_time_multiplier: float,
    max_setup_time_multiplier: float,
) -> dict[dict]:
    setup_matrix_lookup = setup_matrix.set_index(["sku_from", "sku_to"])

    def calc_setup_time(product_from: str, product_to: str):
        if product_from != product_to:
            return round(
                float(
                    min_setup_time_multiplier
                    + (
                        setup_matrix_lookup.loc[(product_from, product_to)].setup_cost
                        * (max_setup_time_multiplier - min_setup_time_multiplier)
                    )
                ),
                DEFAULT_PRECISION,
            )
        else:
            return 0.0

    lookup_dict = {}
    for product_from in products:
        tmp_from_dict = lookup_dict.get(product_from, {})
        for product_to in products:
            try:
                tmp_from_dict[product_to] = calc_setup_time(
                    product_from=product_from, product_to=product_to
                )
            except KeyError:
                tmp_from_dict[product_to] = min_setup_time_multiplier

        lookup_dict[product_from] = tmp_from_dict

    return lookup_dict


def _gen_periods(
    horizon_start: datetime,
    planning_hours_slot: int,
    last_demand_date: datetime,
) -> list[dict]:
    """
    Generate the list of periods given the starting datetime, the planning hours slot
    and the last demand datetime.

    @param horizon_start : datetime
        Starting datetime for the planning period (it will be adjusted to HH:00:00)
    @oaram planning_hours_slot : int
        Planning periods size (in hours)
    @param last_demand_date : datetime
        Date of the last demand (it will be extended to the 00:00:00 of next day)

    Returns
        List of periods objects in the format [{"<periods_label>": <period_size>}]
    """
    # Init with the first period (out of the planning horizon)
    periods = [
        {
            "label": "T0",
            "start_date": None,
            "end_date": None,
            "prod_time_available": 0,
        }
    ]

    horizon_start = horizon_start.replace(minute=0, second=0, microsecond=0)
    last_demand_date = (last_demand_date + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    period_start = horizon_start
    period_idx = 1
    while period_start < last_demand_date:
        # Calculate the end datetime for the period with last_demand_date as a limit
        period_end = min(
            period_start + timedelta(hours=planning_hours_slot),
            last_demand_date,
        )

        # Add the new period to the list
        periods.append(
            {
                "label": f"T{period_idx}",
                "start_date": period_start,
                "end_date": period_end,
                "prod_time_available": int(
                    (period_end - period_start).total_seconds() / 60
                ),
            }
        )

        # Calculate the new loop params
        period_start = period_end
        period_idx += 1

    return periods


def _gen_time_to_produce(
    periods: list[dict], production_rate: pd.DataFrame
) -> list[dict]:
    """
    Add the "Time to Produce 1000s of Products" into the periods based on the
    production rate information.

    @param periods : list[dict]
        Periods dicts list as generated in _gen_periods function.

    @param production_rate : pd.DataFrame
        Dataframe with production rate reference by date.

    Returns
        New list of periods with two new fields called "prod_rate" and "time_to_produce".
    """
    production_rate_ref = production_rate.copy()
    production_rate_ref["day"] = production_rate_ref["ref_date"].dt.strftime("%d/%m/%Y")
    production_rate_ref = production_rate_ref.set_index(["day"])

    last_prod_rate = None
    for period in periods:
        if period["start_date"] is None:
            period["prod_rate"] = 0
            period["time_to_produce"] = inf
        else:
            tmp_day = period["start_date"].strftime("%d/%m/%Y")
            try:
                last_prod_rate = float(production_rate_ref.loc[tmp_day].prod_rate)
            except KeyError:
                if last_prod_rate is None:
                    raise MissingProductionRateError(
                        f"No production rate for {tmp_day}"
                    )
                else:
                    logging.warning(
                        f"No production rate for {tmp_day}, using the last know of {last_prod_rate}."
                    )

            # Production rate are provided by day and we need to convert it to production / minute
            period["prod_rate"] = round(last_prod_rate / 24 / 60, DEFAULT_PRECISION)
            period["time_to_produce"] = round(
                1 / period["prod_rate"], DEFAULT_PRECISION
            )

    return sorted(
        periods,
        key=lambda p: (
            p["start_date"] if p["start_date"] is not None else datetime(1970, 1, 1)
        ),  # Very old Data,
    )


def _get_related_period_label(date: datetime, periods: list[dict]) -> str:
    """
    Given a list of periods dicts and date, return the period label
    which the date is inserted.

    @params date : datetime:
        Datetime object to find the related period.

    @params periods : list[dict]
        List of periods dicts in the format generated by _gen_time_to_produce

    Returns
        The period label related with the date.
    """

    return list(
        filter(
            lambda p: (p["start_date"] is not None)
            and (p["start_date"] <= date <= p["end_date"]),
            periods,
        )
    )[0]["label"]


def _secure_label(label: str) -> str:
    """
    The PuLP internally uses "_" to consolidate variables names and its related
    indexes. So we need to ensure that the labels do not has "_" on it and ensure
    that we can rebuild the labels with "_" properly.

    Parameters
        label : str
            The label to make secure
    """

    try:
        return __DUMMY_PRODUCTS_MAP_TO[label]
    except KeyError:
        last_id = len(__DUMMY_PRODUCTS_MAP_TO)
        secured_label = f"PROD{last_id + 1}"
        __DUMMY_PRODUCTS_MAP_TO[label] = secured_label
        __DUMMY_PRODUCTS_MAP_FROM[secured_label] = label

        return secured_label


def _from_secured_label(secured_label: str) -> str:
    """
    Convert secured labels, recovering any "_" char originally into the label.

    Parameters
        label : str
            The secured label to recover the original label
    """

    return __DUMMY_PRODUCTS_MAP_FROM[secured_label]


class LineSchedulingV1(Model):
    optz_run: OptzRun
    line: str
    products: list[str]
    initial_setup: str
    periods: list[dict]
    periods_by_lbl: dict
    prod_need: pd.DataFrame
    setup_cost: dict[list]
    setup_time: dict[list]
    status: int
    lp_model: LpProblem
    X: LpVariable = None
    I: LpVariable = None
    B: LpVariable = None
    Z: LpVariable = None
    ZA: LpVariable = None

    def get_periods_labels(self) -> list[str]:
        """
        Convert the periods objects list into the labels only list
        """
        return [period["label"] for period in self.periods]

    def __instantiate_problem(self) -> LpProblem:
        """
        Create the PuLP's problem instance with the object data. The
        problem instance will have all equations configured and will
        be ready to run the optimization algorithm.

        Returns
            A pulp.LpProblem instance with the configured values according
            to the LineSchedulingV1's object data.
        """

        # Create the instance
        prob = LpProblem("LineSchedulingV1", LpMinimize)

        # Prepare some data to be used during the problem configuration
        periods_lbls = self.get_periods_labels()
        self.periods_by_lbl = {period["label"]: period for period in self.periods}

        # Secured products
        secured_products = list(map(_secure_label, self.products))

        # Prepare Production Need lookup
        self.prod_need["period_label"] = self.prod_need["deadline"].apply(
            lambda deadline: _get_related_period_label(
                date=deadline, periods=self.periods
            )
        )
        prod_need_lookup = (
            self.prod_need[["sku", "period_label", "prod_need"]]
            .groupby(by=["sku", "period_label"])
            .sum(numeric_only=True)
            .sort_index()
        )

        # Define Variables

        # Decision Variable for Production (Non-Negative)
        self.X = LpVariable.dicts(
            name=_PRODUCTION_VAR_NAME,
            indices=(secured_products, periods_lbls),
            lowBound=0,
            cat=LpContinuous,
        )

        # Inventory Variable (Non-Negative)
        self.I = LpVariable.dicts(
            name=_INVENTORY_VAR_NAME,
            indices=(secured_products, periods_lbls),
            lowBound=0,
            cat=LpContinuous,
        )

        # Backlog Variable (Non-Negative)
        self.B = LpVariable.dicts(
            name=_BACKLOG_VAR_NAME,
            indices=(secured_products, periods_lbls),
            lowBound=0,
            cat=LpContinuous,
        )

        # Setup Variable (Binary)
        self.Z = LpVariable.dicts(
            name=_SETUP_VAR_NAME,
            indices=(secured_products, secured_products, periods_lbls),
            cat=LpBinary,
        )

        # Setup Carry-Over Variable (Binary)
        self.ZA = LpVariable.dicts(
            name=_SETUP_CARRY_OVER_NAME,
            indices=(secured_products, periods_lbls),
            cat=LpBinary,
        )

        # Restrictions
        # 1) Demand Fulfillment: Production, Inventory and Backlog Balance:
        # X_it + I_i(t-1) - I_it + B_it - B_i(t-1) == D_it        (for all products i and periods t)
        for product in secured_products:
            for period in periods_lbls[1:]:
                try:
                    tmp_prod_need = prod_need_lookup.loc[
                        (_from_secured_label(product), period), "prod_need"
                    ]
                except KeyError:
                    tmp_prod_need = 0.0

                prob += (
                    (
                        self.X[product][period]
                        + self.I[product][periods_lbls[periods_lbls.index(period) - 1]]
                        - self.I[product][period]
                        + self.B[product][period]
                        - self.B[product][periods_lbls[periods_lbls.index(period) - 1]]
                    )
                    == tmp_prod_need,
                    f"Rest 1 (Demand Fulfillment): {tmp_prod_need} of {product} in {period}",
                )

        # 2) Ensure Setup to Produce:
        # X_it <= sum(D_it, t=0..n) * (sum(Z_ijt, j=0..n) + ZA_it)        (for all products i and periods t)
        for product in secured_products:
            for period in periods_lbls[1:]:
                tmp_demand_for_product = 0.0
                for t in periods_lbls:
                    try:
                        tmp_demand_for_product += prod_need_lookup.loc[
                            (_from_secured_label(product), t), "prod_need"
                        ]
                    except KeyError:
                        pass  # Ignore 0s

                prob += (
                    self.X[product][period]
                    <= (
                        tmp_demand_for_product
                        * (
                            lpSum(
                                [
                                    self.Z[product][product_from][period]
                                    for product_from in secured_products
                                    if product_from != product
                                ]
                            )
                            + self.ZA[product][period]
                        )
                    ),
                    f"Rest 2 (Setup to Produce) {round(tmp_demand_for_product, 2)} of {product} at {period}",
                )

        # 3) Ensure Setup Only When Produce:
        # sum(Z_ijt, j=0..n) <= X_it + X_i(t+1)        (for all products i and periods t)
        for product in secured_products:
            for period_idx, period in enumerate(periods_lbls[1:-1]):
                period_idx += 1  # Ajust the offset (slice start on [1:])
                prob += (
                    lpSum(
                        [
                            self.Z[product][product_from][period]
                            for product_from in secured_products
                            if product_from != product
                        ]
                    )
                    <= self.X[product][period]
                    + self.X[product][periods_lbls[period_idx + 1]],
                    f"Rest 3 (Setup Only When Produce) of {product} at {period} or {periods_lbls[period_idx + 1]}",
                )

        # 4) Capacity Restriction:
        # sum(X_it * T_t, i=0..n) + sum(sum(Z_ijt * ST_ji, i=0..n), j=0..n) <= Ct            (for all periods t)
        for period in periods_lbls[1:]:
            prob += (
                (
                    lpSum(
                        [
                            self.X[product][period]
                            * self.periods_by_lbl[period]["time_to_produce"]
                            for product in secured_products
                        ]
                    )
                    + lpSum(
                        [
                            (
                                self.Z[product_to][product_from][period]
                                * int(
                                    round(
                                        self.setup_time[
                                            _from_secured_label(product_from)
                                        ][_from_secured_label(product_to)],
                                        0,
                                    )
                                )
                            )
                            for product_to in secured_products
                            for product_from in secured_products
                        ]
                    )
                    <= self.periods_by_lbl[period]["prod_time_available"]
                ),
                f"Rest 4 (Capacity Restriction): {self.periods_by_lbl[period]['prod_time_available']} limit at {period}",
            )

        # 5) Single Setup per Period:
        # sum(sum(Z_ijt, i=0..n), j=0..n) <= 1             (for all periods t)
        for period in periods_lbls[1:]:
            setup_pairs = []
            for product_to in secured_products:
                for product_from in secured_products:
                    if product_to != product_from:
                        setup_pairs.append((product_to, product_from))

            prob += (
                (
                    lpSum(
                        [
                            self.Z[product_to][product_from][period]
                            for product_to, product_from in setup_pairs
                        ]
                    )
                    <= 1
                ),
                f"Rest 5 (Single Setup per Period): {period}",
            )

        # 6) Single Setup Carry-Over:
        # sum(ZA_it) <= 1             (for all periods t)
        for period in periods_lbls[1:]:
            prob += (
                (
                    lpSum([self.ZA[product][period] for product in secured_products])
                    <= 1
                ),
                f"Rest 6 (Single Setup Carry-Over): {period}",
            )

        # 7) Ensure Setup Carry-Over consistency:
        # ZA_it <= sum(Z_ij(t-1), j=0..n) + ZA_i(t-1)           (for all products i and periods t)
        for product_to in secured_products:
            for period in periods_lbls[1:]:
                prob += (
                    self.ZA[product_to][period]
                    <= (
                        lpSum(
                            [
                                self.Z[product_to][product_from][
                                    periods_lbls[periods_lbls.index(period) - 1]
                                ]
                                for product_from in secured_products
                                if product_from != product_to
                            ]
                        )
                        + self.ZA[product_to][
                            periods_lbls[periods_lbls.index(period) - 1]
                        ]
                    ),
                    f"Rest 7 (Ensure Setup Carry-Over Consistency): {product_to} at {period}",
                )

        # 8) Ensure Setup Carry-Over consistency:
        # ZA_it <= 1 + sum(Z_ij(t-1), j=0..n) - sum(Z_kj(t-1), j=0..n)         (for all products i and k where i != k and for all periods t)
        for product_to in secured_products:
            for product_k_to in secured_products:
                if product_k_to != product_to:
                    for period in periods_lbls[1:]:
                        prob += (
                            self.ZA[product_to][period]
                            <= (
                                1
                                + lpSum(
                                    [
                                        self.Z[product_to][product_from][
                                            periods_lbls[periods_lbls.index(period) - 1]
                                        ]
                                        for product_from in secured_products
                                        if product_from != product_to
                                    ]
                                )
                                - lpSum(
                                    [
                                        self.Z[product_k_to][product_from][
                                            periods_lbls[periods_lbls.index(period) - 1]
                                        ]
                                        for product_from in secured_products
                                        if product_from != product_k_to
                                    ]
                                )
                            ),
                            f"Rest 8 (Ensure Setup Carry-Over Consistency): {product_to} {product_k_to} {period}",
                        )

        # 9) Setup from products connection:
        # Z_ijt <= ZA_jt             (for all products i and j, and for all periods t)
        for product_to in secured_products:
            for product_from in secured_products:
                # Ignore same products on this restriciton
                if product_from == product_to:
                    continue

                for period in periods_lbls[1:]:
                    prob += (
                        self.Z[product_to][product_from][period]
                        <= self.ZA[product_from][period],
                        f"Rest 9: ({product_to},{product_from},{period})",
                    )

        # 10) No setup between same product
        # Z_iit = 0             (for all products i and for all periods t)
        for product in secured_products:
            for period in periods_lbls[1:]:
                prob += (
                    self.Z[product][product][period] == 0,
                    f"Rest 10 (No Setup between same products): {product} {period}",
                )

        # 11) Zeroed Production, Inventory, Backlog, Setup and Setup Carry-Over at period T0 (not in the planning horizon):
        #     X_i0 = 0 (for all products i)
        #     I_i0 = 0 (for all products i)
        #     B_i0 = 0 (for all products i)
        #     Z_ij0 = 0 (for all products i and j)
        #     ZA_i0 = 0 (for all products i)
        for product in secured_products:
            prob += (
                self.X[product][periods_lbls[0]] == 0,
                f"Rest 11 (Zeroed Production): {product}",
            )
            prob += (
                self.I[product][periods_lbls[0]] == 0,
                f"Rest 11 (Zeroed Inventory): {product}",
            )
            prob += (
                self.B[product][periods_lbls[0]] == 0,
                f"Rest 11 (Zeroed Backlog): {product}",
            )

            for product_from in secured_products:
                prob += (
                    self.Z[product][product_from][periods_lbls[0]] == 0,
                    f"Rest 11 (Zeroed Setup): {product} {product_from}",
                )

            # Setup Carry-Over will be used to set the initial setup
            if _from_secured_label(product) != self.initial_setup:
                prob += (
                    self.ZA[product][periods_lbls[0]] == 0,
                    f"Rest 11 (Zeroed Setup Carry-Over): {product}",
                )

        # 12) Initial Setup:
        #     For the first setup (transition) to happen we need to have a initial setup running on line at period 1 (first production period). This
        #     is done setting the setup carry-over of the initial setup to 1.
        #     ZA_i1 = 0 (for all products i, except for the product initially available in line)
        #     Z_i1 = 0 (for all products i. Not Setup allowed in the first period)
        for product in secured_products:
            if _from_secured_label(product) == self.initial_setup:
                # Period 0 (non planning)
                prob += (
                    self.ZA[product][periods_lbls[0]] == 1,
                    f"Rest 12 Initial Setup T0: {product}",
                )

                # First production planning enabled to produce initial setup product
                prob += (
                    self.ZA[product][periods_lbls[1]] == 1,
                    f"Rest 12 Initial Setup T1: {product}",
                )

                # Cannot setup from the initial label to any other label at the first planning period (generate stability with current running production)
                for product_to in secured_products:
                    prob += (
                        self.Z[product_to][product][periods_lbls[1]] == 0,
                        f"Rest 12 Initial Setup: {product} -> {product_to}",
                    )

                break  # Done

        # TODO - Update docs from here
        # 13) Ensure no stock at last period
        # Ensure that we don't have inventory in the last period, or meet all demand or it is backlog.
        for product in secured_products:
            prob += (
                self.I[product][periods_lbls[-1]] == 0,
                f"Rest 13 Ensure inventory at last period is lower or equal to 0: {product}",
            )

        # 14) Do not produce more then demand
        #     To ensure the production to be <= then the demand of each product

        #     sum(X_it, t=0..m) <= sum(D_it, t=0..n)
        for product in secured_products:
            tmp_total_product_demand = round(
                float(
                    prod_need_lookup.loc[
                        (_from_secured_label(product)), "prod_need"
                    ].sum()
                ),
                DEFAULT_PRECISION,
            )

            prob += (
                lpSum([self.X[product][period] for period in periods_lbls[1:]])
                <= tmp_total_product_demand
            )

        # Objective Function
        # Minimize the Postponing, Backlog, Setup Cost and Number of Setups
        prob += (
            # Postponing cost
            lpSum(
                [
                    _POSTPONING_WEIGHT * self.B[product][period]
                    for product in secured_products
                    for period in periods_lbls[1:-1]
                ]
            )
            # Unmet demand cost
            + lpSum(
                [
                    _UNMET_WEIGHT * self.B[product][periods_lbls[-1]]
                    for product in secured_products
                ]
            )
            # Number of Setups - Drive less setups
            + lpSum(
                [
                    (
                        self.optz_run.setup_cost
                        * 2  # 2x in relating to "Setup Cost"
                        * self.Z[product_to][product_from][period]
                    )
                    for product_from in secured_products
                    for product_to in secured_products
                    for period in periods_lbls
                ]
            )
            # Inventory cost to drive production to happen closer to the demand date
            + lpSum(
                [
                    _INVENTORY_WEIGHT * self.I[product][period]
                    for product in secured_products
                    for period in periods_lbls[1:]  # No stock on first period
                ]
            ),
            # TODO - Disabled while "too similar" labels restrictions are not aligned with production teams
            # # Setup Cost - Drive better setups, when a setup happens
            # + lpSum(
            #     [
            #         (
            #             self.setup_cost[_from_secured_label(product_from)][
            #                 _from_secured_label(product_to)
            #             ]
            #             * self.optz_run.setup_cost
            #             * self.Z[product_to][product_from][period]
            #         )
            #         for product_from in secured_products
            #         for product_to in secured_products
            #         for period in periods_lbls
            #     ]
            # ),
            "Objective Function",
        )

        return prob

    def build_model(
        self, plant: str, line: str, scenario: Scenario, optz_run: OptzRun
    ) -> bool:
        """
        Create the PuLP's problem given the parameters.
        """
        # Save the plant and line in the object
        self.plant = plant
        self.line = line

        # Save the production need dataframe into the object
        self.prod_need = scenario.production_need.get_prod_need_for_line(line=line)

        # Set initial setup
        self.initial_setup = scenario.initial_setup.get_initial_setup_for_line(
            line=line
        )

        # Generate products list
        self.products = _gen_products_list(
            prod_need=self.prod_need, initial_setup=self.initial_setup
        )

        # Calculate Setup Cost and Time matrices.
        setup_matrix = scenario.setup_matrix.get_setup_matrix_for_skus(
            skus=self.products
        )
        self.setup_cost = _gen_setup_cost_matrix(
            products=self.products, setup_matrix=setup_matrix
        )
        self.setup_time = _gen_setup_time_matrix(
            products=self.products,
            setup_matrix=setup_matrix,
            min_setup_time_multiplier=optz_run.min_setup_time,
            max_setup_time_multiplier=optz_run.max_setup_time,
        )

        # Generate periods blocks
        self.periods = _gen_periods(
            horizon_start=scenario.horizon_start,
            planning_hours_slot=optz_run.planning_slot_size,
            last_demand_date=scenario.production_need.last_date,
        )

        # Add production rate and time to produce
        self.periods = _gen_time_to_produce(
            periods=self.periods,
            production_rate=scenario.production_rate.get_prod_rate_for_line(line=line),
        )

        # Create the LpProblem instance
        self.lp_model = self.__instantiate_problem()

        return True

    def __init__(
        self,
        scenario: Scenario = None,
        optz_run: OptzRun = None,
        plant: str = None,
        line: str = None,
    ) -> None:
        if scenario is not None and optz_run is not None and line is not None:
            self.optz_run = optz_run
            assert self.build_model(
                plant=plant, line=line, scenario=scenario, optz_run=optz_run
            )

    def dump_model(self, file_path: Path | str) -> None:
        """
        Dumps the Linear Programming model to a LP file.

        Parameters:
            file_path : Path | str
                Destination file path to dump the model.
        """

        if type(file_path) == str:
            file_path = Path(file_path)

        if file_path.suffix == ".lp":
            self.lp_model.writeLP(file_path)
        elif file_path.suffix == ".mps":
            self.lp_model.writeMPS(file_path)
        elif file_path.suffix == ".json":
            self.lp_model.to_json(file_path)
        else:
            raise DumpModelUnknowFormatError(
                f"Unknow '{file_path.suffix}' file format to dump the model."
            )

    @staticmethod
    def load_model(file_path: Path | str) -> LpProblem:
        """
        Load the Linear Programming model to a LP file.

        Parameters:
            file_path : Path | str
                Destination file path to dump the model.
        """

        if type(file_path) == str:
            file_path = Path(file_path)

        if file_path.suffix == ".mps":
            return LpProblem.fromMPS(file_path)[1]
        elif file_path.suffix == ".json":
            return LpProblem.fromJson(file_path)[1]
        else:
            raise ReadModelUnknowFormatError(
                f"Unknow '{file_path.suffix}' file format to read the model."
            )

    def _split_variables(self) -> dict:
        """
        Separate the model variables according with the problem variables.
        """
        production_vars = []
        inventory_vars = []
        backlog_vars = []
        setup_vars = []
        setup_carry_over_vars = []

        for var in self.lp_model.variables():
            if _PRODUCTION_VAR_NAME in var.name:
                production_vars.append(var)
            elif _INVENTORY_VAR_NAME in var.name:
                inventory_vars.append(var)
            elif _BACKLOG_VAR_NAME in var.name:
                backlog_vars.append(var)
            elif _SETUP_CARRY_OVER_NAME in var.name:
                setup_carry_over_vars.append(var)
            elif _SETUP_VAR_NAME in var.name:
                setup_vars.append(var)
            else:
                logging.error(f"Unknow variable: {var.name}")

        return {
            "production_vars": production_vars,
            "inventory_vars": inventory_vars,
            "backlog_vars": backlog_vars,
            "setup_vars": setup_vars,
            "setup_carry_over_vars": setup_carry_over_vars,
        }

    def _decode_variables(
        self, variables: list[LpVariable], variable_name: str
    ) -> list[dict]:
        """
        Given the variable name pattern and a list of variables, decode the variable
        into its attributes (e.g., Product, Period, etc).
        """

        # TODO - Now the variables are stored into the class variables (X, I, B, etc). Refactor this code to iterate the variables dicts.
        decoded_data = []

        if variable_name == _PRODUCTION_VAR_NAME:
            # Production variables are in the following format
            # "<_PRODUCTION_VAR_NAME>_<PRODUCT>_<PERIOD>"
            for var in variables:
                splited_name = var.name.split("_")
                tmp_product = splited_name[1]
                tmp_period = splited_name[2]

                # Skip the T0 period and 0 productions
                if (
                    tmp_period != "T0" and var.varValue > 1e-3
                ):  # Avoid noise from small numbers
                    decoded_data.append(
                        ProductionItem(
                            line=self.line,
                            plant=self.plant,
                            product=_from_secured_label(tmp_product),
                            period_start=self.periods_by_lbl[tmp_period]["start_date"],
                            period_end=self.periods_by_lbl[tmp_period]["end_date"],
                            amount=round(float(var.varValue), DEFAULT_PRECISION),
                            time_minutes=int(
                                var.varValue
                                * self.periods_by_lbl[tmp_period]["time_to_produce"]
                            ),
                        )
                    )
        elif variable_name == _INVENTORY_VAR_NAME:
            # Inventory variables are in the following format
            # "<_INVENTORY_VAR_NAME>_<PRODUCT>_<PERIOD>"
            for var in variables:
                splited_name = var.name.split("_")
                tmp_product = splited_name[1]
                tmp_period = splited_name[2]

                # Skip the T0 period and zeroed inventory
                if tmp_period != "T0" and var.varValue > 0.0:
                    decoded_data.append(
                        InventoryItem(
                            line=self.line,
                            plant=self.plant,
                            product=_from_secured_label(tmp_product),
                            period_start=self.periods_by_lbl[tmp_period]["start_date"],
                            period_end=self.periods_by_lbl[tmp_period]["end_date"],
                            amount=round(float(var.varValue), DEFAULT_PRECISION),
                        )
                    )
        elif variable_name == _BACKLOG_VAR_NAME:
            # Backlog variables are in the following format
            # "<_BACKLOG_VAR_NAME>_<PRODUCT>_<PERIOD>"
            for var in variables:
                splited_name = var.name.split("_")
                tmp_product = splited_name[1]
                tmp_period = splited_name[2]

                # Skip the T0 period and zeroed backlog
                if tmp_period != "T0" and var.varValue > 0.0:
                    decoded_data.append(
                        BacklogItem(
                            line=self.line,
                            plant=self.plant,
                            product=_from_secured_label(tmp_product),
                            period_start=self.periods_by_lbl[tmp_period]["start_date"],
                            period_end=self.periods_by_lbl[tmp_period]["end_date"],
                            amount=round(float(var.varValue), DEFAULT_PRECISION),
                            cost=round(
                                _POSTPONING_WEIGHT * var.varValue,
                                DEFAULT_PRECISION,
                            ),
                        )
                    )
        elif variable_name == _SETUP_VAR_NAME:
            # Setup variables are in the following format
            # "<_SETUP_VAR_NAME>_<PRODUCT_TO>_<PRODUCT_FROM>_<PERIOD>"
            for var in variables:
                splited_name = var.name.split("_")
                tmp_product_to = splited_name[1]
                tmp_product_from = splited_name[2]
                tmp_period = splited_name[3]

                # Skip the T0 period, evaluation of same product setups and setups that didn't had happened
                if (
                    tmp_period != "T0"
                    and tmp_product_from != tmp_product_to
                    and var.varValue is not None
                    and var.varValue > 1e-5  # Default Int Precision of the solver
                ):
                    tmp_setup_time_minutes = int(
                        self.setup_time[_from_secured_label(tmp_product_from)][
                            _from_secured_label(tmp_product_to)
                        ]
                    )
                    tmp_setup_cost = self.setup_cost[
                        _from_secured_label(tmp_product_from)
                    ][_from_secured_label(tmp_product_to)]

                    decoded_data.append(
                        SetupItem(
                            line=self.line,
                            plant=self.plant,
                            product_from=_from_secured_label(tmp_product_from),
                            product_to=_from_secured_label(tmp_product_to),
                            period_start=self.periods_by_lbl[tmp_period]["start_date"],
                            period_end=self.periods_by_lbl[tmp_period]["end_date"],
                            setup=True,
                            time_minutes=tmp_setup_time_minutes,
                            cost=tmp_setup_cost,
                        )
                    )
        elif variable_name == _SETUP_CARRY_OVER_NAME:
            # Setup carry over variables are in the following format
            # "<_SETUP_CARRY_OVER_NAME>_<PRODUCT>_<PERIOD>"
            for var in variables:
                splited_name = var.name.split("_")
                tmp_product = splited_name[1]
                tmp_period = splited_name[2]

                # Skip the T0 period and setup that didn't had happened
                if (
                    tmp_period != "T0"
                    and var.varValue is not None
                    and var.varValue > 1e-5  # Default Int Precision of the solver
                ):
                    decoded_data.append(
                        SetupCarryOverItem(
                            line=self.line,
                            plant=self.plant,
                            product=_from_secured_label(tmp_product),
                            period_start=self.periods_by_lbl[tmp_period]["start_date"],
                            period_end=self.periods_by_lbl[tmp_period]["end_date"],
                            setup_carry_over=True,
                        )
                    )
        else:
            raise UnknowVariableNameError(f"Unknow variable name: {variable_name}")

        return sorted(decoded_data, key=lambda tmp_data: tmp_data.period_start)

    def get_fo(self) -> float:
        return self.lp_model.objective.value()

    def build_results(self) -> OptimizationResult:
        """
        Convert the model variables into the expected OptimizationResult class.
        """

        variables = self._split_variables()

        production_items = self._decode_variables(
            variables=variables["production_vars"],
            variable_name=_PRODUCTION_VAR_NAME,
        )
        inventory_items = self._decode_variables(
            variables=variables["inventory_vars"],
            variable_name=_INVENTORY_VAR_NAME,
        )
        backlog_items = self._decode_variables(
            variables=variables["backlog_vars"],
            variable_name=_BACKLOG_VAR_NAME,
        )
        setup_items = self._decode_variables(
            variables=variables["setup_vars"], variable_name=_SETUP_VAR_NAME
        )
        setup_carry_over_items = self._decode_variables(
            variables=variables["setup_carry_over_vars"],
            variable_name=_SETUP_CARRY_OVER_NAME,
        )

        if _DUMP_VARIABLES_FOR_DEBUG:
            self.dump_rest_2()
            self.dump_rest_14()

        return OptimizationResult(
            optz_run_id=self.optz_run.id,
            time_s=self.lp_model.solutionTime,
            objective_value=self.get_fo(),
            solver_status=self.status,
            production=production_items,
            inventory=inventory_items,
            backlog=backlog_items,
            setup=setup_items,
            setup_carry_over=setup_carry_over_items,
        )

    def optimize(
        self,
        solver: str = _DEFAULT_SOLVER,
        time_limit_s=None,
        verbose=False,
        relative_gap=_SOLVER_RELATIVE_GAP,
        threads=_SOLVER_THREADS_TO_USE,
    ) -> OptimizationResult:
        """
        Execute the solver against the problem built and update
        the problem variables with the result.
        """
        time_limit_s = (
            time_limit_s if time_limit_s is not None else self.optz_run.time_limit * 60
        )
        logging.debug(f"Solving Line Scheduling V1.0 with {solver}")
        logging.debug(f"Line: {self.line}")
        logging.debug(f"Products: {len(self.products)}")
        logging.debug(f"Periods: {len(self.periods)}")
        logging.debug(f"Time Limit: {time_limit_s} seconds")

        if solver == _SOLVER_GUROBI:
            # Optimized params using GUROBI tunning tool
            # Focus on best feasiable
            solver = getSolver(
                solver,
                mip=True,
                msg=verbose,
                timeLimit=time_limit_s,
                gapRel=relative_gap,
                threads=threads,
            )

            # Customer params. They have being selected running a benchmark for the model
            # it probably can be improved with more experiments.
            solver.optionsDict["Method"] = 0
            solver.optionsDict["Heuristics"] = 0.5
            solver.optionsDict["MIPFocus"] = 2
            solver.optionsDict["PrePasses"] = 2

            self.status = self.lp_model.solve(solver)
            self.status = GUROBI_STATUS_MAPPING.get(
                solver.model.Status, SolverStatus.UNKNOWN
            )

            logging.debug(
                f"Optimization finished with status {self.status} ({GUROBI_STATUS_MAPPING.get(self.status, 'Unknown')})"
            )
        else:
            solver = getSolver(
                solver,
                mip=True,
                msg=verbose,
                timeLimit=time_limit_s,
                gapRel=relative_gap,
            )
            self.status = self.lp_model.solve(solver)
            self.status = (
                SolverStatus.SOLVED_OPTIMAL
                if self.status == LpSolutionOptimal
                else SolverStatus.NOT_SOLVED
            )
        return self.build_results()

    def dump_rest_2(self) -> None:
        """
        Save Rest 2 equation in XLSX for analysis.
        """
        equations_vals = []
        setups = []
        for product in self.products:
            for period in self.periods:
                x_val = self.X[_secure_label(product)][period["label"]].varValue

                tmp_setups = []
                for product_from in self.products:
                    if product_from != product:
                        tmp_setups.append(
                            {
                                "period": period["label"],
                                "product_from": product_from,
                                "product_to": product,
                                "val": self.Z[_secure_label(product)][
                                    _secure_label(product_from)
                                ][period["label"]].varValue,
                            }
                        )
                setup_sum = sum([s["val"] for s in tmp_setups])
                setup_carry_over = self.ZA[_secure_label(product)][
                    period["label"]
                ].varValue
                equations_vals.append(
                    {
                        "period": period["label"],
                        "product": product,
                        "x": x_val,
                        "setup_sum": setup_sum,
                        "setup_carry_over": setup_carry_over,
                    }
                )

                setups = setups + tmp_setups

        pd.DataFrame(equations_vals).to_excel("rest_2.xlsx")
        pd.DataFrame(setups).to_excel("setups.xlsx")

    def dump_rest_14(self) -> None:
        prod_need_lookup = (
            self.prod_need[["sku", "period_label", "prod_need"]]
            .groupby(by=["sku", "period_label"])
            .sum(numeric_only=True)
            .sort_index()
        )
        prods = []
        for product in self.products:
            tmp_total_product_demand = round(
                float(
                    prod_need_lookup.loc[
                        (_from_secured_label(_secure_label(product))), "prod_need"
                    ].sum()
                ),
                DEFAULT_PRECISION,
            )

            prod_sum = sum(
                [
                    self.X[_secure_label(product)][period["label"]].varValue
                    for period in self.periods[1:]
                ]
            )
            prods.append(
                {
                    "product": product,
                    "production": prod_sum,
                    "total_demand_product": tmp_total_product_demand,
                }
            )

        pd.DataFrame(prods).to_excel("rest_14.xlsx")
