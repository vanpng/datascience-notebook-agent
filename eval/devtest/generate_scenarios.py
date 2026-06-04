#!/usr/bin/env python
"""Generate the curated dev/test scenario set for the DS Notebook Agent.

Prompt-leakage prevention (dev/test protocol)
---------------------------------------------
To tune the agent and improve performance WITHOUT contaminating the public
benchmarks (DS-1000, DABStep, DSBench, DSCodeBench), we maintain an independent
hand-curated set of 50 simple, general data-science / analytics scenarios.

These scenarios are NOT sourced from any benchmark in the evaluation suite.
Every scenario is fully self-contained: the data is created inline with small,
deterministic synthetic tables (no external files, no random seeds), so the
ground-truth answer is reproducible and the agent works against the same
notebook context a real user would have after loading and inspecting data.

This script:
  1. Defines 6 small synthetic datasets (created inline in seed cells).
  2. Defines 50 natural-language scenarios over those datasets.
  3. Computes the ground-truth answer for each answer-based scenario by running
     a reference solution locally.
  4. Writes eval/devtest/scenarios.json consumed by the DevTestEvaluator.

Run:  uv run python eval/devtest/generate_scenarios.py
"""
from __future__ import annotations

import io
import json
import contextlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "scenarios.json"


# ── synthetic datasets (inline, deterministic) ────────────────────────────────
# Each dataset is a (creation, inspect) pair. The seed cell sent to the agent is
# creation + inspect (so the agent sees columns + a head() dump in stdout, just
# like a real notebook). Ground-truth is computed by running creation + a
# reference solution.

DATASETS: dict[str, dict] = {
    "sales": {
        "create": (
            "import pandas as pd\n"
            "sales = pd.DataFrame({\n"
            "    'region': ['North','South','North','East','West','South','North','East','West','South','North','East'],\n"
            "    'product': ['Widget','Gadget','Widget','Gizmo','Widget','Gadget','Gizmo','Widget','Gadget','Gizmo','Widget','Gadget'],\n"
            "    'units': [10, 5, 8, 12, 7, 3, 9, 15, 6, 4, 11, 2],\n"
            "    'price': [2.5, 4.0, 2.5, 3.0, 2.5, 4.0, 3.0, 2.5, 4.0, 3.0, 2.5, 4.0],\n"
            "})\n"
        ),
        "inspect": "print(list(sales.columns))\nprint(sales.head())\n",
        "var": "sales",
    },
    "employees": {
        "create": (
            "import pandas as pd\n"
            "employees = pd.DataFrame({\n"
            "    'name': ['Alice','Bob','Carol','Dave','Eve','Frank','Grace','Heidi','Ivan','Judy'],\n"
            "    'department': ['Eng','Sales','Eng','HR','Sales','Eng','HR','Sales','Eng','HR'],\n"
            "    'salary': [95000, 60000, 105000, 55000, 62000, 99000, 58000, 64000, 110000, 60000],\n"
            "    'age': [30, 45, 38, 29, 50, 41, 35, 28, 47, 33],\n"
            "    'years': [3, 10, 8, 2, 15, 9, 5, 1, 12, 4],\n"
            "})\n"
        ),
        "inspect": "print(list(employees.columns))\nprint(employees.head())\n",
        "var": "employees",
    },
    "students": {
        "create": (
            "import pandas as pd\n"
            "students = pd.DataFrame({\n"
            "    'name': ['Ann','Ben','Cara','Dan','Ella','Finn','Gia','Hugo','Iris','Jack'],\n"
            "    'gender': ['F','M','F','M','F','M','F','M','F','M'],\n"
            "    'math': [78, 85, 92, 60, 73, 88, 95, 50, 81, 67],\n"
            "    'science': [82, 79, 90, 65, 70, 85, 99, 55, 78, 72],\n"
            "    'english': [75, 80, 88, 70, 68, 90, 84, 60, 79, 74],\n"
            "})\n"
        ),
        "inspect": "print(list(students.columns))\nprint(students.head())\n",
        "var": "students",
    },
    "customers": {
        "create": (
            "import pandas as pd\n"
            "import numpy as np\n"
            "customers = pd.DataFrame({\n"
            "    'customer_id': list(range(1, 13)),\n"
            "    'age': [25, np.nan, 34, 45, np.nan, 52, 23, 38, np.nan, 41, 29, 36],\n"
            "    'spend': [120.0, 200.0, np.nan, 340.0, 150.0, 90.0, np.nan, 410.0, 75.0, 260.0, 130.0, np.nan],\n"
            "    'segment': ['A','B','A','C','B','C','A','C','B','A','B','C'],\n"
            "})\n"
        ),
        "inspect": "print(list(customers.columns))\nprint(customers.head())\n",
        "var": "customers",
    },
    "timeseries": {
        "create": (
            "import pandas as pd\n"
            "ts = pd.DataFrame({\n"
            "    'date': pd.date_range('2023-01-01', periods=14, freq='D'),\n"
            "    'visitors': [100, 120, 90, 140, 160, 130, 110, 95, 105, 150, 170, 125, 115, 135],\n"
            "})\n"
        ),
        "inspect": "print(list(ts.columns))\nprint(ts.head())\n",
        "var": "ts",
    },
    "products": {
        "create": (
            "import pandas as pd\n"
            "products = pd.DataFrame({\n"
            "    'product': ['A','B','C','D','E','F','G','H','I','J'],\n"
            "    'category': ['Electronics','Home','Electronics','Toys','Home','Electronics','Toys','Home','Electronics','Toys'],\n"
            "    'price': [299, 49, 199, 25, 89, 599, 15, 120, 350, 30],\n"
            "    'stock': [10, 50, 5, 200, 30, 3, 150, 40, 8, 100],\n"
            "})\n"
        ),
        "inspect": "print(list(products.columns))\nprint(products.head())\n",
        "var": "products",
    },
}


# ── scenarios ──────────────────────────────────────────────────────────────────
# Each scenario:
#   id, dataset, intent, query
#   check: "numeric" | "contains" | "plot"
#   ref:   reference solution code that prints the answer as its LAST line
#          (for numeric/contains); used only to compute ground truth.
#   plot_markers: substrings, any of which marks a plotting call (for "plot").

PLOT_MARKERS = ["plt.show", ".savefig", ".plot(", "sns.", "plt.bar", "plt.hist",
                "plt.scatter", "plt.pie", "plt.boxplot", "plt.plot"]

SCENARIOS: list[dict] = [
    # ── sales (1-12) ──
    dict(id="sales_shape", dataset="sales", intent="exploration",
         query="How many rows and columns does the sales DataFrame have? Print its shape.",
         check="contains", ref="print(sales.shape)"),
    dict(id="sales_total_units", dataset="sales", intent="aggregation",
         query="What is the total number of units sold across all rows? Print the total.",
         check="numeric", ref="print(sales['units'].sum())"),
    dict(id="sales_total_revenue", dataset="sales", intent="aggregation",
         query="Compute revenue as units times price and print the total revenue across all rows.",
         check="numeric", ref="print((sales['units'] * sales['price']).sum())"),
    dict(id="sales_top_region_revenue", dataset="sales", intent="aggregation",
         query="Which region has the highest total revenue (units times price)? Print the region name.",
         check="contains",
         ref="rev = (sales['units']*sales['price']).groupby(sales['region']).sum()\nprint(rev.idxmax())"),
    dict(id="sales_unique_products", dataset="sales", intent="exploration",
         query="How many distinct products are there in the sales data? Print the count.",
         check="numeric", ref="print(sales['product'].nunique())"),
    dict(id="sales_count_units_gt8", dataset="sales", intent="filtering",
         query="How many rows have more than 8 units sold? Print the count.",
         check="numeric", ref="print((sales['units'] > 8).sum())"),
    dict(id="sales_avg_units_widget", dataset="sales", intent="filtering",
         query="What is the average number of units sold for the product 'Widget'? Print the average.",
         check="numeric", ref="print(sales.loc[sales['product']=='Widget','units'].mean())"),
    dict(id="sales_mean_price", dataset="sales", intent="exploration",
         query="What is the mean price across all sales rows? Print the mean.",
         check="numeric", ref="print(sales['price'].mean())"),
    dict(id="sales_top_product_units", dataset="sales", intent="aggregation",
         query="Which product has the highest total units sold? Print the product name.",
         check="contains", ref="print(sales.groupby('product')['units'].sum().idxmax())"),
    dict(id="sales_count_gadget", dataset="sales", intent="exploration",
         query="How many sales rows are for the product 'Gadget'? Print the count.",
         check="numeric", ref="print((sales['product']=='Gadget').sum())"),
    dict(id="sales_bar_units_region", dataset="sales", intent="visualization",
         query="Plot a bar chart of total units sold by region.",
         check="plot"),
    dict(id="sales_hist_units", dataset="sales", intent="visualization",
         query="Plot a histogram of the units column.",
         check="plot"),

    # ── employees (13-22) ──
    dict(id="emp_avg_salary", dataset="employees", intent="aggregation",
         query="What is the average salary across all employees? Print the average.",
         check="numeric", ref="print(employees['salary'].mean())"),
    dict(id="emp_highest_paid", dataset="employees", intent="exploration",
         query="Which employee has the highest salary? Print their name.",
         check="contains", ref="print(employees.loc[employees['salary'].idxmax(),'name'])"),
    dict(id="emp_top_dept_salary", dataset="employees", intent="aggregation",
         query="Which department has the highest average salary? Print the department name.",
         check="contains", ref="print(employees.groupby('department')['salary'].mean().idxmax())"),
    dict(id="emp_count_eng", dataset="employees", intent="exploration",
         query="How many employees are in the 'Eng' department? Print the count.",
         check="numeric", ref="print((employees['department']=='Eng').sum())"),
    dict(id="emp_corr_age_years", dataset="employees", intent="statistics",
         query="What is the Pearson correlation between age and years of tenure? Print the correlation rounded to 2 decimals.",
         check="numeric", ref="print(round(employees['age'].corr(employees['years']), 2))"),
    dict(id="emp_count_over40", dataset="employees", intent="filtering",
         query="How many employees are older than 40? Print the count.",
         check="numeric", ref="print((employees['age'] > 40).sum())"),
    dict(id="emp_median_salary", dataset="employees", intent="statistics",
         query="What is the median salary of all employees? Print the median.",
         check="numeric", ref="print(employees['salary'].median())"),
    dict(id="emp_total_sales_salary", dataset="employees", intent="filtering",
         query="What is the total salary paid to employees in the 'Sales' department? Print the total.",
         check="numeric", ref="print(employees.loc[employees['department']=='Sales','salary'].sum())"),
    dict(id="emp_bar_salary_dept", dataset="employees", intent="visualization",
         query="Plot a bar chart of the average salary by department.",
         check="plot"),
    dict(id="emp_scatter_age_salary", dataset="employees", intent="visualization",
         query="Create a scatter plot of age versus salary.",
         check="plot"),

    # ── students (23-32) ──
    dict(id="stu_avg_math", dataset="students", intent="aggregation",
         query="What is the average math score of all students? Print the average.",
         check="numeric", ref="print(students['math'].mean())"),
    dict(id="stu_top_math", dataset="students", intent="exploration",
         query="Which student has the highest math score? Print their name.",
         check="contains", ref="print(students.loc[students['math'].idxmax(),'name'])"),
    dict(id="stu_max_total", dataset="students", intent="preprocessing",
         query="Add a 'total' column that sums math, science and english, then print the highest total score.",
         check="numeric", ref="print((students['math']+students['science']+students['english']).max())"),
    dict(id="stu_female_avg_math", dataset="students", intent="aggregation",
         query="What is the average math score of the female students (gender 'F')? Print the average.",
         check="numeric", ref="print(students.loc[students['gender']=='F','math'].mean())"),
    dict(id="stu_count_sci90", dataset="students", intent="filtering",
         query="How many students scored at least 90 in science? Print the count.",
         check="numeric", ref="print((students['science'] >= 90).sum())"),
    dict(id="stu_corr_math_sci", dataset="students", intent="statistics",
         query="What is the correlation between math and science scores? Print it rounded to 2 decimals.",
         check="numeric", ref="print(round(students['math'].corr(students['science']), 2))"),
    dict(id="stu_median_english", dataset="students", intent="statistics",
         query="What is the median english score? Print the median.",
         check="numeric", ref="print(students['english'].median())"),
    dict(id="stu_count_math_below70", dataset="students", intent="filtering",
         query="How many students have a math score below 70? Print the count.",
         check="numeric", ref="print((students['math'] < 70).sum())"),
    dict(id="stu_hist_math", dataset="students", intent="visualization",
         query="Plot a histogram of the math scores.",
         check="plot"),
    dict(id="stu_box_subjects", dataset="students", intent="visualization",
         query="Create a box plot comparing the math, science and english score distributions.",
         check="plot"),

    # ── customers with missing values (33-40) ──
    dict(id="cust_missing_age", dataset="customers", intent="exploration",
         query="How many missing values are in the age column? Print the count.",
         check="numeric", ref="print(customers['age'].isna().sum())"),
    dict(id="cust_missing_total", dataset="customers", intent="exploration",
         query="What is the total number of missing values across the whole DataFrame? Print the total.",
         check="numeric", ref="print(int(customers.isna().sum().sum()))"),
    dict(id="cust_mean_spend", dataset="customers", intent="statistics",
         query="What is the mean spend, ignoring missing values? Print the mean.",
         check="numeric", ref="print(customers['spend'].mean())"),
    dict(id="cust_fillna_age_mean", dataset="customers", intent="preprocessing",
         query="Fill the missing ages with the mean age, then print the average age after filling.",
         check="numeric", ref="m = customers['age'].mean()\nprint(customers['age'].fillna(m).mean())"),
    dict(id="cust_count_segA", dataset="customers", intent="exploration",
         query="How many customers are in segment 'A'? Print the count.",
         check="numeric", ref="print((customers['segment']=='A').sum())"),
    dict(id="cust_top_spend_segment", dataset="customers", intent="aggregation",
         query="Which segment has the highest average spend (ignoring missing values)? Print the segment label.",
         check="contains", ref="print(customers.groupby('segment')['spend'].mean().idxmax())"),
    dict(id="cust_dropna_rows", dataset="customers", intent="preprocessing",
         query="Drop all rows that contain any missing value and print how many rows remain.",
         check="numeric", ref="print(len(customers.dropna()))"),
    dict(id="cust_bar_spend_segment", dataset="customers", intent="visualization",
         query="Plot a bar chart of the average spend by segment.",
         check="plot"),

    # ── timeseries (41-46) ──
    dict(id="ts_total_visitors", dataset="timeseries", intent="aggregation",
         query="What is the total number of visitors over the whole period? Print the total.",
         check="numeric", ref="print(ts['visitors'].sum())"),
    dict(id="ts_peak_day", dataset="timeseries", intent="exploration",
         query="On which date was the visitor count the highest? Print the date.",
         check="contains", ref="print(ts.loc[ts['visitors'].idxmax(),'date'].date())"),
    dict(id="ts_avg_visitors", dataset="timeseries", intent="statistics",
         query="What is the average daily number of visitors? Print the average.",
         check="numeric", ref="print(ts['visitors'].mean())"),
    dict(id="ts_rolling7_last", dataset="timeseries", intent="preprocessing",
         query="Compute a 7-day rolling mean of visitors and print the last (most recent) rolling-mean value.",
         check="numeric", ref="print(ts['visitors'].rolling(7).mean().iloc[-1])"),
    dict(id="ts_count_gt130", dataset="timeseries", intent="filtering",
         query="How many days had more than 130 visitors? Print the count.",
         check="numeric", ref="print((ts['visitors'] > 130).sum())"),
    dict(id="ts_line_visitors", dataset="timeseries", intent="visualization",
         query="Plot a line chart of visitors over time.",
         check="plot"),

    # ── products (47-50) ──
    dict(id="prod_inventory_value", dataset="products", intent="aggregation",
         query="Compute the total inventory value as price times stock summed over all products. Print the total.",
         check="numeric", ref="print((products['price'] * products['stock']).sum())"),
    dict(id="prod_top_category_count", dataset="products", intent="exploration",
         query="Which category has the most products? Print the category name.",
         check="contains", ref="print(products['category'].value_counts().idxmax())"),
    dict(id="prod_top_avg_price_cat", dataset="products", intent="aggregation",
         query="Which category has the highest average price? Print the category name.",
         check="contains", ref="print(products.groupby('category')['price'].mean().idxmax())"),
    dict(id="prod_pie_category", dataset="products", intent="visualization",
         query="Create a pie chart of the number of products per category.",
         check="plot"),
]


def _run(code: str) -> str:
    """Exec code in a fresh namespace, return captured stdout (stripped)."""
    buf = io.StringIO()
    ns: dict = {}
    with contextlib.redirect_stdout(buf):
        exec(code, ns)
    return buf.getvalue().strip()


def main() -> None:
    assert len(SCENARIOS) == 50, f"expected 50 scenarios, got {len(SCENARIOS)}"
    ids = [s["id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids)), "duplicate scenario ids"

    problems = []
    for s in SCENARIOS:
        ds = DATASETS[s["dataset"]]
        seed_source = ds["create"] + ds["inspect"]
        seed_stdout = _run(seed_source)

        meta = {"check": s["check"], "intent": s["intent"], "dataset": s["dataset"],
                # bare dataframe-creation code (no inspect prints) so the evaluator
                # can re-execute the agent's generated cell with notebook semantics.
                "setup_code": ds["create"]}
        if s["check"] == "plot":
            meta["plot_markers"] = PLOT_MARKERS
            meta["expected"] = "<plot>"
        else:
            expected = _run(ds["create"] + s["ref"]).strip()
            # take the LAST non-empty line as the canonical answer
            expected = [ln for ln in expected.splitlines() if ln.strip()][-1].strip()
            meta["expected"] = expected

        problems.append({
            "id": s["id"],
            "query": s["query"],
            "seed_cells": [{"source": seed_source, "stdout": seed_stdout, "success": True}],
            "metadata": meta,
        })

    OUT.write_text(json.dumps({"scenarios": problems}, indent=2))
    print(f"Wrote {len(problems)} scenarios → {OUT}")
    print("\nGround-truth answers (answer-based scenarios):")
    for p in problems:
        if p["metadata"]["check"] != "plot":
            print(f"  {p['id']:<28} [{p['metadata']['check']:<8}] → {p['metadata']['expected']}")
    n_plot = sum(1 for p in problems if p["metadata"]["check"] == "plot")
    print(f"\nplot scenarios: {n_plot} | answer scenarios: {len(problems)-n_plot}")


if __name__ == "__main__":
    main()
