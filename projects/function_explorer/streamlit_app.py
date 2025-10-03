# streamlit_app.py
import streamlit as st
from function_explorer import FunctionExplorer
from viz_helpers import plot_analysis, plot_value_hist
import pandas as pd
import json

st.title("Function Explorer")
expr = st.text_input("Enter f(x):", value="sin(x)/x")
xmin, xmax = st.slider("x-window", -50, 50, (-10, 10))
samples = st.slider("Samples", 100, 10000, 3000, step=100)

fx = FunctionExplorer(expr)
res = fx.analyze(window=(float(xmin), float(xmax)), samples=int(samples))

st.subheader("Summary")
domain = FunctionExplorer.format_domain(res.domain_intervals)
summary = {
    "Expression": [expr],
    "Domain": [domain],
    "x-intercepts": [json.dumps(res.x_intercepts)],
    "y-intercept": [res.y_intercept],
    "Singularities": [json.dumps(res.singularities)],
    "Critical points (window)": [json.dumps(res.critical_points)],
    "Range estimate [ymin, ymax]": [json.dumps([res.range_estimate[0], res.range_estimate[1]])],
    "Range note": [res.range_estimate[2]],
}
st.dataframe(pd.DataFrame(summary))

st.subheader("Plots")
st.pyplot(plot_analysis(res, title=f"y = {expr}"))
st.pyplot(plot_value_hist(res, bins=60))
