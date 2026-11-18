import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Car Features & MSRP Explorer", page_icon="🚗", layout="wide")
st.title("Car Features & MSRP Explorer")
st.caption("Interactive exploration of vehicle specifications, fuel economy and manufacturer suggested retail prices.")

@st.cache_data
def load_data():
    data = pd.read_csv(Path(__file__).parent / "data.csv")
    data.columns = [c.strip().lower().replace(" ", "_").replace(".", "_") for c in data.columns]
    numeric = ["year","engine_hp","engine_cylinders","number_of_doors","highway_mpg","city_mpg","popularity","msrp"]
    for c in numeric:
        if c in data: data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.drop_duplicates()
    for c in data.select_dtypes(include="number").columns: data[c] = data[c].fillna(data[c].median())
    for c in data.select_dtypes(exclude="number").columns: data[c] = data[c].fillna("Unknown")
    return data

df = load_data()
st.sidebar.header("Filters")
makes = sorted(df["make"].dropna().unique())
selected_makes = st.sidebar.multiselect("Manufacturer", makes, default=makes[:min(8,len(makes))])
year_min, year_max = int(df.year.min()), int(df.year.max())
years = st.sidebar.slider("Model year", year_min, year_max, (year_min, year_max))
styles = sorted(df["vehicle_style"].dropna().unique())
selected_styles = st.sidebar.multiselect("Vehicle style", styles, default=styles)
filtered = df[df.make.isin(selected_makes) & df.year.between(*years) & df.vehicle_style.isin(selected_styles)]
if filtered.empty:
    st.warning("No records match these filters. Adjust the sidebar selections.")
    st.stop()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Vehicles", f"{len(filtered):,}")
c2.metric("Manufacturers", f"{filtered.make.nunique():,}")
c3.metric("Median MSRP", f"${filtered.msrp.median():,.0f}")
c4.metric("Median horsepower", f"{filtered.engine_hp.median():,.0f} HP")

st.subheader("Univariate analysis")
a,b = st.columns(2)
with a:
    st.plotly_chart(px.histogram(filtered,x="msrp",nbins=40,title="MSRP distribution"),use_container_width=True)
with b:
    st.plotly_chart(px.bar(filtered.make.value_counts().head(12).rename_axis("make").reset_index(name="count"),x="make",y="count",title="Most represented manufacturers"),use_container_width=True)
a,b = st.columns(2)
with a:
    st.plotly_chart(px.histogram(filtered,x="vehicle_style",title="Vehicle styles"),use_container_width=True)
with b:
    st.plotly_chart(px.histogram(filtered,x="engine_hp",title="Engine horsepower distribution"),use_container_width=True)

st.subheader("Bivariate analysis")
a,b = st.columns(2)
with a:
    st.plotly_chart(px.scatter(filtered.sample(min(2500,len(filtered)),random_state=42),x="engine_hp",y="msrp",color="transmission_type",hover_data=["make","model"],title="Horsepower vs MSRP"),use_container_width=True)
with b:
    corr=filtered.select_dtypes("number").corr()
    st.plotly_chart(px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",title="Numeric correlation heatmap"),use_container_width=True)
st.plotly_chart(px.box(filtered,x="vehicle_size",y="msrp",color="vehicle_size",title="MSRP comparison by vehicle size"),use_container_width=True)
st.subheader("Key findings")
st.write(f"- Most represented manufacturer: **{filtered.make.value_counts().idxmax()}**.")
st.write(f"- Median listed MSRP: **${filtered.msrp.median():,.0f}**.")
st.write(f"- Horsepower–MSRP correlation: **{filtered[['engine_hp','msrp']].corr().iloc[0,1]:.2f}**.")
if st.checkbox("Show raw data"):
    st.dataframe(filtered, use_container_width=True)
st.caption("Dataset source: CooperUnion, Car Features and MSRP on Kaggle.")
