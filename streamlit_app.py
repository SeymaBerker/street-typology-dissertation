import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(
    page_title="Street Typology Explorer",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #FAF9F7; }
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .result-card {
        background: white;
        border: 1px solid #E8E6E2;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        cursor: pointer;
    }
    .result-card:hover {
        border-color: #378ADD;
    }
</style>
""", unsafe_allow_html=True)

TYPOLOGY_COLOURS = {
    'Active Commercial High Street': '#BA7517',
    'Arterial Movement Corridor': '#378ADD',
    'Local Mixed Street': '#888780',
    'Institutional Large-Block Street': '#D85A30',
    'Winding Historic Lane': '#1D9E75',
    'Pedestrian-Rich Street': '#9B59B6',
    'Dense Local Street': '#2C3E50',
    'Active Boulevard': '#8E44AD',
    'Winding Connector Lane': '#27AE60',
}

TYPOLOGY_QUERIES = {
    'Active Commercial High Street': [60, 25, 800, 35, 3, 1.0, 1.4, 12, 4, 4.5],
    'Arterial Movement Corridor':    [120, 15, 2000, 8, 5, 1.0, 0.8, 2, 2, 5.5],
    'Local Mixed Street':            [55, 16, 1200, 15, 2, 1.0, 1.1, 4, 3, 4.5],
    'Institutional Large-Block Street': [40, 8, 2500, 6, 2, 1.0, 0.7, 1, 1, 4.0],
    'Winding Historic Lane':         [40, 10, 1200, 5, 1, 2.5, 0.5, 1, 2, 3.0]
}

FEATURE_COLS = [
    'length_m', 'building_count_50m', 'avg_building_area',
    'poi_count_50m', 'highway_rank', 'sinuosity',
    'poi_diversity', 'food_drink_count',
    'street_furniture_count', 'avg_connectivity'
]

@st.cache_data
def load_data():
    # Load London with geometry
    london_geo = gpd.read_file("/home/jovyan/work/data/london_streets.gpkg")
    london_geo = london_geo.to_crs(epsg=4326)

    london = pd.read_csv("/home/jovyan/work/data/london_final_features.csv")
    london = london.merge(
        london_geo[['u', 'v', 'key', 'name', 'geometry']],
        on=['u', 'v', 'key'], how='left'
    )
    london_name_map = {
        'Active Commercial Core': 'Active Commercial High Street',
        'Moderate Mixed Street': 'Local Mixed Street',
        'Institutional Large-Block Street': 'Institutional Large-Block Street',
        'Winding Historic Lane': 'Winding Historic Lane',
        'Pedestrian-Rich Street': 'Pedestrian-Rich Street'
    }
    london['typology'] = london['typology'].map(london_name_map)
    london['city'] = 'London'

    # Load other cities with geometry
    barcelona = gpd.read_file("/home/jovyan/work/data/barcelona_final.gpkg").reset_index()
    barcelona = barcelona.to_crs(epsg=4326)
    barcelona['city'] = 'Barcelona'

    singapore = gpd.read_file("/home/jovyan/work/data/singapore_final.gpkg").reset_index()
    singapore = singapore.to_crs(epsg=4326)
    singapore['city'] = 'Singapore'

    tokyo = gpd.read_file("/home/jovyan/work/data/tokyo_final.gpkg").reset_index()
    tokyo = tokyo.to_crs(epsg=4326)
    tokyo['city'] = 'Tokyo'

    # Combine — keep geometry
    cols_needed = FEATURE_COLS + ['typology', 'city', 'name', 'geometry']

    london_gdf = gpd.GeoDataFrame(london, geometry='geometry', crs='EPSG:4326')

    all_streets = pd.concat([
        london_gdf[cols_needed],
        barcelona[cols_needed],
        singapore[cols_needed],
        tokyo[cols_needed]
    ], ignore_index=True)

    all_streets['name'] = all_streets['name'].fillna('Unnamed Street')

    def clean_name(name):
        s = str(name)
        if s.startswith('['):
            return s.strip("[]'\"").split("',")[0].strip("' ")
        return name
    all_streets['name'] = all_streets['name'].apply(clean_name)

    # Build KNN on features only
    X = all_streets[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(X_scaled)

    return all_streets, scaler, knn

with st.spinner("Loading street database..."):
    all_streets, scaler, knn = load_data()

# ── Header ─────────────────────────────────────────────────
st.markdown("### Street Typology Explorer")
st.caption("UCL CASA · MSc Urban Spatial Science 2026 · Foster + Partners · 5,010 segments across London, Barcelona, Singapore and Tokyo")
st.divider()

col_left, col_right = st.columns([1, 1.8])

with col_left:
    st.markdown("**Select typology**")
    selected = st.selectbox(
        "Typology",
        list(TYPOLOGY_QUERIES.keys()),
        label_visibility="collapsed"
    )

    colour = TYPOLOGY_COLOURS.get(selected, '#888780')
    st.markdown(f"""
    <div style="background:{colour}15; border-left:3px solid {colour};
    border-radius:0 6px 6px 0; padding:8px 12px; margin:8px 0 12px 0;">
    <span style="font-size:12px; font-weight:600; color:{colour};">{selected}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Database coverage**")
    for city, count in all_streets['city'].value_counts().items():
        st.markdown(
            f"<span style='font-size:12px;'>{city} — {count:,} segments</span>",
            unsafe_allow_html=True
        )

    st.divider()
    search = st.button(
        "Find Similar Streets",
        type="primary",
        use_container_width=True
    )

with col_right:
    if search:
        with st.spinner("Searching 5,010 street segments..."):
            query = np.array(TYPOLOGY_QUERIES[selected]).reshape(1, -1)
            query_scaled = scaler.transform(query)
            distances, indices = knn.kneighbors(query_scaled)

            results = []
            for i, idx in enumerate(indices[0][1:6]):
                row = all_streets.iloc[idx]
                similarity = round((1 - distances[0][i+1]) * 100, 1)
                geom = row.get('geometry', None)

                # Get centroid coordinates from actual geometry
                lat, lon = None, None
                if geom is not None and not (isinstance(geom, float) and np.isnan(geom)):
                    try:
                        centroid = geom.centroid
                        lat = centroid.y
                        lon = centroid.x
                    except:
                        pass

                results.append({
                    'rank': i + 1,
                    'name': row['name'],
                    'city': row['city'],
                    'typology': row['typology'],
                    'similarity': similarity,
                    'lat': lat,
                    'lon': lon,
                    'geometry': geom
                })

        st.markdown(f"**Top 5 results for: {selected}**")

        # Result cards
        for r in results:
            tcol = TYPOLOGY_COLOURS.get(r['typology'], '#888780')
            bar_color = (
                '#1D9E75' if r['similarity'] > 93
                else '#BA7517' if r['similarity'] > 85
                else '#378ADD'
            )
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:11px; color:#AAAAAA; font-weight:700;">#{r['rank']}</span>
                        <span style="font-size:14px; font-weight:700; color:#1A1916;
                        margin-left:6px;">{r['name']}</span><br>
                        <span style="font-size:12px; color:#666462;">{r['city']}</span>
                        <span style="background:{tcol}20; color:{tcol}; padding:1px 7px;
                        border-radius:4px; font-size:10px; font-weight:600;
                        margin-left:6px;">{r['typology']}</span>
                    </div>
                    <span style="font-size:15px; font-weight:700;
                    color:{bar_color};">{r['similarity']}%</span>
                </div>
                <div style="height:4px; background:#EEECE8; border-radius:2px; margin-top:8px;">
                    <div style="width:{int(r['similarity'])}%; height:100%;
                    background:{bar_color}; border-radius:2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Map with actual street geometries
        st.markdown("**Street locations — click a marker for details**")

        # Centre map on first result
        first = results[0]
        if first['lat'] and first['lon']:
            center = [first['lat'], first['lon']]
            zoom = 14
        else:
            center = [30, 20]
            zoom = 2

        m = folium.Map(
            location=center,
            zoom_start=zoom,
            tiles='CartoDB positron'
        )

        city_colours_map = {
            'London': '#378ADD',
            'Barcelona': '#BA7517',
            'Singapore': '#1D9E75',
            'Tokyo': '#D85A30'
        }

        for r in results:
            c = city_colours_map.get(r['city'], '#888780')
            geom = r['geometry']

            if geom is not None and not (isinstance(geom, float) and np.isnan(geom)):
                try:
                    # Draw the actual street geometry
                    geo_json = gpd.GeoSeries([geom]).__geo_interface__
                    folium.GeoJson(
                        geo_json,
                        style_function=lambda x, color=c: {
                            'color': color,
                            'weight': 5,
                            'opacity': 0.9
                        },
                        tooltip=f"#{r['rank']} {r['name']} — {r['city']} — {r['similarity']}%",
                        popup=folium.Popup(
                            f"<b>#{r['rank']} {r['name']}</b><br>"
                            f"{r['city']}<br>"
                            f"{r['typology']}<br>"
                            f"{r['similarity']}% match",
                            max_width=200
                        )
                    ).add_to(m)

                    # Also add a circle marker at centroid for visibility
                    if r['lat'] and r['lon']:
                        folium.CircleMarker(
                            location=[r['lat'], r['lon']],
                            radius=6,
                            color=c,
                            fill=True,
                            fill_opacity=0.9,
                            tooltip=f"#{r['rank']} {r['name']}"
                        ).add_to(m)
                except Exception:
                    pass

        st_folium(m, width=700, height=280, returned_objects=[])

    else:
        st.info("Select a typology from the left panel and click **Find Similar Streets**.")

st.divider()
st.caption(
    "Data: OpenStreetMap contributors (2024), accessed via osmnx (Boeing, 2017). "
    "Method: K-nearest neighbours with cosine similarity across 10 spatial features "
    "following the TfL Street Family framework (Transport for London, 2017)."
)