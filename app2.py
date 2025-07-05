import streamlit as st
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
import random
from streamlit_folium import st_folium
import os
from pathlib import Path
import plotly.graph_objects as go


# App configuration
st.set_page_config(
    page_title="Cosmic Radiation Research Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("Cosmic Radiation Research Dashboard")

# Intro section on homepage
st.markdown("""
Welcome to the **Cosmic Radiation Research Dashboard** — an interactive platform to explore real-time and simulated data on cosmic rays, their biological and technological effects, and mission safety.

---

**Select a feature tab below to begin your research:**
""")

# Main Feature Tabs
tabs = st.tabs([
    "Radiation Risk Calculator",
    "Live Cosmic Ray Shower Map",
    "Biological Effects Visualizer",
    "Effects on Electronics",
    "Cosmic Ray Data Explorer",
    "Mission Dose Comparator",
    "Space Weather Live",
    "Research Library",
    "Upload & Analyze Your Data"
])

# ========== TAB 1: Radiation Risk Calculator ==========
with tabs[0]:
    st.subheader("Radiation Risk Calculator")
    st.info("This tool estimates the radiation dose and cancer risk for a space mission based on real-time solar particle flux and selected shielding.")
    mission_days = st.slider("Mission Duration (days)", 1, 1000, 180)
    shielding_material = st.selectbox("Shielding Material", ["None", "Aluminum", "Polyethylene"])

    url = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data)
        df['time_tag'] = pd.to_datetime(df['time_tag'])
        df['flux'] = pd.to_numeric(df['flux'], errors='coerce')
        flux=df['flux'].iloc[-1]
        st.success(f"Live Proton Flux (≥10 MeV): {flux:.2e} protons/cm²/s/sr")
    except:
        flux = 100
        st.warning("Unable to fetch live data. Using default flux: 100 p/cm²/s/sr")

    base_dose_per_day = flux * 0.00005
    shield_factors = {'None': 1.0, 'Aluminum': 0.7, 'Polyethylene': 0.5}
    daily_dose = base_dose_per_day * shield_factors[shielding_material]
    total_dose = daily_dose * mission_days
    risk_percent = (total_dose / 1000) * 5

    st.metric("☢️ Estimated Total Dose (mSv)", f"{total_dose:.2f}")
    st.metric("⚠️ Estimated Cancer Risk", f"{risk_percent:.2f} %")
    st.caption("ICRP model: 5% risk increase per 1 Sv of exposure. Not for clinical use.")

    st.subheader("Dose Accumulation Over Time")
    days = np.arange(1, mission_days + 1)
    dose_over_time = daily_dose * days
    fig, ax = plt.subplots()
    ax.plot(days, dose_over_time, color='crimson')
    ax.set_xlabel("Days")
    ax.set_ylabel("Cumulative Dose (mSv)")
    ax.set_title("Radiation Dose Accumulation")
    st.pyplot(fig)

    st.subheader("Monte Carlo Simulation (1000 Astronauts)")
    simulated_doses = np.random.normal(loc=total_dose, scale=0.1 * total_dose, size=1000)
    fig2, ax2 = plt.subplots()
    ax2.hist(simulated_doses, bins=30, color='orange', edgecolor='black')
    ax2.set_title("Simulated Dose Distribution")
    ax2.set_xlabel("Total Dose (mSv)")
    ax2.set_ylabel("Number of Astronauts")
    st.pyplot(fig2)

    st.subheader("Shielding Material Effectiveness")
    df = pd.DataFrame({"Material": ["None", "Aluminum", "Polyethylene"], "Approx. Dose Reduction (%)": [0, 30, 50]})
    st.dataframe(df)

# ========== TAB 2: Live Cosmic Ray Shower Map ==========
with tabs[1]:
    st.subheader("Live Cosmic Ray Shower Map")
   
    st.info("Map currently shows **mock shower data**. Live data from observatories coming soon!")
    data = pd.read_csv("data/mock_cosmic_shower_data.csv")
    data["date"] = pd.to_datetime(data["date"])

    # ======UI Filters======
    st.markdown("### 🔍 Filter Shower Events")

    intensity_filter = st.multiselect(
        "Select intensity levels to display",
        options=["Low", "Moderate", "High"],
        default=["Low", "Moderate", "High"]
    )

    hemisphere_filter = st.selectbox(
        "Choose Hemisphere",
        options=["Both", "Northern", "Southern"]
    )

    min_date, max_date = data["date"].min(), data["date"].max()
    date_range = st.date_input("Select date range", [min_date, max_date])

    # =====Filter data=====
    
    filtered_data = data[data["intensity"].isin(intensity_filter)]

    if hemisphere_filter == "Northern":
        filtered_data = filtered_data[filtered_data["latitude"] > 0]
    elif hemisphere_filter == "Southern":
        filtered_data = filtered_data[filtered_data["latitude"] < 0]

    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_data = filtered_data[
        (filtered_data["date"] >= start_date) & (filtered_data["date"] <= end_date)
    ]

    # =====Create the map====
    m = folium.Map(
        location=[20, 0],
        zoom_start=2,
        tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL."
    )

    # =====Plot filtered markers=====
    color_map = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}

    for _, row in filtered_data.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=f"{row['date'].date()} — Intensity: {row['intensity']}",
            color=color_map[row["intensity"]],
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    # ====Show the map=====
    st_folium(m, width=700)

    # Legend + caption
    with st.expander("🗺️ Legend"):
        st.markdown("""
        - 🟢 **Low Intensity**
        - 🟠 **Moderate Intensity**
        - 🔴 **High Intensity**
        """)

    st.caption("Mock cosmic ray data used for demonstration. Real detector-based updates will be integrated soon.")

  #  m = folium.Map(location=[20, 0], zoom_start=2, tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png",
   #                attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.")
  #  for _ in range(25):
   #     lat, lon = random.uniform(-60, 60), random.uniform(-180, 180)
   #     intensity = random.choice(['Low', 'Moderate', 'High'])
   #     color = {'Low': 'green', 'Moderate': 'orange', 'High': 'red'}[intensity]
    #    folium.CircleMarker(location=[lat, lon], radius=6, popup=f"Shower\nIntensity: {intensity}", color=color,
   #                         fill=True, fill_opacity=0.7).add_to(m)
 #   st_folium(m, width=700)
 #   st.caption("Simulated data. Future version will include real-time showers from cosmic ray arrays.")

# =====================Tab 3: Biological Effects======================
with tabs[2]:
    import os
    from pathlib import Path
    st.subheader("Biological Effects of Radiation over Time")

    # Separator
    st.write("---")
    st.subheader("Customize for Individual Factors and Duration")

    # Age and Gender Inputs
    age = st.slider("Select Age (Years)", 0, 100, 30)
    gender = st.selectbox("Select Gender", ["Male", "Female", "Prefer not to say"])

    # Duration Input (days)
    days = st.slider("Select Duration (Days)", 0, 36500, 30)

    # Base daily cosmic radiation rate (mSv/day) — average cosmic component at sea level (~0.38 mSv/year ⇒ ~0.00104 mSv/day)
    BASE_RATE = 0.00104  # Source: UNSCEAR 2020 report

    # Calculate raw dose over period
    raw_dose = days * BASE_RATE

    # Calculate modifiers
    age_modifier = 1.0
    gender_modifier = 1.0
    if age < 10:
        age_modifier = 1.4
        st.warning("Children under 10 are more radiosensitive; this is factored into the adjusted dose.")
    elif age < 20:
        age_modifier = 1.2
        st.info("Individuals under 20 have increased sensitivity; applied to adjusted dose.")
    elif age > 60:
        age_modifier = 0.9
        st.info("Older adults may have slightly lower long-term cancer risk; applied to adjusted dose.")

    if gender == "Female":
        gender_modifier = 1.1
        st.info("Female sensitivity modifier applied to adjusted dose.")
    elif gender == "Male":
        gender_modifier = 1.0
        st.info("Male baseline sensitivity modifier applied.")
    else:
        st.info("Using general sensitivity modifiers without specific gender differentiation.")

    # Adjust dose
    adjusted_dose = raw_dose * age_modifier * gender_modifier
    st.markdown(f"***Adjusted Cumulative Dose over {days} days: {adjusted_dose:.2f} mSv***")

    # Determine effect and image
    if adjusted_dose < 1:
        effect = "No observable effects."
        img_file = "human_body_healthy.png"
    elif adjusted_dose < 5:
        effect = "Minor biological impact."
        img_file = "human_body_minor_damage.png"
    elif adjusted_dose < 15:
        effect = "Mild ARS possible."
        img_file = "human_body_moderate_damage.png"
    elif adjusted_dose < 30:
        effect = "Severe ARS symptoms."
        img_file = "human_body_severe_damage.png"
    else:
        effect = "Potentially life-threatening dose."
        img_file = "human_body_critical_damage.png"

    st.info(f"Biological Effect: **{effect}** at {adjusted_dose:.2f} mSv")

    # Image loading: ensure correct path
    script_dir = Path(__file__).parent
    image_dir = script_dir / "images"
    image_path = image_dir / img_file

    try:
        st.image(str(image_path), caption=f"Impact Visualization ({adjusted_dose:.0f} mSv)", use_column_width=True)
        st.caption("Disclaimer: Conceptual illustration only.")
    except Exception as e:
        st.error(f"Could not load image: {e}\nCheck that 'images' folder exists alongside this script and contains {img_file}.")

    # Interactive Risk chart
    st.subheader("📊 Interactive Risk Severity Chart")

    # Define thresholds and labels
    thresholds = [0, 1, 5, 15, 30, 50]
    labels = ["None", "Minor", "Mild ARS", "Severe ARS", "Lethal", "Extreme/Fatal"]
    colors = ["#2ecc71", "#f1c40f", "#f39c12", "#e67e22", "#e74c3c"]

    # Create Plotly figure
    fig = go.Figure()
    # Background zones
    for i in range(len(thresholds) - 1):
        fig.add_shape(
            type="rect",
            x0=thresholds[i], x1=thresholds[i+1], y0=0, y1=1,
            fillcolor=colors[i], opacity=0.3, layer="below", line_width=0
        )
        fig.add_annotation(
            x=(thresholds[i]+thresholds[i+1]) / 2, y=0.95,
            text=labels[i], showarrow=False, font=dict(size=12), opacity=0.8
        )

    # Plot markers: raw dose and adjusted dose
    fig.add_trace(go.Scatter(
        x=[raw_dose], y=[0.5], mode='markers+text', name='Raw Dose',
        marker=dict(size=12), text=['Raw'], textposition='bottom center'
    ))
    fig.add_trace(go.Scatter(
        x=[adjusted_dose], y=[0.5], mode='markers+text', name='Adjusted Dose',
        marker=dict(size=12), text=['Adjusted'], textposition='top center'
    ))

    fig.update_layout(
        xaxis=dict(title="Dose (mSv)", range=[0, thresholds[-1]]),
        yaxis=dict(visible=False), title="Radiation Dose vs. Biological Risk",
        height=300, margin=dict(t=40, b=40), showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # Table: Organ-specific susceptibility
    st.subheader("Organ Susceptibility (Generalized)")
    df = pd.DataFrame({
        "Organ": ["Bone Marrow", "GI Tract", "Skin", "Brain", "Reproductive Organs"],
        "Effect at ≥50 mSv": [
            "Reduced blood cell count", "Nausea, diarrhea", "Burns, hair loss",
            "Cognitive impairment", "Sterility"
        ]
    })
    st.dataframe(df)
    
# Tab 4: Effects on Electronics
with tabs[3]:
    st.subheader("💻 Effects of Cosmic Radiation on Electronics")

    # Inputs
    duration = st.slider("🕒 Mission Duration (days)", 1, 1000, 180)
    shielding = st.selectbox("🛡️ Shielding Level", ["None", "Light", "Heavy"])
    sensitivity = st.selectbox("📦 Electronics Sensitivity", ["Standard", "Hardened", "Critical"])

    # Sensitivity factors
    sensitivity_factor = {
        "Standard": 1.0,
        "Hardened": 0.5,
        "Critical": 2.0
    }

    # Shielding effectiveness
    shielding_factor = {
        "None": 1.0,
        "Light": 0.6,
        "Heavy": 0.3
    }

    # Base SEU rate per day (mock value)
    base_seu_rate = 0.002  # Ups/day

    # Calculate adjusted SEU rate
    adjusted_rate = base_seu_rate * sensitivity_factor[sensitivity] * shielding_factor[shielding]
    total_seus = adjusted_rate * duration

    # Categorize risk
    if total_seus < 1:
        risk = "Low"
        color = "green"
    elif total_seus < 5:
        risk = "Moderate"
        color = "orange"
    else:
        risk = "High"
        color = "red"

    st.metric("📉 Estimated SEUs", f"{total_seus:.2f}")
    st.success(f"⚠️ Failure Risk Level: {risk}")

    # Visualization: Shielding vs SEU Rate
    import matplotlib.pyplot as plt

    st.subheader("📊 SEU Rate vs Shielding")

    levels = ["None", "Light", "Heavy"]
    rates = [base_seu_rate * sensitivity_factor[sensitivity] * shielding_factor[lev] * duration for lev in levels]

    fig, ax = plt.subplots()
    ax.bar(levels, rates, color=['red', 'orange', 'green'])
    ax.set_ylabel("Total SEUs (bit flips)")
    ax.set_title("Effect of Shielding on SEU Risk")
    st.pyplot(fig)

    # Explanation
    st.markdown("""
    Cosmic rays, particularly high-energy protons and heavy ions, can disrupt electronics in space.  
    These **Single Event Upsets (SEUs)** can cause:
    - Memory bit flips
    - Logic faults
    - Temporary or permanent device failure

    **Radiation hardening** and **shielding** are key to reducing these effects in space missions.
    """)


    #Monte Carlo simulation of 1000 devices and the effect on them
    
    st.subheader("🎲 Monte Carlo Simulation (1000 Devices)")
    simulated_failures = np.random.normal(loc=total_seus, scale=0.2 * total_seus, size=1000)
    simulated_failures = np.clip(simulated_failures, 0, None)  # no negative SEUs

    fig2, ax2 = plt.subplots()
    ax2.hist(simulated_failures, bins=30, color='purple', edgecolor='black')
    ax2.set_title("Simulated SEU Distribution Across Devices")
    ax2.set_xlabel("Total SEUs")
    ax2.set_ylabel("Number of Devices")
    st.pyplot(fig2)

    st.caption("Simulates variation in SEU impact across 1000 similar devices.")


# Tab 5: CR Data Explorer
with tabs[4]:
    import numpy as np
    import matplotlib.pyplot as plt

    st.subheader("📈 Cosmic Ray Data Explorer")
    st.markdown("Explore how different particles behave over energy ranges using mock spectra.")

    # Dropdowns for user input
    source = st.selectbox("🔬 Select Data Source", ["AMS-02", "Voyager 1", "Mock Data"])
    particle = st.selectbox("🧪 Select Particle Type", ["Protons", "Helium Nuclei", "Iron Nuclei"])

    # Generate sample spectra (mock data)
    energy = np.logspace(0.1, 3, 50)  # MeV range
    if particle == "Protons":
        flux = 1e4 * energy**-2.7
    elif particle == "Helium Nuclei":
        flux = 1e3 * energy**-2.6
    else:
        flux = 200 * energy**-2.5

    # Add slight noise if mock data is selected
    if source == "Mock Data":
        flux *= np.random.normal(1.0, 0.05, size=flux.shape)

    # Plotting the spectrum
    fig, ax = plt.subplots()
    ax.loglog(energy, flux, label=f"{particle} Spectrum")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Flux (particles/m²·s·sr·MeV)")
    ax.set_title(f"Cosmic Ray Spectrum - {source}")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()

    st.pyplot(fig)

    # Description
    st.markdown("""
    📡 **Cosmic Ray Spectra** represent the distribution of particle flux over different energies.

    These spectra vary based on:
    - ☀️ Source (e.g., solar, galactic, extragalactic)
    - 🧬 Particle type (proton, helium, iron, etc.)
    - 🌍 Location (Earth orbit, interstellar space, etc.)

    Real data from **AMS-02**, **Voyager**, and **CRDB** can be integrated in future releases.
    """)

# Tab 6: Dose Comparison
with tabs[5]:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    st.subheader("🛰️ Space Mission Radiation Dose Comparator")

    # Predefined missions
    missions = ["ISS (LEO)", "Lunar Orbit", "Lunar Surface", "Mars Transit", "Deep Space"]
    daily_doses = [0.3, 0.5, 1.0, 1.8, 2.5]  # mSv/day (based on NASA data ranges)

    # 🔄 Replacing dropdown with a slider for custom duration
    days = st.slider("🕒 Select Mission Duration (days)", min_value=1, max_value=1000, value=180, step=1)

    # Compute total doses
    total_doses = [dose * days for dose in daily_doses]

    # Display table
    df = pd.DataFrame({
        "Mission": missions,
        "Daily Dose (mSv)": daily_doses,
        f"Total Dose for {days} days (mSv)": total_doses
    })
    st.dataframe(df)

    # Plot
    st.subheader("📊 Total Radiation Dose per Mission")
    fig, ax = plt.subplots()
    bars = ax.bar(missions, total_doses, color="mediumslateblue")
    ax.set_ylabel("Total Dose (mSv)")
    ax.set_title(f"Total Radiation Dose Over {days} Days")
    ax.axhline(1000, color='red', linestyle='--', label="1 Sv Cancer Risk Threshold")
    ax.legend()
    st.pyplot(fig)

    # Summary
    st.markdown(f"""
🔎 **Insights:**
- **LEO (e.g., ISS)** is relatively safe due to Earth's magnetic shielding.
- **Lunar & deep space** missions face **much higher radiation exposure**.
- A **1 Sv dose** is considered to increase lifetime cancer risk by ~5%.

This tool helps in comparing the risk factor across different mission environments.
    """)

# Tab 7: Space Weather
with tabs[6]:
    import requests
    import datetime
    import matplotlib.pyplot as plt

    st.subheader("🌞 Real-Time Space Weather Monitor")

    # --- Proton Flux (≥10 MeV) ---
    st.markdown("### ☢️ Proton Flux (≥10 MeV)")
    try:
        url_proton = "https://services.swpc.noaa.gov/json/goes/primary/integral-protons-3-day.json"
        proton_data = requests.get(url_proton).json()
        times = [datetime.datetime.strptime(p["time_tag"], "%Y-%m-%dT%H:%M:%SZ") for p in proton_data if p["energy"] == ">=10 MeV"]
        fluxes = [float(p["flux"]) for p in proton_data if p["energy"] == ">=10 MeV"]

        fig, ax = plt.subplots()
        ax.plot(times, fluxes, color='red')
        ax.set_title("Proton Flux (GOES - ≥10 MeV)")
        ax.set_ylabel("Flux (protons/cm²·s·sr)")
        ax.set_xlabel("UTC Time")
        ax.grid(True)
        st.pyplot(fig)

        if fluxes[-1] > 100:
            st.warning("⚠️ Elevated proton flux — possible solar event in progress.")
        else:
            st.success("✅ Proton flux is at normal background levels.")
    except:
        st.error("Could not load proton flux data.")

    # --- X-Ray Flux (Solar Flares) ---
    st.markdown("### ⚡ X-Ray Flux (Solar Flares)")
    try:
        url_xray = "https://services.swpc.noaa.gov/json/goes/primary/xrays-3-day.json"
        xray_data = requests.get(url_xray).json()
        x_times = [datetime.datetime.strptime(x["time_tag"], "%Y-%m-%dT%H:%M:%SZ") for x in xray_data]
        short = [float(x["flux"]) for x in xray_data]

        fig, ax = plt.subplots()
        ax.plot(x_times, short, color='orange')
        ax.set_title("X-Ray Short Flux (GOES)")
        ax.set_ylabel("Flux (W/m²)")
        ax.set_xlabel("UTC Time")
        ax.set_yscale("log")
        ax.grid(True)
        st.pyplot(fig)

        if short[-1] > 1e-5:
            st.warning("⚠️ Possible solar flare detected!")
        else:
            st.success("✅ No flare activity at the moment.")
    except:
        st.error("Could not load X-ray data.")

    # --- Kp Index (Geomagnetic Storms) ---
    st.markdown("### 🧭 Kp Index (Geomagnetic Storms)")
    
     # --- Kp Index Plot---
import streamlit as st
import pandas as pd

st.subheader("🧭 Geomagnetic Kp Index (From Local JSON File)")

# ✅ Use escaped path or raw string
kp_file_path = r"C:\Users\zeba\Downloads\planetary_k_index_1m.json"

try:
    # Load JSON file
    kp_data = pd.read_json(kp_file_path)

    # If it's a list of dicts (like NOAA format), convert to DataFrame
    if isinstance(kp_data.iloc[0], dict):
        kp_df = pd.DataFrame(kp_data.tolist())
    else:
        kp_df = kp_data

    # Ensure correct column names (adjust if different)
    kp_df['time_tag'] = pd.to_datetime(kp_df['time_tag'], errors='coerce')
    kp_df['Kp'] = pd.to_numeric(kp_df['Kp'], errors='coerce')

    kp_df = kp_df.dropna(subset=['time_tag', 'Kp'])
    kp_df = kp_df.sort_values('time_tag')

    # Plot
    st.line_chart(kp_df.rename(columns={'time_tag': 'index'}).set_index('index')[['Kp']])

except FileNotFoundError:
    st.error(f"File not found: `{kp_file_path}`. Please check the path.")
except Exception as e:
    st.warning(f"Error loading Kp index data: {e}")

# Tab 8: Research Library
with tabs[7]:
    st.subheader("📚 Research Paper Library")

    st.markdown("""
    Browse handpicked research papers on cosmic rays, radiation health, and space missions.
    """)

    # Example static paper list
    import pandas as pd

    papers = pd.DataFrame({
        "Title": [
            "Comparative study of effects of cosmic rays on the earth’s atmospheric processes ",
            "Beyond Earthly Limits: Protection against Cosmic Radiation through Biological Response Pathways",
            "The effect of cosmic rays on biological systems",
            "Microprocessor technology and single event upset susceptibility",
            "Impact Of Cosmic Rays On Satellite Communications"
        ],
        "Authors": [
            "Arshad Rasheed Ganai and Dr. Suryansh Choudhary",
            "Zahida Sultanova and Saleh Sultansoy",
            "N. K. Belisheva, H. Lammer, H. K. Biernat and E. V. Vashenuyk",
            "L.D. Akers",
            "Dr. Premlal P.D"
        ],
        "Link": [
            "https://www.physicsjournal.in/archives/2020.v2.i1.A.27/comparative-study-of-effects-of-cosmic-rays-on-the-earthrsquos-atmospheric-processes",
            "https://arxiv.org/pdf/2405.12151",
            "https://www.researchgate.net/publication/235958260_The_effect_of_cosmic_rays_on_biological_systems_-_An_investigation_during_GLE_events",
            "https://klabs.org/DEI/References/avionics/small_sat_conference/1996/ldakers.pdf",
            "https://www.iosrjournals.org/iosr-jece/papers/Vol.%2019%20Issue%202/Ser-1/D1902013337.pdf"
        ],
        "Year": [2020, 2024, 2012, 1996, 2024],
        "Tags": ["Atmosphere", "Biology", "Biology", "Electronics", "Electronics"]
    })

    tag = st.selectbox("Filter by Tag", ["All", "Atmosphere", "Biology", "Electronics"])
    if tag != "All":
        filtered = papers[papers["Tags"] == tag]
    else:
        filtered = papers

    st.dataframe(filtered)

    # Add download example
    st.markdown("### 📎 Example Paper Download")
    st.download_button(
        "Download Example Paper (PDF)",
        data=b"%PDF-1.4 ... (fake content)",
        file_name="example_paper.pdf",
        mime="application/pdf"
    )
# Tab 9: cosmic ray data explorerwith tabs[8]:
with tabs[8]:
    st.subheader("📤 Upload & Analyze Your Own Cosmic Ray Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file (must include 'Energy' and 'Flux' columns, max 2MB)", type=["csv"])
    if uploaded_file is not None:
        if uploaded_file.size > 2 * 1024 * 1024:
            st.error("File too large. Please upload a file smaller than 2MB.")
        else:
            try:
                df = pd.read_csv(uploaded_file)
                if 'Energy' in df.columns and 'Flux' in df.columns:
                    st.success("File uploaded and read successfully!")
                    st.markdown("### 📄 Preview of Uploaded Data")
                    st.dataframe(df.head())
                    log_scale = st.checkbox("Log scale", value=True)
                    fig, ax = plt.subplots()
                    ax.plot(df['Energy'], df['Flux'], marker='o', linestyle='-', color='blue')
                    ax.set_xlabel("Energy")
                    ax.set_ylabel("Flux")
                    ax.set_title("Uploaded Cosmic Ray Spectrum")
                    if log_scale:
                        ax.set_yscale("log")
                        ax.set_xscale("log")
                    ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    st.pyplot(fig)
                else:
                    st.error("CSV must contain 'Energy' and 'Flux' columns.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

# ========== FOOTER ==========
st.markdown(f"""
---
<p style='text-align: center; color: gray'>
Built by Tanmay Rajput | Last updated: {datetime.datetime.now().strftime('%B %d, %Y')}
</p>
""", unsafe_allow_html=True)
