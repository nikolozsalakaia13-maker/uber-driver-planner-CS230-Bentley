"""
CS 230 Final Project - Uber Driver Shift Planner 
Name: Nikoloz Salakaia
Section: 7
Dataset: Uber Pickups in NYC (April 2014 sample)
Description: Interactive data explorer helping NYC Uber drivers identify peak pickup hours,
             expected shift volume, and spatial pickup clusters.
LLMs Used: Gemini
Streamlit Cloud URL: https://uber-driver-planner-cs230-bentley-nikoloz-salakaia.streamlit.app/
"""

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

# --- PAGE SETUP ---
st.set_page_config(page_title="NYC Uber Driver Shift Planner", layout="wide")

DATA_FILE = "uber_nyc_pickups_april2014_sample.csv"
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Color mapping for NYC Uber dispatching bases (used in Q3 when "Show base colors" is checked)
BASE_COLORS = {        # dictionary w values that are lists (RGBA - color format needed for PyDeck)
    "B02512": [231, 76, 60, 180],    # Red (Unter)
    "B02598": [52, 152, 219, 180],   # Blue (Hinter)
    "B02617": [46, 204, 113, 180],   # Green (Weiter)
    "B02682": [155, 89, 182, 180],   # Purple (Schmecken)
    "B02764": [241, 196, 15, 180],   # Gold (Danach-NY)
}
DEFAULT_COLOR = [149, 165, 166, 180]    #list

#[DICT_DRIVEN] Dictionary w text keys and text values that represent HEX codes - color code needed for Altair
CHART_COLORS = {
    "subtitle": "#9AA0A6",       # gray that is readable on both dark and light backgrounds
    "line": "#2E86C1",           # blue trend line (Q2)
    "points": "#1B4F72",         # dark blue circle markers (Q2)
    "average_line": "#E74C3C",   # red dashed average line (Q2)
    "peak_bar": "#FF4B4B",       # coral red for top 3 peak hours (Q1)
    "standard_bar": "#4A90E2",   # blue for regular hours (Q1)
}


# --- 1. DATA LOADING & CLEANING ---
@st.cache_data
def load_and_clean_data(filepath):
    """
    Load the Uber pickups CSV and add the time columns the questions need.
    Input: filepath (string, relative path to the CSV file)
    Output: cleaned DataFrame with Day_of_Week, Hour and Date columns added
    """
    df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Parse Date/Time column into datetime objects
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], errors="coerce")
    df = df.dropna(subset=["Date/Time", "Lat", "Lon"])

    # Clean Base column if present so color lookups match cleanly
    if "Base" in df.columns:
        df["Base"] = df["Base"].astype(str).str.strip().str.upper()

    # Extract helpful time columns for all three questions
    df["Day_of_Week"] = df["Date/Time"].dt.day_name()
    df["Hour"] = df["Date/Time"].dt.hour
    df["Date"] = df["Date/Time"].dt.date

    return df



# --- 2. #[REUSE_FUNC] One filter used by Q1, Q2 and Q3. ---
def filter_pickups(df, day, hour_range=(0, 23)):   
    """
    Keep only the pickups on one day of the week inside an hour range.
    Input: df (cleaned DataFrame), day (weekday name like "Friday"),
           hour_range (tuple of start hour and end hour, 0 to 23, default (0, 23))
    Output: filtered DataFrame, which can be empty
    """
    start_hr, end_hr = hour_range               # Split the pair into two separate variables for start and end hours
    filtered = df[                              # keep only rows that pass all three conditions
        (df["Day_of_Week"] == day) &            # condition 1 - day of the week
        (df["Hour"] >= start_hr) &              # condition 2 - pickup time is after the start hour
        (df["Hour"] <= end_hr)                  # condition 3 - pickup hour is before end hour inputed by user
    ]
    return filtered


def hour_label(hour):
    """
    Turn an hour number (0 to 23) into a 12-hour label like "12 AM" or "5 PM".
    Input: hour (integer from 0 to 23)
    Output: string label such as "12 AM", "9 AM", "12 PM" or "5 PM"
    """
    if hour == 0:
        return "12 AM"
    elif hour < 12:
        return f"{hour} AM"
    elif hour == 12:
        return "12 PM"
    else:
        return f"{hour - 12} PM"

#[LOOP_VS_PANDAS] Loop + dictionary version of the hourly count 
def count_pickups_by_hour_loop(day_df):
    """
    Count pickups for each hour using a loop and a dictionary.
    Input: day_df (DataFrame of pickups for one day)
    Output: dictionary like {0: 9, 1: 6, ...} with all 24 hours
    """
    counts = {}                  #empty dictionary: the key is the hour, the value is the pickup count
    for hour in range(24):
        counts[hour] = 0         #give every hour a starting count of 0
    for hour in day_df["Hour"]:
        counts[hour] += 1        #go through each pickup's hour and add 1 to that hour's count
    return counts

# --- 3. QUESTION 1 ---
def show_question_1(df, selected_day):
    """
    Show Question 1: a bar chart and top-3 table of pickups by hour for one day.
    Input: df (cleaned DataFrame), selected_day (weekday name chosen in the sidebar)
    Output: nothing returned; draws the chart, table and driver takeaway on the page
    """
    st.header(f"1. Busiest Pickup Hours on {selected_day}s")

    q1_df = filter_pickups(df, selected_day)

    if q1_df.empty:
        st.warning(f"No pickup records found for {selected_day}. Try selecting another day.")
        return

    hourly_counts = (
        q1_df.groupby("Hour")
        .size()
        .reindex(range(24), fill_value=0)
        .reset_index(name="Total Pickups")
    )

    #[LOOP_VS_PANDAS] Same question answered twice: loop + dictionary vs Pandas groupby
    loop_counts = count_pickups_by_hour_loop(q1_df)                                              #loop answer
    pandas_counts = q1_df.groupby("Hour").size().reindex(range(24), fill_value=0).to_dict()      #pandas answer: group by hour, count rows, fill missing hours with 0, make it a dictionary
    with st.expander("Verify Loop + Dictionary vs. Pandas (hourly counts)"):                     #show the check in a collapsed box
        if loop_counts == pandas_counts:
            st.success("Both methods give the same pickups for all 24 hours.")                   #if the dictionaries are identical, both methods agree on all 24 hours
        else:
            st.error("The two methods give different results. Check the code.")
        
        compare_df = pd.DataFrame({                                                              #build a table with one row per hour
            "Hour": [hour_label(h) for h in range(24)],                                          #readable hour labels, like 12 AM and 5 PM
            "Loop + Dictionary": [loop_counts[h] for h in range(24)],                            #the loop's count for each hour
            "Pandas": [pandas_counts[h] for h in range(24)]                                      #Pandas's count for each hour
        })
        st.dataframe(compare_df, hide_index=True)  


    # ORIGINAL AI VERSION (nested lambda):
    # hourly_counts["Hour Label"] = hourly_counts["Hour"].apply(
    #     lambda h: "12 AM" if h == 0 else ("12 PM" if h == 12 else (f"{h} AM" if h < 12 else f"{h - 12} PM"))
    # )
    # REWRITTEN VERSION: same result, simple if/elif/else in its own function

    #[AI_REWRITE] replaced the nested lambda with the hour_label function
    hourly_counts["Hour Label"] = hourly_counts["Hour"].apply(hour_label)

    top_3_hours = set(hourly_counts.nlargest(3, "Total Pickups")["Hour"])
    hourly_counts["Demand Tier"] = hourly_counts["Hour"].apply(
        lambda h: "Top 3 Peak Hours" if h in top_3_hours else "Standard Hours"
    )

    hour_label_order = hourly_counts.sort_values("Hour")["Hour Label"].tolist()

    bars = (
            alt.Chart(hourly_counts)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X(
                    "Hour Label:N",
                    sort=hour_label_order,
                    title="Time of Day",
                    axis=alt.Axis(labelAngle=-45)
                ),
                y=alt.Y("Total Pickups:Q", title="Number of Pickups"),
                color=alt.Color(
                    "Demand Tier:N",
                    title="Demand Level",
                    scale=alt.Scale(
                        domain=["Top 3 Peak Hours", "Standard Hours"],
                        range=[CHART_COLORS["peak_bar"], CHART_COLORS["standard_bar"]]
                    ),
                    legend=alt.Legend(orient="top")
                ),
                tooltip=[
                    alt.Tooltip("Hour Label:N", title="Hour"),
                    alt.Tooltip("Total Pickups:Q", title="Total Pickups", format=","),
                    alt.Tooltip("Demand Tier:N", title="Category")
                ]
            )
        )

    text_labels = bars.mark_text(
        align="center",
        baseline="bottom",
        dy=-4,
        fontSize=11
    ).encode(
        text=alt.Text("Total Pickups:Q", format=",")
    )

    chart_q1 = (
        (bars + text_labels)
        .properties(
            title=alt.TitleParams(
                text=f"Total Uber Pickups by Hour on {selected_day}s",
                subtitle="Highlighted bars show the top 3 highest-demand hours",
                subtitleColor=CHART_COLORS["subtitle"], #augment - make subtitle more visible
                fontSize=16
            ),
            height=420
        )
        .configure_axis(grid=True, gridOpacity=0.3)
        .configure_view(strokeWidth=0)
    )

    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        st.altair_chart(chart_q1, use_container_width=True)

    with col_table:
        st.subheader(f"Top 3 Hours ({selected_day})")
        top_3_table = (
            hourly_counts.nlargest(3, "Total Pickups")[["Hour Label", "Total Pickups"]]
            .rename(columns={"Hour Label": "Hour", "Total Pickups": "Pickups"})
            .reset_index(drop=True)
        )
        top_3_table.insert(0, "Rank", [1, 2, 3])
        st.dataframe(top_3_table, hide_index=True, use_container_width=True)

        best_hour = top_3_table.loc[0, "Hour"]
        best_count = top_3_table.loc[0, "Pickups"]
        total_day_rides = hourly_counts["Total Pickups"].sum()
        top_3_share = (top_3_table["Pickups"].sum() / total_day_rides) * 100

        st.info(
            f"**Driver Takeaway:** On {selected_day}s, demand peaks at **{best_hour}** "
            f"({best_count:,} pickups). The top 3 hours account for **{top_3_share:.1f}%** "
            f"of all {selected_day} pickups."
        )


# --- 4. QUESTION 2 --- #[MANUAL_CALC]  loop for average, min and max --> checked against Pandas in the Q2 table
def calculate_shift_stats_manual(counts_list):
    """
    Calculate average, minimum, and maximum pickups per hour using a loop.
    Input: counts_list (list of numeric pickup counts for each hour in the shift)
    Output: tuple (avg_val, min_val, max_val)
    """
    total = 0                   #running total used for avrg
    min_val = counts_list[0]    #start min and max w first hour's count
    max_val = counts_list[0]

    for count in counts_list:   #look at one hour at a time
        total += count          #add this hour to running total
        if count < min_val:     #lower than lowest --> new min
            min_val = count
        if count > max_val:     #higher than highest --> new max
            max_val = count

    avg_val = total / len(counts_list)      #average = total pickups divided by the number of hours
    return avg_val, min_val, max_val


def show_question_2(df, selected_day, hour_range):      #pandas built in version to compare w loop 
    """
    Show Question 2: expected pickups per hour across a chosen shift window.
    Input: df (cleaned DataFrame), selected_day (weekday name),
           hour_range (tuple of (start_hour, end_hour))
    Output: nothing returned; draws metrics, manual vs Pandas check, and line chart
    """
    start_hr, end_hr = hour_range
    st.divider()
    st.header(
        f"2. Expected Shift Volume: {selected_day}s ({hour_label(start_hr)} – {hour_label(end_hr)})"
    )

    # Validate hour range and check for empty data
    try:
        if start_hr > end_hr:
            st.error("Invalid shift range: Start hour must be less than or equal to end hour.")
            return
        q2_df = filter_pickups(df, selected_day, hour_range)
    except Exception as err:
        st.error(f"Unexpected error filtering shift data: {err}")
        return

    if q2_df.empty:
        st.warning(
            f"No pickups recorded on {selected_day}s between {hour_label(start_hr)} and {hour_label(end_hr)}. "
            "Try widening your shift window."
        )
        return

    # Count pickups for every hour in the selected shift (including 0-pickup hours)
    shift_hours = list(range(start_hr, end_hr + 1))
    shift_hourly = (
        q2_df.groupby("Hour")
        .size()
        .reindex(shift_hours, fill_value=0)
        .reset_index(name="Pickups")
    )
    shift_hourly["Hour Label"] = shift_hourly["Hour"].apply(hour_label)

    # Manual loop calculation vs. Pandas built-in methods
    pickup_values = shift_hourly["Pickups"].tolist()
    loop_avg, loop_min, loop_max = calculate_shift_stats_manual(pickup_values)

    pd_avg = shift_hourly["Pickups"].mean()
    pd_min = shift_hourly["Pickups"].min()
    pd_max = shift_hourly["Pickups"].max()

    # Display key shift metrics at the top
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Shift Length", f"{len(shift_hours)} hrs")
    m2.metric("Avg Pickups / Hr", f"{loop_avg:.1f}")
    m3.metric("Min Pickups / Hr", f"{loop_min:,}")
    m4.metric("Max Pickups / Hr", f"{loop_max:,}")

    # Verification expander comparing manual loop vs Pandas
    with st.expander("Verify Manual Loop vs. Pandas Calculations"):
        comparison_df = pd.DataFrame({
            "Statistic": ["Average Pickups / Hour", "Minimum Pickups / Hour", "Maximum Pickups / Hour"],
            "Manual Loop Result": [round(loop_avg, 2), loop_min, loop_max],
            "Pandas Result": [round(pd_avg, 2), pd_min, pd_max]
        })
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    # --- CUSTOMIZED ALTAIR LINE CHART ---
    shift_order = shift_hourly.sort_values("Hour")["Hour Label"].tolist()

    line = (
        alt.Chart(shift_hourly)
        .mark_line(color=CHART_COLORS["line"], strokeWidth=3)
        .encode(
            x=alt.X(
                "Hour Label:N",
                sort=shift_order,
                title="Shift Hour",
                axis=alt.Axis(labelAngle=-30)
            ),
            y=alt.Y(
                "Pickups:Q",
                title="Pickups per Hour",
                scale=alt.Scale(zero=True)
            )
        )
    )

    points = (
        alt.Chart(shift_hourly)
        .mark_circle(size=90, color=CHART_COLORS["points"])
        .encode(
            x=alt.X("Hour Label:N", sort=shift_order),
            y=alt.Y("Pickups:Q"),
            tooltip=[
                alt.Tooltip("Hour Label:N", title="Hour"),
                alt.Tooltip("Pickups:Q", title="Pickups", format=",")
            ]
        )
    )

    # Horizontal dashed reference line showing the shift average
    avg_rule = (
        alt.Chart(pd.DataFrame({"Avg": [loop_avg]}))
        .mark_rule(color=CHART_COLORS["average_line"], strokeDash=[6, 4], strokeWidth=2)
        .encode(y="Avg:Q")
    )

    chart_q2 = (
        (line + points + avg_rule)
        .properties(
            title=alt.TitleParams(
                text=f"Hourly Pickup Trend During {selected_day} Shift ({hour_label(start_hr)} – {hour_label(end_hr)})",
                subtitle=f"Dashed red line marks shift average ({loop_avg:.1f} pickups/hr)",
                subtitleColor=CHART_COLORS["subtitle"],
                fontSize=16
            ),
            height=380
        )
        .configure_axis(grid=True, gridOpacity=0.3)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart_q2, use_container_width=True)
    st.caption(
        "Note: Counts represent total pickups recorded in the April 2014 sample dataset "
        "for this weekday and shift window."
    )


# --- 5. QUESTION 3 ---
def show_question_3(df, selected_day, hour_range, show_base_colors):
    """
    Show Question 3: spatial pickup clusters on a customized PyDeck map.
    Input: df (cleaned DataFrame), selected_day (weekday name),
           hour_range (tuple of (start_hour, end_hour)), show_base_colors (bool)
    Output: nothing returned; renders PyDeck heatmap or base-colored scatterplot
    """
    start_hr, end_hr = hour_range
    st.divider()
    st.header(
        f"3. Pickup Hotspots on {selected_day}s ({hour_label(start_hr)} – {hour_label(end_hr)})"
    )

    q3_df = filter_pickups(df, selected_day, hour_range).copy()

    if q3_df.empty:
        st.warning(
            f"No locations to map on {selected_day}s between {hour_label(start_hr)} and {hour_label(end_hr)}."
        )
        return

    # Add readable hour label for map tooltips
    q3_df["Hour Label"] = q3_df["Hour"].apply(hour_label)

    # Center camera on Midtown Manhattan
    view_state = pdk.ViewState(
        latitude=40.7549,
        longitude=-73.9840,
        zoom=11,
        pitch=35
    )

    if show_base_colors and "Base" in q3_df.columns:
        # Map each Uber base code to its RGBA color
        q3_df["color"] = q3_df["Base"].map(BASE_COLORS).fillna(
            pd.Series([DEFAULT_COLOR] * len(q3_df), index=q3_df.index)
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=q3_df,
            get_position="[Lon, Lat]",
            get_fill_color="color",
            get_radius=90,
            pickable=True,
            auto_highlight=True
        )
        tooltip = {
            "html": "<b>Dispatch Base:</b> {Base}<br/><b>Time:</b> {Hour Label}<br/><b>Coords:</b> {Lat}, {Lon}",
            "style": {"backgroundColor": "#1E293B", "color": "white"}
        }
        st.markdown(
            "**Base Color Legend:** "
            "🔴 `B02512` (Unter) · 🔵 `B02598` (Hinter) · 🟢 `B02617` (Weiter) · "
            "🟣 `B02682` (Schmecken) · 🟡 `B02764` (Danach-NY)"
        )
    else:
        # Default PyDeck HeatmapLayer showing pickup density
        layer = pdk.Layer(
            "HeatmapLayer",
            data=q3_df,
            get_position="[Lon, Lat]",
            aggregation=pdk.types.String("SUM"),
            radiusPixels=45,
            intensity=1.2,
            threshold=0.05
        )
        tooltip = None

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip
    )

    st.pydeck_chart(deck, use_container_width=True)
    st.info(
        f"**Map Summary:** Showing **{len(q3_df):,}** pickups on {selected_day}s between "
        f"**{hour_label(start_hr)}** and **{hour_label(end_hr)}**. "
        "Toggle **Show base colors** in the sidebar to inspect individual dispatching bases and hover tooltips."
    )


# --- MAIN APP ---
def main():
    """
    Build the page: title, load the data, sidebar controls, and the question sections.
    Input: none
    Output: nothing returned; draws the whole app
    """
    st.title("NYC Uber Driver Demand & Shift Explorer")
    st.markdown(
        "Explore historical NYC Uber pickup patterns to decide **when** to drive, "
        "**what volume** to expect during a shift, and **where** demand clusters."
    )

    # [INPUT_CHECK] - Load data safely from file 
    try:
        df = load_and_clean_data(DATA_FILE)                          #load the csv. raises FileNotFoundError if the file is missing
    except FileNotFoundError:
        st.error(f"Could not find `{DATA_FILE}`. Make sure the CSV is in the same folder as `app.py`.")   #show a friendly message with the missing file name
        st.stop()                                                    #stop the app as df was never created

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Driver Shift Filters")
    selected_day = st.sidebar.selectbox("Select Day of the Week", DAYS_ORDER, index=4)  # Default: Friday

    shift_range = st.sidebar.slider(
        "Select Shift Hours (for Q2 & Q3)",
        min_value=0,
        max_value=23,
        value=(16, 22),  # Default: 4 PM to 10 PM evening rush
        format="%d:00"
    )
    st.sidebar.caption(
        f"Selected Shift: **{hour_label(shift_range[0])}** to **{hour_label(shift_range[1])}**"
    )

    show_base_colors = st.sidebar.checkbox(
        "Show base colors (for Q3 Map)",
        value=False
    )

    # Render all three question sections
    show_question_1(df, selected_day)
    show_question_2(df, selected_day, shift_range)
    show_question_3(df, selected_day, shift_range, show_base_colors)


if __name__ == "__main__":
    main()
