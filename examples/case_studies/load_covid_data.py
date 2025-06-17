import polars as pl


def load_individual_timeseries(name):
    base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series"
    url = f"{base_url}/time_series_covid19_{name}_global.csv"
    df = pl.read_csv(url)

    # Rename columns to match expected names
    df = df.rename(
        {"Country/Region": "country", "Province/State": "state", "Lat": "lat", "Long": "long"}
    )

    # Drop lat/long columns
    df = df.drop(["lat", "long"])

    # Get date columns (all columns except country and state)
    date_cols = [col for col in df.columns if col not in ["country", "state"]]

    # Unpivot to long format
    df = df.unpivot(
        index=["country", "state"], on=date_cols, variable_name="date", value_name="cases"
    )

    # Add type column and convert date
    df = df.with_columns(
        [pl.col("date").str.to_date("%m/%d/%y"), pl.lit(name.lower()).alias("type")]
    )

    # Move HK to country level
    df = df.with_columns(
        [
            pl.when(pl.col("state") == "Hong Kong")
            .then(pl.lit("Hong Kong"))
            .otherwise(pl.col("country"))
            .alias("country"),
            pl.when(pl.col("state") == "Hong Kong")
            .then(None)
            .otherwise(pl.col("state"))
            .alias("state"),
        ]
    )

    # Create aggregated data for countries with states
    df_with_states = df.filter(pl.col("state").is_not_null())
    if len(df_with_states) > 0:
        df_aggregated = (
            df_with_states.group_by(["country", "date", "type"])
            .agg(pl.col("cases").sum())
            .with_columns(
                [(pl.col("country") + " (total)").alias("country"), pl.lit(None).alias("state")]
            )
            .select(["country", "state", "date", "cases", "type"])
        )  # Ensure same column order

        # Combine original and aggregated data
        df = pl.concat([df, df_aggregated], how="vertical")

    return df


def load_data(drop_states=False, p_crit=0.05, filter_n_days_100=None):
    df = load_individual_timeseries("confirmed")
    df = df.rename({"cases": "confirmed"})

    if drop_states:
        # Drop states for simplicity
        df = df.filter(pl.col("state").is_null())

    # Ensure consistent string type for state column and replace nulls
    df = df.with_columns([pl.col("state").cast(pl.String).fill_null("NO_STATE")])

    # Estimated critical cases
    df = df.with_columns((pl.col("confirmed") * p_crit).alias("critical_estimate"))

    # Compute days relative to when 100 confirmed cases was crossed
    # Create a list to store results for each country/state combination
    days_since_100_data = []

    # Get unique combinations of country and state
    country_state_combinations = df.select(["country", "state"]).unique()

    for row in country_state_combinations.iter_rows(named=True):
        country = row["country"]
        state = row["state"]

        # Filter data for this country/state combination
        state_filter = "NO_STATE" if state is None else state
        subset = df.filter((pl.col("country") == country) & (pl.col("state") == state_filter)).sort(
            "date"
        )

        # Calculate days since 100
        confirmed_values = subset["confirmed"].to_list()
        dates = subset["date"].to_list()

        days_before_100 = sum(1 for x in confirmed_values if x < 100)
        days_after_100 = sum(1 for x in confirmed_values if x >= 100)

        days_since_100 = list(range(-days_before_100, days_after_100))

        # Create dataframe for this combination
        state_value = "NO_STATE" if state is None else state
        temp_df = pl.DataFrame(
            {
                "country": [country] * len(dates),
                "state": [state_value] * len(dates),
                "date": dates,
                "days_since_100": days_since_100,
            }
        )

        days_since_100_data.append(temp_df)

    # Combine all days_since_100 data
    if days_since_100_data:
        days_df = pl.concat(days_since_100_data, how="vertical")

        # Join with main dataframe
        df = df.join(days_df, on=["country", "state", "date"], how="left")

    # Add deaths
    df_deaths = load_individual_timeseries("deaths")
    df_deaths = df_deaths.rename({"cases": "deaths"}).drop("type")

    # Apply same state handling to deaths data
    df_deaths = df_deaths.with_columns([pl.col("state").cast(pl.String).fill_null("NO_STATE")])

    df = df.join(df_deaths, on=["country", "state", "date"], how="left")

    if filter_n_days_100 is not None:
        # Select countries for which we have at least some information
        countries = df.filter(pl.col("days_since_100") >= filter_n_days_100)["country"].unique()
        df = df.filter(pl.col("country").is_in(countries))

    # Convert NO_STATE back to null for compatibility with original pandas version
    df = df.with_columns(
        [
            pl.when(pl.col("state") == "NO_STATE")
            .then(None)
            .otherwise(pl.col("state"))
            .alias("state")
        ]
    )

    return df
