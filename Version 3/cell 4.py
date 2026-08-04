# TRAIN THE LIGHTGBM MODEL
def train_model(
    training_rows,
    feature_columns,
):

    model = LGBMRegressor(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        training_rows[feature_columns],
        training_rows[TARGET_COLUMN],
    )

    return model


print("Training function created.")
