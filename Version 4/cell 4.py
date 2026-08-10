def train_model(training_rows, feature_columns):

    model = LGBMRegressor(
        objective="poisson",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=5,
        min_child_samples=10,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        training_rows[feature_columns],
        training_rows[TARGET_COLUMN],
    )

    return model


print("Training function created.")
