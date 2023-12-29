import mlflow


def setup_mlflow_experiment():
    experiment_name = "restaurant_rating_prediction"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment


def start_mlflow_run():
    experiment_name = "restaurant_rating_prediction"
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def end_mlflow_run():
    mlflow.end_run()


def get_active_run_id():
    return mlflow.active_run().info.run_id
