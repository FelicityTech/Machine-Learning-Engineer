# Import MLflow Module
import mlflow

# Run local Project
mlflow.projects.run(
    uri='./', entry_point='main',
    experiment_name='Salary Model',
    Parameters={
        'n_jobs_param': 2,
        'fit_intercept_param': False
    }
)