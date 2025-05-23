# Harsh Review Detector

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Detecting guests who consistently give overly harsh reviews on the Nocarz booking platform.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         harsh_review_detector and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── harsh_review_detector   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes harsh_review_detector a Python module
    │
    ├── analyze_ab_log.py
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── service.py              <- Flask microservice
    │
    ├── service_utils.py        <- Helper functions for microservice
    │
    ├── simulate_experiment_ab.py
    │
    ├── modeling                
    │   ├── dataset_operations.py 
    │   ├── naive_bayes_train_and_evaluate.py          
    │   └── svm_train_and_ewaluate.py
    │
```

--------

