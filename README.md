# Harsh Review Detector

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Detecting guests who consistently give overly harsh reviews on the Nocarz booking platform.

## Available commands

* `make create_environment` - creates Python environment
* `make requirements` - installs Python dependencies
* `make clean` - deletes all compiled Python files
* `make lint` - lints using ruff (use `make format` to do formatting)
* `make format` - formats source code with ruff
* `make mypy` - checks type hints with mypy
* `make svm` - trains and evaluates SVM models for harsh review detection
* `make nb` - trains and evaluates Naive Bayes models for harsh review detection
* `make service` - runs Flask microservice
* `make simulate_ab` - simulates A/B experiment

## Microservice endpoints

### POST /predict/base-model

* Performs prediction using only the base model
* Accepts data in JSON format
* Saves prediction logs
* Returns the classification result

#### Example of valid requests


```
curl -X POST http://localhost:8080/predict/base-model \
-H "Content-Type: application/json" \
-d '{"review": "This apartment was dirty and noisy. Not recommended!"}'
```



```
curl -X POST http://localhost:8080/predict/base-model \
-H "Content-Type: application/json" \
-d '{"review": "This apartment was dirty and noisy. Not recommended!", "true_label": 1}'
```

Returned result for the above requests:
```
{"prediction":1}
```

Label 1 indicates that the provided review is harsh.

### POST /predict/advanced-model

* Performs prediction using only the advanced model
* Accepts data in JSON format
* Saves prediction logs
* Returns the classification result

#### Example of valid requests


```
curl -X POST http://localhost:8080/predict/advanced-model \
-H "Content-Type: application/json" \
-d '{"review": "This apartment was dirty and noisy. Not recommended!"}'
```



```
curl -X POST http://localhost:8080/predict/advanced-model \
-H "Content-Type: application/json" \
-d '{"review": "This apartment was dirty and noisy. Not recommended!", "true_label": 1}'
```

Returned result for the above requests:
```
{"prediction":1}
```

Label 1 indicates that the provided review is harsh.

### POST /experiment_ab

* Accepts data in JSON format
* Automatically assigns the user to one of the experimental groups, using a balanced user distribution between models
* Saves prediction logs
* Returns the classification result

#### Example of valid requests

```
curl -X POST http://localhost:8080/experiment_ab \
-H "Content-Type: application/json" \
-d '{"review": "This apartment was dirty and noisy. Not recommended!", "user_id": 2}'
```



```
curl -X POST http://localhost:8080/experiment_ab \
-H "Content-Type: application/json" \
-d '{"review": "This apartment was dirty and noisy. Not recommended!", "user_id": 2,
"true_label": 1}'
```

Returned result for the above requests:
```
{"prediction":1}
```

Label 1 indicates that the provided review is harsh.

## Project Organization

```
├── LICENSE            <- The MIT License
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

