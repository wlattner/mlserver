mlserver
========

This is a simple application that provides an HTTP/JSON api for machine learning. Currently, only classification is implemented. The server is written in Go, the machine learning algorithms are Python, from the [Scikit-Learn](http://scikit-learn.org/stable/) library. Each model is run inside a separate child process. The models fitted in fit.py are pickled using [joblib](http://scikit-learn.org/stable/modules/model_persistence.html) and saved to a folder named models in the working directory of mlserver.

### Building/Installing

Building the app is fairly simple (assuming Go is installed and $GOPAH is set):

```bash
go get github.com/wlattner/mlserver
```
This will clone the repo to `$GOPATH/src/github.com/wlattner/mlserver` and copy the `mlserver` binary to `$GOPATH/bin`.

The code in `fit.py` and `train.py` require Python 3, NumPy, SciPy and Scikit-Learn; these are sometimes tricky to install, look elsewhere for instructions.

### Running

Start the server:
```bash
mlserver
```

By default, the server will listen on port 5000.

TODO
====
- [ ] error handling, especially with fit/predict input
- [ ] automatically stop unused models
- [ ] store models in S3
- [ ] add regression, detect which based on input data
- [ ] better model selection in fit.py
- [ ] better project name
- [ ] config options
- [X] csv file upload for fit/predict input
- [ ] docker container for fit.py and predict.py
- [ ] tests

API
===

Get Models
----------

* `GET /models` will return all models on the server

```json
[
  {
    "model_id": "0e12bb73-e49a-4dcd-87aa-cb0338b1c758",
    "metadata": {
      "name": "iris model 1",
      "created_at": "2014-11-06T21:52:16.143688Z"
    },
    "performance": {
      "algorithm": "GradientBoostingClassifier",
      "confusion_matrix": {
        "setosa": {
          "setosa": 50,
          "versicolor": 0,
          "virginica": 0
        },
        "versicolor": {
          "setosa": 0,
          "versicolor": 50,
          "virginica": 0
        },
        "virginica": {
          "setosa": 0,
          "versicolor": 0,
          "virginica": 50
        }
      },
      "score": 0.9673202614379085
    },
    "running": false,
    "trained": true
  },
  {
    "model_id": "26f786c1-5e59-432f-a3b0-8b87025043f8",
    "metadata": {
      "name": "ESL 10.2 Generated Data",
      "created_at": "2014-11-07T00:47:14.602932Z"
    },
    "performance": {
      "algorithm": "GradientBoostingClassifier",
      "confusion_matrix": {
        "-1.0": {
          "-1.0": 5931,
          "1.0": 111
        },
        "1.0": {
          "-1.0": 307,
          "1.0": 5651
        }
      },
      "score": 0.9285000000000001
    },
    "running": false,
    "trained": true
]
```

Get Model
---------
* `GET /models/:model_id` will return the specified model.

```json
{
  "model_id": "0e12bb73-e49a-4dcd-87aa-cb0338b1c758",
  "metadata": {
    "name": "iris model 1",
    "created_at": "2014-11-06T21:52:16.143688Z"
  },
  "performance": {
    "algorithm": "GradientBoostingClassifier",
    "confusion_matrix": {
      "setosa": {
        "setosa": 50,
        "versicolor": 0,
        "virginica": 0
      },
      "versicolor": {
        "setosa": 0,
        "versicolor": 50,
        "virginica": 0
      },
      "virginica": {
        "setosa": 0,
        "versicolor": 0,
        "virginica": 50
      }
    },
    "score": 0.9673202614379085
  },
  "running": false,
  "trained": true
}
```

Fit
---

* `POST /models` will create and fit a new model with the supplied training data.

The request body should be JSON with the following fields:

* `name` the name of the model
* `data` an array of objects, each element represents a single row/observation
* `labels` an array of strings representing the target value/label of each training example

To fit a model for predicting the species variable from the [Iris data](http://en.wikipedia.org/wiki/Iris_flower_data_set):

sepal_length | sepal_width | petal_length | petal_width | species
------------ | ----------- | ------------ | ----------- | -------
5.1 | 3.5 | 1.4 | 0.2 | setosa
4.9 | 3.0 | 1.4 | 0.2 | setosa
4.7 | 3.2 | 1.3 | 0.2 | setosa
4.6 | 3.1 | 1.5 | 0.2 | setosa
5.0 | 3.6 | 1.4 | 0.2 | setosa
... | ... | ... | ... | ...

```json
{
  "name": "iris model",
  "data": [
    {
      "sepal_length": 5.1,
      "petal_length": 1.4,
      "sepal_width": 3.5,
      "petal_width": 0.2
    },
    {
      "sepal_length": 4.9,
      "petal_length": 1.4,
      "sepal_width": 3.0,
      "petal_width": 0.2
    },
    {
      "sepal_length": 4.7,
      "petal_length": 1.3,
      "sepal_width": 3.2,
      "petal_width": 0.2
    },
    {
      "sepal_length": 4.6,
      "petal_length": 1.5,
      "sepal_width": 3.1,
      "petal_width": 0.2
    },
    {
      "sepal_length": 5.0,
      "petal_length": 1.4,
      "sepal_width": 3.6,
      "petal_width": 0.2
    }
  ],
  "labels": [
    "setosa",
    "setosa",
    "setosa",
    "setosa",
    "setosa"
  ]
}
```

This will return `202 Accepted` along with the id of the newly created model. The model will be fitted in the background.

```json
{
  "model_id": "07421303-62f9-40f3-bf14-23cf44af05e2"
}
```

Alternatively, the data for fitting a model can be uploaded as a csv file. The file must have a header row and the target variable must be the first column. The table above would be encoded as:
	
	"species","sepal_length","sepal_width","petal_length","petal_width"
	"setosa",5.1,3.5,1.4,0.2
	"setosa",4.9,3,1.4,0.2
	"setosa",4.7,3.2,1.3,0.2
	"setosa",4.6,3.1,1.5,0.2
	"setosa",5,3.6,1.4,0.2
	"setosa",5.4,3.9,1.7,0.4
	"setosa",4.6,3.4,1.4,0.3
	"setosa",5,3.4,1.5,0.2
	"setosa",4.4,2.9,1.4,0.2

The request should be encoded as multipart/form with the following fields:

* `name` the name to use for the model
* `file` the csv file

```bash
curl --form name="iris model csv" --form file=@iris.csv http://localhost:5000/models
```

Predict
-------

* `POST /models/:model_id` will return predictions using the model for the supplied data

The request body should have the following fields:

* `data` an array of objects, each element represents a single row/observation

To make predict labels (species) for the following data:

sepal_length | sepal_width | petal_length | petal_width | species
------------ | ----------- | ------------ | ----------- | -------
6.7 | 3.0 | 5.2 | 2.3 | ?
6.3 | 2.5 | 5.0 | 1.9 | ?
6.5 | 3.0 | 5.2 | 2.0 | ?
6.2 | 3.4 | 5.4 | 2.3 | ?
5.9 | 3.0 | 5.1 | 1.8 | ? 

```json
{
  "data": [
    {
      "sepal_length": 6.7,
      "petal_length": 5.2,
      "sepal_width": 3.0,
      "petal_width": 2.3
    },
    {
      "sepal_length": 6.3,
      "petal_length": 5.0,
      "sepal_width": 2.5,
      "petal_width": 1.9
    },
    {
      "sepal_length": 6.5,
      "petal_length": 5.2,
      "sepal_width": 3.0,
      "petal_width": 2.0
    },
    {
      "sepal_length": 6.2,
      "petal_length": 5.4,
      "sepal_width": 3.4,
      "petal_width": 2.3
    },
    {
      "sepal_length": 5.9,
      "petal_length": 5.1,
      "sepal_width": 3.0,
      "petal_width": 1.8
    }
  ]
}
```

The response will contain class probabilities for each example submitted:

```json
{
  "labels": [
    {
      "versicolor": 0.000005590474449815602,
      "virginica": 0.9999925658927716,
      "setosa": 0.000001843632778976535
    },
    {
      "versicolor": 0.00003448150394080962,
      "virginica": 0.9999626991986605,
      "setosa": 0.0000028192973987744193
    },
    {
      "versicolor": 0.00000583767259563357,
      "virginica": 0.9999923186950813,
      "setosa": 0.0000018436323232313824
    },
    {
      "versicolor": 0.000025292685027774954,
      "virginica": 0.9999702563844668,
      "setosa": 0.000004450930505165
    },
    {
      "versicolor": 0.00006891207512697766,
      "virginica": 0.9999281614880159,
      "setosa": 0.000002926436856866432
    }
  ],
  "model_id": "0e12bb73-e49a-4dcd-87aa-cb0338b1c758"
}
```

Alternatively, the data could be uploaded as a csv file, see above description for fitting a model using a csv file. In the case of making predictions, the csv file should not have the label/target data in the first column.

Start Model
----------
The prediction woker is started with the first prediction request for a model. A model can be started manually however.

* `POST /models/running` will start a model

```json
{
  "model_id": "0e12bb73-e49a-4dcd-87aa-cb0338b1c758"
}
```

This will return `201 Created` with an empty body. The model will be started in the background.

Stop Model
----------
Once started, models will run until the server process exits. Models can be stopped manually.

* `DELETE /models/running/:model_id` will stop a model

This will return `202 Accepted` with an empty body. The model will be stopped in the background.
