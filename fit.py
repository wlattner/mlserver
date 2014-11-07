import os
import json
import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix

def fit(X, Y):
  models = {
    'LogisticRegression': LogisticRegression(),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=150),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=150)
  }

  best_score = 0
  best_model = ''
  for model in models:
    vec = DictVectorizer(sparse=False)
    clf = models[model]
    pl = Pipeline([('vec', vec), ('clf', clf)])

    #TODO: grid search for model params
    scores = cross_val_score(pl, X, Y, n_jobs=3)
    if scores.mean() > best_score:
      best_score = scores.mean()
      best_model = model

  # retrain best model with all data
  vec = DictVectorizer(sparse=False)
  clf = models[best_model]
  pl = Pipeline([('vec', vec), ('clf', clf)])
  pl.fit(X, Y)
  pl.score_ = best_score  # report cv score
  return pl

def save(path, model_id, model):
	fname = model_id + '.pkl'
	if not os.path.exists(path):
		os.makedirs(path)
	joblib.dump(model, os.path.join(path, fname))

def save_metadata(path, model_id, model_name, model, X, Y):
	Y_hat = model.predict(X)

	labels = [l for l in model.named_steps['clf'].classes_]

	cm = confusion_matrix(Y, Y_hat, labels=labels)
	# this is an insane dict comprehension, need to encode the val as a float, json will not encode 0
	cm_dict = {str(labels[inx]): {str(labels[c]):float(val) for c, val in enumerate(row)} for inx, row in enumerate(cm)}

	model_data = {
		"model_id": model_id,
		"metadata": {
			"name": model_name,
			"created_at": datetime.datetime.utcnow().isoformat('T') + 'Z'
		},
		"performance" : {
			"algorithm": model.named_steps['clf'].__class__.__name__,
			"score": model.score_,
			"confusion_matrix": cm_dict
		}
	}

	json.dump(model_data, open(os.path.join(path, model_id + '.json'), 'w'))


if __name__ == "__main__":
	import sys

	data = json.load(open(sys.argv[2]))
	model_save_path = sys.argv[1]
	model_id = os.path.basename(model_save_path)

	model = fit(data['data'], data['labels'])
	save(model_save_path, model_id, model)
	save_metadata(model_save_path, model_id, data['name'], model, data['data'], data['labels'])
