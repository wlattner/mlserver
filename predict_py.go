package main

var predictPy = `
import zmq
import signal
import sys
from sklearn.externals import joblib

def predict(model, X):
	predictions = []
	labels = [str(label) for label in model.steps[-1][-1].classes_]
	for prediction in model.predict_proba(X):
		predictions.append({labels[lab]: prob for lab, prob in enumerate(prediction)})
	return predictions

def load(path):
	return joblib.load(path)

def run(model_path, socket_path):
	model = load(model_path)

	context = zmq.Context()
	socket = context.socket(zmq.REP)
	socket.bind(socket_path)

	try:
		while True:
			message = socket.recv_json()
			predictions = predict(model, message['data'])
			socket.send_json(predictions)
	finally:
		context.destroy()

def exit_on_sigint(_sig, _stack_frame):
	sys.exit(0)

if __name__ == "__main__":
	signal.signal(signal.SIGINT, exit_on_sigint)

	socket_path = sys.argv[1]
	model_path = sys.argv[2]

	run(model_path, socket_path)

`
