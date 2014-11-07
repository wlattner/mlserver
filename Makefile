all: mlserver

install: mlserver
	go install

mlserver: fit_py.go predict_py.go server.go
	go build

fit_py.go: fit.py
	./py_to_go.py fitPy < fit.py > fit_py.go

predict_py.go: predict.py
	./py_to_go.py predictPy < predict.py > predict_py.go 