package main

/*
This app allows Scikit-Learn classifiers to fitted and used through an HTTP/JSON
api. Each models is run inside a dedicated python child process. Go communicates with
each process using zeromq, although using stdin/stdout may also work. The fitting
script does some primitive model selection. Currently using RandomForestClassifier,
LogisticRegression, and GradientBoostingClassifier. RandomForestClassifier and
GradientBoostingClassifier are each called with n_estimators=150, LogisticRegression
uses the default arguments.
*/

import (
	"flag"
	"net/http"

	"github.com/coreos/go-log/log"
)

var (
	port     = flag.String("port", "5000", "port for api server")
	modelDir = flag.String("model-path", "models", "location of model directory")
)

func main() {
	flag.Parse()

	models := NewModelRepo(*modelDir)

	log.Info("started indexing model directory")
	models.IndexModelDir()
	log.Info("finished indexing model directory")

	s := NewAPIHandler(models)

	log.Info("listening on http://localhost:" + *port)
	log.Fatalln(http.ListenAndServe(":"+*port, requestLogger(s)))
}
