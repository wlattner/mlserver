package main

/*
This app allows Scikit-Learn classifiers to fitted and used through an HTTP/JSON
api. Each models is run inside a dedicated python child process. Go communicates with
each process using zeromq, although using stdin/stdout may also work. The fitting
script does some primitive model selection. Currently using RandomForestClassifier,
LogisticRegression, and GradientBoostingClassifier. RandomForestClassifier and
GradientBoostingClassifier are each called with n_estimators=150, LogisticRegression
uses the default arguments.

TODO:
	1) more error handling, especially user-supplied input
	4) LRU cache or some method to shutdown unused models
	5) option for writing models to S3, Go could copy to temp location on demand
	6) feels dirty passing the python scripts to python via stdin, other methods?
	7) add regression
	8) grid search for param values of each model
   10) allow instances multiple copies of model
   11) use git pre-commit hook + makefile to copy fit.py and predict.py
   12) find a super sweet name
   13) command line args/config options
   14) accept csv upload for fit/predict
   15) docker containers for fit/predict
*/

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"code.google.com/p/go-uuid/uuid"
	"github.com/coreos/go-log/log"
	zmq "github.com/pebbe/zmq4"
)

// Prediction is the parsed result from the Python worker
type Prediction struct {
	ModelID string               `json:"model_id"`
	Labels  []map[string]float64 `json:"labels"`
}

// ModelReq represents an incoming request for fit or predict
type ModelReq struct {
	ModelID string                   `json:"model_id"`
	Name    string                   `json:"name"`
	Date    time.Time                `json:"created_at"`
	Data    []map[string]interface{} `json:"data"`
	Labels  []interface{}            `json:"labels"`
}

// Model represents a previously fitted model
type Model struct {
	ID       string `json:"model_id"`
	Metadata struct {
		Name string    `json:"name"`
		Date time.Time `json:"created_at"`
	} `json:"metadata"`
	Performance struct {
		Algorithm       string                        `json:"algorithm"`
		ConfusionMatrix map[string]map[string]float64 `json:"confusion_matrix,omitempty"`
		Score           float64                       `json:"score"`
	} `json:"performance"`
	runLock sync.RWMutex // protect running attribute
	Running bool         `json:"running"`
	Trained bool         `json:"trained"`
	// req and rep follow the zmq semantics for REQ/REP socket pairs,
	// data sent to the req channel is piped to the REQ socket connected
	// to the running Python process, replies from Python are piped to the
	// rep channel
	req, rep chan []byte
	dir      string    // path to the directory containing <model_id>.pkl and <model_id>.json
	cmd      *exec.Cmd // the running process
}

// Predict encodes the client supplied data, passes it to the Python process for
// the model via zmq, parses and returns the response.
func (m *Model) Predict(r ModelReq) Prediction {
	// should find a way to do this w/o re-encoding
	var buf bytes.Buffer
	err := json.NewEncoder(&buf).Encode(r)
	if err != nil {
		log.Error("error encoding prediction ", err)
		return Prediction{}
	}

	if m.req == nil {
		log.Errorf("request chan for model %v is nil", m.ID)
		return Prediction{}
	}
	m.req <- buf.Bytes()
	resp := <-m.rep

	var pred []map[string]float64
	err = json.NewDecoder(bytes.NewReader(resp)).Decode(&pred)
	if err != nil {
		log.Error("error decoding prediction ", err)
	}

	prediction := Prediction{
		ModelID: r.ModelID,
		Labels:  pred,
	}

	return prediction
}

// Stop sends SIGINT to the underlying process running the model
func (m *Model) Stop() error {
	if m.cmd != nil {
		return m.cmd.Process.Signal(os.Interrupt)
	}
	return nil
}

// Models is the global index of all models previously fitted or indexed from
// the model directory
var Models *ModelRepo

// ModelRepo represents a collection of models
type ModelRepo struct {
	sync.RWMutex
	collection map[string]*Model
	path       string
}

// NewModelRepo initializes and returns a pointer to a ModelRepo, the supplied
// path argument refers to the directory where pickled models will be saved.
func NewModelRepo(path string) *ModelRepo {
	return &ModelRepo{
		collection: make(map[string]*Model),
		path:       path,
	}
}

// Add inserts a model into the model collection
func (r *ModelRepo) Add(m *Model) {
	r.Lock()
	defer r.Unlock()
	r.collection[m.ID] = m
}

// Remove deletes a model from the model collection
func (r *ModelRepo) Remove(id string) {
	r.Lock()
	defer r.Unlock()
	// TODO: make sure the python process has exited or kill
	// prior to delete
	delete(r.collection, id)
}

// NewModel initializes a model with a generated ID and dir
func (r *ModelRepo) NewModel() *Model {
	id := uuid.New()
	m := Model{ID: id, dir: filepath.Join(r.path, id)}
	return &m
}

// All returns a slice of all models currently in the collection
func (r *ModelRepo) All() []*Model {
	var models []*Model

	r.RLock()
	for _, model := range r.collection {
		models = append(models, model)
	}
	r.RUnlock()

	return models
}

// ErrModelNotFound is returned when the model can't be found in the model dir
var ErrModelNotFound = errors.New("model not found")

// Get fetches a model by id, if the model is not present in the collection, it
// will attempt to load from disk adding it to the collection. If the model is
// not in the model directory, Get will return ErrModelNotFound.
func (r *ModelRepo) Get(id string) (*Model, error) {
	r.RLock()
	m, ok := r.collection[id]
	r.RUnlock()

	var err error
	if !ok {
		m, err = r.LoadModelData(id)
		if err != nil {
			return nil, err
		}
	}

	r.Add(m) // add to cache

	// start/restart if not running
	m.runLock.Lock() // make sure we don't start twice
	defer m.runLock.Unlock()
	if !m.Running {
		err = startModel(m)
		if err != nil {
			return nil, err
		}
	}

	return m, nil
}

// LoadModelData loads the model metadata from the file
// <path>/<model_id>/<model_id>.json, if the file does not exist, ErrModelNotFound
// is returned. The json file is expected to contain the model score, confusion matrix,
// and algorithm used, see Model.Metadata. The loaded model is added to the collection.
func (r *ModelRepo) LoadModelData(id string) (*Model, error) {
	// check the collection first
	r.RLock()
	m, ok := r.collection[id]
	r.RUnlock()
	if !ok { // not currently loaded
		modelDir := filepath.Join(r.path, id)
		f, err := os.Open(filepath.Join(modelDir, id+".json"))
		if os.IsNotExist(err) {
			return nil, ErrModelNotFound
		}

		if err != nil {
			return nil, err
		}

		var m Model
		err = json.NewDecoder(f).Decode(&m)
		if err != nil {
			return nil, err
		}
		m.dir = modelDir
		m.Trained = true

		r.Add(&m) // add to cache
	}

	return m, nil
}

func (r *ModelRepo) IndexModelDir() error {
	models, err := filepath.Glob(filepath.Join(r.path, "/*"))
	if err != nil {
		return err
	}

	for _, model := range models {
		modelID := strings.TrimPrefix(model, r.path+"/")
		r.LoadModelData(modelID)
	}
	return nil
}

//-----------------------------------------------------------------------------
// Python Fit/Predict Child Process
//-----------------------------------------------------------------------------

// fitModel writes the training data in json format to a temporary file. Next
// it launches the fit.py in a child process, passing the filename of the trainig
// data and the location where the model should be saved as arguments. Since we
// do not know the path the app will be run, we instruct python to read the fit.py
// source from stdin instead of executing a file. This would be equivalent to:
//
// 	$ python3 - < fit.py tmp.json models/model-id
//
// The source for fit.py as encoded as a raw/formatted string in the file
// fit_py.go
//
// When the command completes, go checks the exit status, anything other than exit(0)
// will result in a non-nil value for the error returned by cmd.Run().

func fitModel(m *Model, d ModelReq) {
	log.Infof("started fitting model %v", m.ID)
	// write data to temp file
	f, err := ioutil.TempFile("", m.ID)
	if err != nil {
		log.Error("unable to open temp file for fitting model ", err)
		return
	}
	defer os.Remove(f.Name())

	err = json.NewEncoder(f).Encode(d)
	if err != nil {
		log.Error("error encoding training data ", err)
		return
	}
	f.Close()

	cmd := exec.Command("python3", "-", m.dir, f.Name())
	cmd.Stdin = strings.NewReader(fitPy)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	err = cmd.Run()
	if err != nil {
		log.Errorf("error fitting model %v: %v %v", m.ID, err.Error(), stderr.String())
	}

	// load the model into the index after fitted
	_, err = Models.LoadModelData(m.ID)
	if err != nil {
		log.Errorf("error loading model %v: %v", m.ID, err.Error())
	}
	log.Infof("finished fitting model %v", m.ID)
}

// startModel launches the prediction script for a model in a child process.
//
// Requests and responses between Go and the prediction process occur via a zmq
// REQ/REP socket pair. The ipc socket path and model file name are passed to the
// python script as command line args. On startup, predicy.py loads the model and
// binds a REP socket to the provided ipc path. The script than starts a loop,
// reading data from the the socket, returning predicitons back over the socket.
// On the Go side, one goroutine manages the running python process, (it doesn't
// really do much, just sets the Running attribute to false on exit), another
// goroutine accepts requests via the model's req chan, forwards these to the REQ
// socket, reads the python response, and forwards these to the model's rep chan.
func startModel(m *Model) error {

	// create channels and set running flag
	m.req = make(chan []byte)
	m.rep = make(chan []byte)
	m.Running = true

	socketPath := fmt.Sprint("ipc:///tmp/", m.ID)

	socket, err := zmq.NewSocket(zmq.REQ)
	if err != nil {
		return err
	}

	fileName := fmt.Sprintf("%s.pkl", m.ID)

	cmd := exec.Command("python3", "-", socketPath, filepath.Join(m.dir, fileName))
	cmd.Stdin = strings.NewReader(predictPy)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	m.cmd = cmd // attach cmd to the model object

	// run the predict.py in a dedicated goroutine, this function will return
	// when predict.py exits
	go func() {
		defer func() {
			m.runLock.Lock()
			m.Running = false
			m.runLock.Unlock()
			close(m.req) // no more requests after process exits
		}()

		log.Infof("starting model %v", m.ID)
		err := cmd.Run()
		if err != nil {
			log.Errorf("model %v exited: %v %v", m.ID, err.Error(), stderr.String())
			return
		}
		// fit.py exited normally
		log.Infof("model %v exited", m.ID)
	}()

	err = socket.Connect(socketPath)
	if err != nil {
		return err
	}

	// forward requests sent to model.req channel to the zeromq REQ socket,
	// read the response from zeromq and push to model.rep channel, the loop
	// will run until model.req is closed by the goroutine running predict.py
	go func() {
		for request := range m.req {
			_, err := socket.SendBytes(request, 0)
			if err != nil {
				log.Errorf("error sending data to model %v: %v", m.ID, err.Error())
			}
			resp, err := socket.RecvBytes(0)
			if err != nil {
				log.Errorf("error receiving data from model %v: %v", m.ID, err.Error())
			}
			m.rep <- resp
		}
		close(m.rep) // no more replies after req closed
	}()

	return nil
}

//-----------------------------------------------------------------------------
// Training Data Parsing
//-----------------------------------------------------------------------------

// ParseCSV parses a csv file with the following format:
//
//		<target_var>,<var_1>,<var_2>,...<var_n>
//		"true",1.5,"red",...
//
// returning a slice of maps representing the feature:value pairs for each row,
// a slice of labels, and an error. If the hasTarget flag is true, the first
// column of input data will be copied to the label slice and excluded from the
// feature:value pairs. If hasTarget is false, the label slice will be empty and
// all columns will be included in the feature:value pairs.
func ParseCSV(r io.Reader, hasTarget bool) (ModelReq, error) {
	reader := csv.NewReader(r)

	// grab the var names from the first row
	fieldNames, err := reader.Read()
	if err != nil {
		return ModelReq{}, err
	}

	xStart := 0 // column where feature data starts
	if hasTarget {
		xStart = 1
	}

	var d ModelReq

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return ModelReq{}, err
		}

		if len(row) != len(fieldNames) {
			return ModelReq{}, errors.New("mlserver: csv header and row length mismatch")
		}

		if hasTarget {
			// first column is the target variable
			d.Labels = append(d.Labels, row[0])
		}

		// save the rest as <feature_name>:<value> pairs
		features := make(map[string]interface{})
		for i := xStart; i < len(row); i++ {
			val := row[i]
			// check for numeric value
			numVal, err := strconv.ParseFloat(row[i], 64)
			if err != nil {
				features[fieldNames[i]] = val // use string val
			} else {
				features[fieldNames[i]] = numVal // use numeric val
			}
		}
		d.Data = append(d.Data, features)
	}

	return d, nil
}

var ErrCSVFileMissing = errors.New("csv file missing")

// parseFileUpload parses ModelReq from a csv file uploaded in a POST request.
// the hasTarget arg should be true when the uploaded csv file has the target
// variable in the first column (i.e. when parsing a request for fitting a model).
// ErrCSVFileMissing will be returned if there is no file associated with the key
// 'file'.
func parseFileUpload(r *http.Request, hasTarget bool) (ModelReq, error) {

	err := r.ParseMultipartForm(1 << 28)
	if err != nil {
		return ModelReq{}, err
	}

	defer func() {
		err := r.MultipartForm.RemoveAll()
		if err != nil {
			log.Error("error removing file uploads ", err)
		}
	}()

	files, ok := r.MultipartForm.File["file"]
	if !ok || len(files) < 1 {
		return ModelReq{}, ErrCSVFileMissing
	}

	f, err := files[0].Open()
	if err != nil {
		return ModelReq{}, err
	}
	defer f.Close()

	d, err := ParseCSV(f, hasTarget)
	if err != nil {
		return ModelReq{}, err
	}

	d.Name = strings.Join(r.MultipartForm.Value["name"], " ")

	return d, nil
}

//-----------------------------------------------------------------------------
// HTTP Handlers
//-----------------------------------------------------------------------------

// HandleModel is the http handler for requests made to /models/<id>, GET
// returns the model status, PUT/POST return predictions by the model. Other
// HTTP methods result in a Method Not Allowed response.
func HandleModel(w http.ResponseWriter, r *http.Request) {
	modelID := filepath.Base(r.URL.Path)

	switch r.Method {
	case "GET": // status
		m, err := Models.LoadModelData(modelID)
		if err == ErrModelNotFound {
			http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		writeJSONOK(w, m)

	case "PUT", "POST": // predict
		var err error

		m, err := Models.Get(modelID)
		if err == ErrModelNotFound {
			http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var newData ModelReq

		if r.Header.Get("Content-Type") == "application/json" {
			err = json.NewDecoder(r.Body).Decode(&newData)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		} else {
			newData, err = parseFileUpload(r, false)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
		}

		newData.ModelID = modelID

		pred := m.Predict(newData)
		writeJSONOK(w, pred)

	default:
		notAllowed(w)
	}

}

// HandleModels is the http handler for requests made to /models, POST
// fits a new model with the supplied data. Data for fitting the model can
// be encoded as JSON in the request body or uploaded as a csv file. GET responds
// with a list of all models in the index. Other HTTP methods result in a
// Method Not Allowed response.
func HandleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET": // list models
		writeJSONOK(w, Models.All())

	case "POST": // new model

		var trainData ModelReq
		var err error

		if r.Header.Get("Content-Type") == "application/json" {
			// try parsing as json
			err = json.NewDecoder(r.Body).Decode(&trainData)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}

		} else {
			// parse as csv file upload
			trainData, err = parseFileUpload(r, true)
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if trainData.Name == "" {
				http.Error(w, "missing model name", http.StatusBadRequest)
				return
			}
		}

		m := Models.NewModel()
		go fitModel(m, trainData)

		resp := struct {
			ModelID string `json:"model_id"`
		}{
			m.ID,
		}

		writeJSON(w, resp, http.StatusAccepted)

	default:
		notAllowed(w)
	}
}

// HandleRunningModels accepts GET, PUT, POST requests made to /models/running
// GET response with all running models, PUT/POST will start a model, the modelID
// should be passed as a json encoded object in the body of the request. All other
// methods result in a Method Not Allowed response.
func HandleRunningModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET": // list running models
		models := Models.All()
		runningModels := []*Model{}
		for _, model := range models {
			model.runLock.RLock()
			if model.Running {
				runningModels = append(runningModels, model)
			}
			model.runLock.RUnlock()
		}
		writeJSONOK(w, runningModels)

	case "PUT", "POST": // start a model
		var msg struct {
			ModelID string `json:"model_id"`
		}
		err := json.NewDecoder(r.Body).Decode(&msg)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		_, err = Models.Get(msg.ModelID)
		if err == ErrModelNotFound {
			http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusCreated)

	default:
		notAllowed(w)
	}
}

// HandleStopModel accepts DELETE requests made to /models/running/<id> and stops
// the model if it's currently running. All other request methods result in a
// Method Not Allowed response. If the model is not found, it will return 404
func HandleStopModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != "DELETE" {
		notAllowed(w)
		return
	}

	modelID := filepath.Base(r.URL.Path)
	m, err := Models.LoadModelData(modelID)
	if err == ErrModelNotFound {
		http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	err = m.Stop()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusAccepted)
}

func writeJSONOK(w http.ResponseWriter, v interface{}) {
	writeJSON(w, v, http.StatusOK)
}

func writeJSON(w http.ResponseWriter, v interface{}, status int) {

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	err := json.NewEncoder(w).Encode(v)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func notAllowed(w http.ResponseWriter) {
	http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
}

//-----------------------------------------------------------------------------
// Startup
//-----------------------------------------------------------------------------
func main() {
	Models = NewModelRepo("models")

	log.Info("started indexing model directory")
	Models.IndexModelDir()
	log.Info("finished indexing model directory")

	m := http.NewServeMux()
	m.HandleFunc("/models", HandleModels)
	m.HandleFunc("/models/", HandleModel)
	m.HandleFunc("/models/running", HandleRunningModels)
	m.HandleFunc("/models/running/", HandleStopModel)

	log.Info("listening on http://localhost:5000")
	log.Fatalln(http.ListenAndServe(":5000", requestLogger(m)))
}

//-----------------------------------------------------------------------------
// HTTP Request Logging
//-----------------------------------------------------------------------------
// mostly copied from github.com/wlattner/logger

// requestLogger wraps an http.Handler, logging all requests
func requestLogger(fn http.Handler) http.Handler {
	return logger{fn}
}

type logger struct {
	h http.Handler
}

func (l logger) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	resp := &responseLogger{w: w}
	l.h.ServeHTTP(resp, r)
	go printLog(r, resp.status, resp.size, time.Since(start))
}

// responseLogger allows us to trap the response size and status code
type responseLogger struct {
	w      http.ResponseWriter
	status int
	size   int
}

func (l *responseLogger) Header() http.Header {
	return l.w.Header()
}

// wrap ResponseWriter.Write to capture response size
func (l *responseLogger) Write(b []byte) (int, error) {
	if l.status == 0 {
		l.status = http.StatusOK
	}
	size, err := l.w.Write(b)
	l.size += size
	return size, err
}

// wrap ResponseWriter.WriteHeader to capture http status code
func (l *responseLogger) WriteHeader(s int) {
	l.w.WriteHeader(s)
	l.status = s
}

func printLog(req *http.Request, status int, size int, d time.Duration) {
	host, _, _ := net.SplitHostPort(req.RemoteAddr)
	requestTime := float64(d.Nanoseconds()) / 1e6
	// ip method path status size time
	// 0.0.0.0 GET /api/users 200 312 34
	log.Infof("%s %s %s %d %d %.2f",
		host,
		req.Method,
		req.URL.RequestURI(),
		status,
		size,
		requestTime,
	)
}
