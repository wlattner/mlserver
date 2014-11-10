package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"code.google.com/p/go-uuid/uuid"
	"github.com/coreos/go-log/log"
)

// Prediction is the parsed result from the Python worker
type Prediction struct {
	ModelID string               `json:"model_id"`
	Labels  []map[string]float64 `json:"labels"`
}

// ModelReq represents an incoming request for fit or predict
type ModelReq struct {
	ModelID      string                   `json:"model_id"`
	Name         string                   `json:"name"`
	Date         time.Time                `json:"created_at"`
	Data         []map[string]interface{} `json:"data"`
	Labels       []interface{}            `json:"labels"`
	isRegression bool
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
