package main

import (
	"encoding/json"
	"net/http"
	"path/filepath"
)

type server struct {
	*ModelRepo
}

// NewAPIHandler returns an http.Handler for responding to api requests to
// mlserver. The ModelRepo parameter should be a pointer to an initialized
// and indexed ModelRepo.
func NewAPIHandler(r *ModelRepo) http.Handler {
	s := &server{r}

	m := http.NewServeMux()
	m.HandleFunc("/models", s.HandleModels)
	m.HandleFunc("/models/", s.HandleModel)
	m.HandleFunc("/models/running", s.HandleRunningModels)
	m.HandleFunc("/models/running/", s.HandleStopModel)

	return m
}

// HandleModel is the http handler for requests made to /models/<id>, GET
// returns the model status, PUT/POST return predictions by the model. Other
// HTTP methods result in a Method Not Allowed response.
func (s *server) HandleModel(w http.ResponseWriter, r *http.Request) {
	modelID := filepath.Base(r.URL.Path)

	switch r.Method {
	case "GET": // status
		m, err := s.LoadModelData(modelID)
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

		m, err := s.Get(modelID)
		if err == ErrModelNotFound {
			http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		newData, err := parseFitPredictRequest(r, false)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
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
func (s *server) HandleModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET": // list models
		writeJSONOK(w, s.All())

	case "POST": // new model

		trainData, err := parseFitPredictRequest(r, true)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		m := s.NewModel()
		go fitModel(m, trainData, s.ModelRepo)

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
func (s *server) HandleRunningModels(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case "GET": // list running models
		models := s.All()
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

		_, err = s.Get(msg.ModelID)
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
func (s *server) HandleStopModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != "DELETE" {
		notAllowed(w)
		return
	}

	modelID := filepath.Base(r.URL.Path)
	m, err := s.LoadModelData(modelID)
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
