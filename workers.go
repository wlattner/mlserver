package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/coreos/go-log/log"
	zmq "github.com/pebbe/zmq4"
)

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

func fitModel(m *Model, d ModelReq, r *ModelRepo) {
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
	_, err = r.LoadModelData(m.ID)
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
