package main

import (
	"encoding/json"
	"net"
	"net/http"
	"time"

	"github.com/coreos/go-log/log"
)

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
