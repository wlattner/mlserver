package main

import (
	"encoding/csv"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strconv"
	"strings"

	"github.com/coreos/go-log/log"
)

// ParseJSON parses a JSON encoded request:
//
//		{
//			"name": "iris model",
//			"data": [
//				{
//					"var_1": 2.5,
//					"var_2": 3.6,
//					...
//				},
//				...
//			],
//			"labels": [
//				"yes",
//				"no",
//				...
//			]
//		}
//
// into a ModelReq struct. If the hasTarget arg is true, ParseJSON will also set
// the isRegression attribute if the returned ModelReq if all the values in the
// label slice can be parsed as floats.
func ParseJSON(r io.Reader, hasTarget bool) (ModelReq, error) {
	var d ModelReq
	err := json.NewDecoder(r).Decode(&d)
	if err != nil {
		return ModelReq{}, err
	}

	// the json decoder will correctly parse string vs float for the label slice
	// check a few values to determine if this is a regression or classification
	// task
	if hasTarget {
		allFloats := true
		for _, val := range d.Labels {
			_, ok := val.(float64)
			if !ok {
				allFloats = false
				break
			}
		}
		d.isRegression = allFloats
	}

	return d, nil
}

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
	allFloats := true // regression if all labels are floats, classification otherwise

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

		if hasTarget { // first column is the target variable
			// check if float
			numVal, err := strconv.ParseFloat(row[0], 64)
			if err != nil {
				d.Labels = append(d.Labels, row[0])
				allFloats = false
			} else {
				d.Labels = append(d.Labels, numVal)
			}
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

	if hasTarget {
		d.isRegression = allFloats
	}

	return d, nil
}

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
		return ModelReq{}, errors.New("csv file missing")
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

// parseFitPredictRequest parses an http request into a ModelReq struct. The appropriate
// parser (json or csv) is determined from the content-type.
func parseFitPredictRequest(r *http.Request, isFitReq bool) (ModelReq, error) {
	if r.Header.Get("Content-Type") == "application/json" {
		return ParseJSON(r.Body, isFitReq)
	} else {
		return parseFileUpload(r, isFitReq)
	}
}
