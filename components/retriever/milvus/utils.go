package milvus

import (
	"encoding/json"
	"fmt"
)

func GetType() string {
	return typ
}

func tryMarshalJsonString(input any) string {
	if b, err := json.Marshal(input); err == nil {
		return string(b)
	}

	return ""
}

func interfaceTof64Slice(raw interface{}) ([]float64, error) {
	rawSlice, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("raw type not []interface, raw=%v", raw)
	}

	resp := make([]float64, len(rawSlice))
	for i := range rawSlice {
		f64, ok := rawSlice[i].(float64)
		if !ok {
			return nil, fmt.Errorf("item[%d] not float64, item=%v, raw slice=%v", i, rawSlice[i], raw)
		}

		resp[i] = f64
	}

	return resp, nil
}

func dereferenceOrZero[T any](v *T) T {
	if v == nil {
		var t T
		return t
	}

	return *v
}

func ptrOf[T any](v T) *T {
	return &v
}
