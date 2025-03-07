package milvus

import "fmt"

func chunk[T any](slice []T, size int) [][]T {
	if size <= 0 {
		return nil
	}

	var chunks [][]T
	for size < len(slice) {
		slice, chunks = slice[size:], append(chunks, slice[0:size:size])
	}

	if len(slice) > 0 {
		chunks = append(chunks, slice)
	}

	return chunks
}

func iter[T, D any](src []T, fn func(T) D) []D {
	resp := make([]D, len(src))
	for i := range src {
		resp[i] = fn(src[i])
	}

	return resp
}

func iterWithErr[T, D any](src []T, fn func(T) (D, error)) ([]D, error) {
	resp := make([]D, 0, len(src))
	for i := range src {
		d, err := fn(src[i])
		if err != nil {
			return nil, err
		}

		resp = append(resp, d)
	}

	return resp, nil
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

func interfaceToSparse(raw interface{}) (map[string]interface{}, error) {
	sparse, ok := raw.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("raw type not map[string]interface{}, raw=%v", raw)
	}

	return sparse, nil
}
