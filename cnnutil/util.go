// Package cnnutil contains various utility functions.
package cnnutil

import "fmt"

// Window stores _size_ number of values
// and returns averages. Useful for keeping running
// track of validation or training accuracy during SGD
type Window struct {
	v       []float64
	index   int
	size    int
	minsize int
}

func NewWindow(size, minsize int) *Window {
	return &Window{
		v:       make([]float64, 0, size),
		size:    size,
		minsize: minsize,
	}
}

func (w *Window) Add(x float64) {
	if len(w.v) < w.size {
		w.v = append(w.v, x)
	} else {
		w.v[w.index] = x
		w.index++

		if w.index >= w.size {
			w.index = 0
		}
	}
}
func (w *Window) Average() float64 {
	if len(w.v) < w.minsize {
		return -1
	}

	sum := 0.0

	for _, f := range w.v {
		sum += f
	}

	return sum / float64(len(w.v))
}
func (w *Window) Reset() {
	w.v = w.v[:0]
	w.index = 0
}

// returns min, max and indices of an array
func MaxMin(w []float64) (maxi int, maxv float64, mini int, minv, dv float64) {
	if len(w) == 0 {
		return // ... ;s
	}

	maxv, minv = w[0], w[0]
	maxi, mini = 0, 0

	for i := 1; i < len(w); i++ {
		if w[i] > maxv {
			maxv, maxi = w[i], i
		}
		if w[i] < minv {
			minv, mini = w[i], i
		}
	}

	return maxi, maxv, mini, minv, maxv - minv
}

// returns string representation of float
// but truncated to length of d digits
func F2T(x float64, d int) string {
	return fmt.Sprintf("%.*[2][1]f", x, d)
}
