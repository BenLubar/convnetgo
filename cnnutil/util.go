// Package cnnutil contains various utility functions.
package cnnutil

// Window stores _size_ number of values
// and returns averages. Useful for keeping running
// track of validation or training accuracy during SGD
type Window struct {
	V       []float64
	Index   int
	Size    int
	MinSize int
}

func NewWindow(size, minsize int) *Window {
	return &Window{
		V:       make([]float64, 0, size),
		Size:    size,
		MinSize: minsize,
	}
}

func (w *Window) Add(x float64) {
	if len(w.V) < w.Size {
		w.V = append(w.V, x)
	} else {
		w.V[w.Index] = x
		w.Index++

		if w.Index >= w.Size {
			w.Index = 0
		}
	}
}
func (w *Window) Average() float64 {
	if len(w.V) < w.MinSize {
		return -1
	}

	sum := 0.0

	for _, f := range w.V {
		sum += f
	}

	return sum / float64(len(w.V))
}
func (w *Window) Reset() {
	w.V = w.V[:0]
	w.Index = 0
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
