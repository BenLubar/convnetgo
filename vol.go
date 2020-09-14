package convnet

import (
	"encoding/json"
	"math"
	"math/rand"
)

// Vol is the basic building block of all data in a net.
// it is essentially just a 3D volume of numbers, with a
// width (sx), height (sy), and depth (depth).
// it is used to hold data for all filters, all volumes,
// all weights, and also stores all gradients w.r.t.
// the data. c is optionally a value to initialize the volume
// with. If c is missing, fills the Vol with random numbers.
type Vol struct {
	Sx    int       `json:"sx"`
	Sy    int       `json:"sy"`
	Depth int       `json:"depth"`
	W     []float64 `json:"w"`
	Dw    []float64 `json:"-"`
}

func NewVol1D(w []float64) *Vol {
	v := &Vol{
		Sx:    1,
		Sy:    1,
		Depth: len(w),
		W:     make([]float64, len(w)),
		Dw:    make([]float64, len(w)),
	}

	copy(v.W, w)

	return v
}

func NewVol(sx, sy, depth int, c float64) *Vol {
	n := sx * sy * depth

	v := &Vol{
		Sx:    sx,
		Sy:    sy,
		Depth: depth,
		W:     make([]float64, n),
		Dw:    make([]float64, n),
	}

	for i := range v.W {
		v.W[i] = c
	}

	return v
}

func NewVolRand(sx, sy, depth int, r *rand.Rand) *Vol {
	n := sx * sy * depth

	v := &Vol{
		Sx:    sx,
		Sy:    sy,
		Depth: depth,
		W:     make([]float64, n),
		Dw:    make([]float64, n),
	}

	// weight normalization is done to equalize the output
	// variance of every neuron, otherwise neurons with a lot
	// of incoming connections have outputs of larger variance
	scale := math.Sqrt(1.0 / float64(sx*sy*depth))

	for i := range v.W {
		v.W[i] = r.NormFloat64() * scale
	}

	return v
}

func (v *Vol) index(x, y, d int) int {
	return ((v.Sx*y)+x)*v.Depth + d
}
func (v *Vol) Get(x, y, d int) float64 {
	return v.W[v.index(x, y, d)]
}
func (v *Vol) Set(x, y, d int, value float64) {
	v.W[v.index(x, y, d)] = value
}
func (v *Vol) Add(x, y, d int, value float64) {
	v.W[v.index(x, y, d)] += value
}
func (v *Vol) GetGrad(x, y, d int) float64 {
	return v.Dw[v.index(x, y, d)]
}
func (v *Vol) SetGrad(x, y, d int, value float64) {
	v.Dw[v.index(x, y, d)] = value
}
func (v *Vol) AddGrad(x, y, d int, value float64) {
	v.Dw[v.index(x, y, d)] += value
}
func (v *Vol) CloneAndZero() *Vol {
	return NewVol(v.Sx, v.Sy, v.Depth, 0.0)
}
func (v *Vol) Clone() *Vol {
	v2 := &Vol{
		Sx: v.Sx, Sy: v.Sy,
		Depth: v.Depth,
		W:     make([]float64, len(v.W)),
		Dw:    make([]float64, len(v.W)),
	}

	copy(v2.W, v.W)

	return v2
}
func (v *Vol) AddFrom(v2 *Vol) {
	for k := range v.W {
		v.W[k] += v2.W[k]
	}
}
func (v *Vol) AddFromScaled(v2 *Vol, a float64) {
	for k := range v.W {
		v.W[k] += a * v2.W[k]
	}
}
func (v *Vol) SetConst(a float64) {
	for k := range v.W {
		v.W[k] = a
	}
}

func (v *Vol) UnmarshalJSON(b []byte) error {
	var data struct {
		Sx    int       `json:"sx"`
		Sy    int       `json:"sy"`
		Depth int       `json:"depth"`
		W     []float64 `json:"w"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	v.Sx = data.Sx
	v.Sy = data.Sy
	v.Depth = data.Depth

	n := v.Sx * v.Sy * v.Depth
	v.W = make([]float64, n)
	v.Dw = make([]float64, n)

	copy(v.W, data.W)

	return nil
}
