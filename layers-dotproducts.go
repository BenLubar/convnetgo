package convnet

import (
	"encoding/json"
	"math"
	"math/rand"
)

// This file contains all layers that do dot products with input,
// but usually in a different connectivity pattern and weight sharing
// schemes:
// - FullyConn is fully connected dot products
// - ConvLayer does convolutions (so weight sharing spatially)
// putting them together in one file because they are very similar

type ConvLayer struct {
	sx         int
	sy         int
	inSx       int
	inSy       int
	inDepth    int
	outSx      int
	outSy      int
	outDepth   int
	stride     int
	pad        int
	l1DecayMul float64
	l2DecayMul float64
	filters    []*Vol
	biases     *Vol
	inAct      *Vol
	outAct     *Vol
}

func (l *ConvLayer) OutDepth() int { return l.outDepth }
func (l *ConvLayer) OutSx() int    { return l.outSx }
func (l *ConvLayer) OutSy() int    { return l.outSy }
func (l *ConvLayer) fromDef(def LayerDef, r *rand.Rand) {
	// required
	l.outDepth = def.Filters
	l.sx = def.Sx // filter size. Should be odd if possible, it's cleaner.
	l.inDepth = def.InDepth
	l.inSx = def.InSx
	l.inSy = def.InSy

	// optional
	l.sy = def.Sy
	if l.sy == 0 && !def.SyZero {
		l.sy = l.sx
	}

	l.stride = def.Stride // stride at which we apply filters to input volume
	if l.stride == 0 && !def.StrideZero {
		l.stride = 1
	}

	l.pad = def.Pad // amount of 0 padding to add around borders of input volume
	l.l1DecayMul = def.L1DecayMul
	l.l2DecayMul = def.L2DecayMul

	if l.l2DecayMul == 0.0 && !def.L2DecayMulZero {
		l.l2DecayMul = 1.0
	}

	// computed
	// note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
	// volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
	// final application.
	l.outSx = (l.inSx+l.pad*2-l.sx)/l.stride + 1
	l.outSy = (l.inSy+l.pad*2-l.sy)/l.stride + 1

	// initializations
	l.filters = make([]*Vol, l.outDepth)

	for i := range l.filters {
		l.filters[i] = NewVolRand(l.sx, l.sy, l.inDepth, r)
	}

	l.biases = NewVol(1, 1, l.outDepth, def.BiasPref)
}
func (l *ConvLayer) ParamsAndGrads() []ParamsAndGrads {
	response := make([]ParamsAndGrads, 0, l.outDepth+1)

	for i := 0; i < l.outDepth; i++ {
		response = append(response, ParamsAndGrads{
			Params:     l.filters[i].W,
			Grads:      l.filters[i].Dw,
			L2DecayMul: l.l2DecayMul,
			L1DecayMul: l.l1DecayMul,
		})
	}

	response = append(response, ParamsAndGrads{
		Params:     l.biases.W,
		Grads:      l.biases.Dw,
		L1DecayMul: 0.0,
		L2DecayMul: 0.0,
	})

	return response
}
func (l *ConvLayer) Forward(v *Vol, isTraining bool) *Vol {
	// optimized code by @mdda that achieves 2x speedup over previous version

	l.inAct = v
	a := NewVol(l.outSx, l.outSy, l.outDepth, 0.0)

	for d := 0; d < l.outDepth; d++ {
		f := l.filters[d]
		y := -l.pad

		for ay := 0; ay < l.outSy; y, ay = y+l.stride, ay+1 { // l.stride
			x := -l.pad

			for ax := 0; ax < l.outSx; x, ax = x+l.stride, ax+1 { // l.stride
				// convolve centered at this particular location
				sum := 0.0

				for fy := 0; fy < f.Sy; fy++ {
					oy := y + fy // coordinates in the original input array coordinates

					for fx := 0; fx < f.Sx; fx++ {
						ox := x + fx

						if oy >= 0 && oy < v.Sy && ox >= 0 && ox < v.Sx {
							for fd := 0; fd < f.Depth; fd++ {
								sum += f.Get(fx, fy, fd) * v.Get(ox, oy, fd)
							}
						}
					}
				}

				sum += l.biases.W[d]

				a.Set(ax, ay, d, sum)
			}
		}
	}

	l.outAct = a

	return l.outAct
}
func (l *ConvLayer) Backward() {
	var V = l.inAct
	V.Dw = make([]float64, len(V.W)) // zero out gradient wrt bottom data, we're about to fill it

	for d := 0; d < l.outDepth; d++ {
		f := l.filters[d]
		y := -l.pad

		for ay := 0; ay < l.outSy; y, ay = y+l.stride, ay+1 {
			x := -l.pad

			for ax := 0; ax < l.outSx; x, ax = x+l.stride, ax+1 {
				// convolve centered at this particular location
				chainGrad := l.outAct.GetGrad(ax, ay, d) // gradient from above, from chain rule

				for fy := 0; fy < f.Sy; fy++ {
					oy := y + fy // coordinates in the original input array coordinates

					for fx := 0; fx < f.Sx; fx++ {
						ox := x + fx

						if oy >= 0 && oy < V.Sy && ox >= 0 && ox < V.Sx {
							for fd := 0; fd < f.Depth; fd++ {
								ix1 := V.index(ox, oy, fd)
								ix2 := f.index(fx, fy, fd)

								f.Dw[ix2] += V.W[ix1] * chainGrad
								V.Dw[ix1] += f.W[ix2] * chainGrad
							}
						}
					}
				}

				l.biases.Dw[d] += chainGrad
			}
		}
	}
}
func (l *ConvLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Sx         int     `json:"sx"`
		Sy         int     `json:"sy"`
		Stride     int     `json:"stride"`
		InDepth    int     `json:"in_depth"`
		OutDepth   int     `json:"out_depth"`
		OutSx      int     `json:"out_sx"`
		OutSy      int     `json:"out_sy"`
		LayerType  string  `json:"layer_type"`
		L1DecayMul float64 `json:"l1_decay_mul"`
		L2DecayMul float64 `json:"l2_decay_mul"`
		Pad        int     `json:"pad"`
		Filters    []*Vol  `json:"filters"`
		Biases     *Vol    `json:"biases"`
	}{
		Sx:         l.sx, // filter size in x, y dims
		Sy:         l.sy,
		Stride:     l.stride,
		InDepth:    l.inDepth,
		OutDepth:   l.outDepth,
		OutSx:      l.outSx,
		OutSy:      l.outSy,
		LayerType:  LayerConv.String(),
		L1DecayMul: l.l1DecayMul,
		L2DecayMul: l.l2DecayMul,
		Pad:        l.pad,
		Filters:    l.filters,
		Biases:     l.biases,
	})
}
func (l *ConvLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		Sx         int     `json:"sx"`
		Sy         int     `json:"sy"`
		Stride     int     `json:"stride"`
		InDepth    int     `json:"in_depth"`
		OutDepth   int     `json:"out_depth"`
		OutSx      int     `json:"out_sx"`
		OutSy      int     `json:"out_sy"`
		LayerType  string  `json:"layer_type"`
		L1DecayMul float64 `json:"l1_decay_mul"`
		L2DecayMul float64 `json:"l2_decay_mul"`
		Pad        int     `json:"pad"`
		Filters    []*Vol  `json:"filters"`
		Biases     *Vol    `json:"biases"`
	}

	data.L1DecayMul = 1.0
	data.L2DecayMul = 1.0

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.outSx = data.OutSx
	l.outSy = data.OutSy
	l.sx = data.Sx // filter size in x, y dims
	l.sy = data.Sy
	l.stride = data.Stride
	l.inDepth = data.InDepth // depth of input volume
	l.l1DecayMul = data.L1DecayMul
	l.l2DecayMul = data.L2DecayMul
	l.pad = data.Pad
	l.filters = data.Filters
	l.biases = data.Biases

	return nil
}

type FullyConnLayer struct {
	outDepth   int
	l1DecayMul float64
	l2DecayMul float64
	numInputs  int
	filters    []*Vol
	biases     *Vol
	inAct      *Vol
	outAct     *Vol
}

func (l *FullyConnLayer) OutSx() int    { return 1 }
func (l *FullyConnLayer) OutSy() int    { return 1 }
func (l *FullyConnLayer) OutDepth() int { return l.outDepth }

func (l *FullyConnLayer) fromDef(def LayerDef, r *rand.Rand) {
	// required
	l.outDepth = def.NumNeurons

	// optional
	l.l1DecayMul = def.L1DecayMul
	l.l2DecayMul = def.L2DecayMul

	if l.l2DecayMul == 0 && !def.L2DecayMulZero {
		l.l2DecayMul = 1.0
	}

	// computed
	l.numInputs = def.InSx * def.InSy * def.InDepth

	// initializations
	bias := def.BiasPref
	l.filters = make([]*Vol, l.outDepth)

	for i := 0; i < l.outDepth; i++ {
		l.filters[i] = NewVolRand(1, 1, l.numInputs, r)
	}

	l.biases = NewVol(1, 1, l.outDepth, bias)
}
func (l *FullyConnLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v
	a := NewVol(1, 1, l.outDepth, 0.0)

	for i, f := range l.filters {
		sum0, sum1, sum2, sum3 := 0.0, 0.0, 0.0, 0.0

		// unrolled dot product
		d := 0
		for ; d < l.numInputs&^3; d += 4 {
			sum0 = math.FMA(v.W[d], f.W[d], sum0)
			sum1 = math.FMA(v.W[d+1], f.W[d+1], sum1)
			sum2 = math.FMA(v.W[d+2], f.W[d+2], sum2)
			sum3 = math.FMA(v.W[d+3], f.W[d+3], sum3)
		}

		sum := sum0 + sum1 + sum2 + sum3

		// finish any remaining elements
		for ; d < l.numInputs; d++ {
			sum = math.FMA(v.W[d], f.W[d], sum)
		}

		sum += l.biases.W[i]
		a.W[i] = sum
	}

	l.outAct = a

	return l.outAct
}
func (l *FullyConnLayer) Backward() {
	v := l.inAct
	v.Dw = make([]float64, len(v.W)) // zero out the gradient in input Vol

	// compute gradient wrt weights and data
	for i, f := range l.filters {
		chainGrad := l.outAct.Dw[i]

		for d := 0; d < l.numInputs; d++ {
			v.Dw[d] = math.FMA(f.W[d], chainGrad, v.Dw[d]) // grad wrt input data
			f.Dw[d] = math.FMA(v.W[d], chainGrad, f.Dw[d]) // grad wrt params
		}

		l.biases.Dw[i] += chainGrad
	}
}
func (l *FullyConnLayer) ParamsAndGrads() []ParamsAndGrads {
	response := make([]ParamsAndGrads, 0, l.outDepth+1)

	for _, f := range l.filters {
		response = append(response, ParamsAndGrads{
			Params:     f.W,
			Grads:      f.Dw,
			L1DecayMul: l.l1DecayMul,
			L2DecayMul: l.l2DecayMul,
		})
	}

	response = append(response, ParamsAndGrads{
		Params:     l.biases.W,
		Grads:      l.biases.Dw,
		L1DecayMul: 0.0,
		L2DecayMul: 0.0,
	})

	return response
}
func (l *FullyConnLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth   int     `json:"out_depth"`
		OutSx      int     `json:"out_sx"`
		OutSy      int     `json:"out_sy"`
		LayerType  string  `json:"layer_type"`
		NumInputs  int     `json:"num_inputs"`
		L1DecayMul float64 `json:"l1_decay_mul"`
		L2DecayMul float64 `json:"l2_decay_mul"`
		Filters    []*Vol  `json:"filters"`
		Biases     *Vol    `json:"biases"`
	}{
		OutDepth:   l.outDepth,
		OutSx:      1,
		OutSy:      1,
		LayerType:  LayerFC.String(),
		NumInputs:  l.numInputs,
		L1DecayMul: l.l1DecayMul,
		L2DecayMul: l.l2DecayMul,
		Filters:    l.filters,
		Biases:     l.biases,
	})
}
func (l *FullyConnLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth   int     `json:"out_depth"`
		OutSx      int     `json:"out_sx"`
		OutSy      int     `json:"out_sy"`
		LayerType  string  `json:"layer_type"`
		NumInputs  int     `json:"num_inputs"`
		L1DecayMul float64 `json:"l1_decay_mul"`
		L2DecayMul float64 `json:"l2_decay_mul"`
		Filters    []*Vol  `json:"filters"`
		Biases     *Vol    `json:"biases"`
	}

	data.L1DecayMul = 1.0
	data.L2DecayMul = 1.0

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.numInputs = data.NumInputs
	l.l1DecayMul = data.L1DecayMul
	l.l2DecayMul = data.L2DecayMul
	l.filters = data.Filters
	l.biases = data.Biases

	return nil
}
