package convnet

import (
	"encoding/json"
	"math"
	"math/rand"
)

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
type ReluLayer struct {
	outDepth int
	outSx    int
	outSy    int
	inAct    *Vol
	outAct   *Vol
}

func (l *ReluLayer) OutDepth() int { return l.outDepth }
func (l *ReluLayer) OutSx() int    { return l.outSx }
func (l *ReluLayer) OutSy() int    { return l.outSy }
func (l *ReluLayer) fromDef(def LayerDef, r *rand.Rand) {
	// computed
	l.outSx = def.InSx
	l.outSy = def.InSy
	l.outDepth = def.InDepth
}
func (l *ReluLayer) ParamsAndGrads() []ParamsAndGrads { return nil }
func (l *ReluLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v
	v2 := v.Clone()

	for i := range v2.W {
		if v2.W[i] < 0 {
			v2.W[i] = 0 // threshold at 0
		}
	}

	l.outAct = v2

	return l.outAct
}
func (l *ReluLayer) Backward() {
	v := l.inAct // we need to set dw of this
	v2 := l.outAct
	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data

	for i := range v.Dw {
		if v2.W[i] <= 0 {
			v.Dw[i] = 0 // threshold
		} else {
			v.Dw[i] = v2.Dw[i]
		}
	}
}
func (l *ReluLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerRelu.String(),
	})
}
func (l *ReluLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.outSx = data.OutSx
	l.outSy = data.OutSy

	return nil
}

// Implements Sigmoid nonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
type SigmoidLayer struct {
	outDepth int
	outSx    int
	outSy    int
	inAct    *Vol
	outAct   *Vol
}

func (l *SigmoidLayer) OutDepth() int { return l.outDepth }
func (l *SigmoidLayer) OutSx() int    { return l.outSx }
func (l *SigmoidLayer) OutSy() int    { return l.outSy }
func (l *SigmoidLayer) fromDef(def LayerDef, r *rand.Rand) {
	// computed
	l.outSx = def.InSx
	l.outSy = def.InSy
	l.outDepth = def.InDepth
}
func (l *SigmoidLayer) ParamsAndGrads() []ParamsAndGrads { panic("TODO") }
func (l *SigmoidLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v
	v2 := v.CloneAndZero()

	for i := range v2.W {
		v2.W[i] = 1.0 / (1.0 + math.Exp(-v.W[i]))
	}

	l.outAct = v2

	return l.outAct
}
func (l *SigmoidLayer) Backward() {
	v := l.inAct // we need to set dw of this
	v2 := l.outAct

	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data

	for i := range v.Dw {
		v.Dw[i] = v2.W[i] * (1.0 - v2.W[i]) * v2.Dw[i]
	}
}
func (l *SigmoidLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerSigmoid.String(),
	})
}
func (l *SigmoidLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.outSx = data.OutSx
	l.outSy = data.OutSy

	return nil
}

// Implements Maxout nonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size
type MaxoutLayer struct {
	groupSize int
	outDepth  int
	outSx     int
	outSy     int
	switches  []int
	inAct     *Vol
	outAct    *Vol
}

func (l *MaxoutLayer) OutDepth() int { panic("TODO") }
func (l *MaxoutLayer) OutSx() int    { panic("TODO") }
func (l *MaxoutLayer) OutSy() int    { panic("TODO") }
func (l *MaxoutLayer) fromDef(def LayerDef, r *rand.Rand) {
	// required
	l.groupSize = def.GroupSize
	if l.groupSize == 0 && !def.GroupSizeZero {
		l.groupSize = 2
	}

	// computed
	l.outSx = def.InSx
	l.outSy = def.InSy
	l.outDepth = def.InDepth / l.groupSize

	l.switches = make([]int, l.outSx*l.outSy*l.outDepth) // useful for backprop
}
func (l *MaxoutLayer) ParamsAndGrads() []ParamsAndGrads { panic("TODO") }
func (l *MaxoutLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v
	v2 := NewVol(l.outSx, l.outSy, l.outDepth, 0.0)

	// optimization branch. If we're operating on 1D arrays we dont have
	// to worry about keeping track of x,y,d coordinates inside
	// input volumes. In convnets we do :(
	if l.outSx == 1 && l.outSy == 1 {
		for i := 0; i < l.outDepth; i++ {
			ix := i * l.groupSize // base index offset
			a := v.W[ix]
			ai := 0

			for j := 1; j < l.groupSize; j++ {
				a2 := v.W[ix+j]

				if a2 > a {
					a = a2
					ai = j
				}
			}
			v2.W[i] = a
			l.switches[i] = ix + ai
		}
	} else {
		n := 0 // counter for switches

		for x := 0; x < v.Sx; x++ {
			for y := 0; y < v.Sy; y++ {
				for i := 0; i < l.outDepth; i++ {
					ix := i * l.groupSize
					a := v.Get(x, y, ix)
					ai := 0

					for j := 1; j < l.groupSize; j++ {
						a2 := v.Get(x, y, ix+j)

						if a2 > a {
							a = a2
							ai = j
						}
					}

					v2.Set(x, y, i, a)
					l.switches[n] = ix + ai

					n++
				}
			}
		}
	}

	l.outAct = v2

	return l.outAct
}
func (l *MaxoutLayer) Backward() {
	v := l.inAct // we need to set dw of this
	v2 := l.outAct
	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data

	// pass the gradient through the appropriate switch
	if l.outSx == 1 && l.outSy == 1 {
		for i := range v.Dw {
			chainGrad := v2.Dw[i]

			v.Dw[l.switches[i]] = chainGrad
		}
	} else {
		// bleh okay, lets do this the hard way
		n := 0 // counter for switches

		for x := 0; x < v2.Sx; x++ {
			for y := 0; y < v2.Sy; y++ {
				for i := 0; i < l.outDepth; i++ {
					chainGrad := v2.GetGrad(x, y, i)
					v.SetGrad(x, y, l.switches[n], chainGrad)

					n++
				}
			}
		}
	}
}
func (l *MaxoutLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		GroupSize int    `json:"group_size"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerMaxout.String(),
		GroupSize: l.groupSize,
	})
}
func (l *MaxoutLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		GroupSize int    `json:"group_size"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.outSx = data.OutSx
	l.outSy = data.OutSy
	l.groupSize = data.GroupSize
	l.switches = make([]int, l.outSx*l.outSy*l.outDepth)

	return nil
}

// Implements Tanh nnonlinearity elementwise
// x -> tanh(x)
// so the output is between -1 and 1.
type TanhLayer struct {
	outSx    int
	outSy    int
	outDepth int
	inAct    *Vol
	outAct   *Vol
}

func (l *TanhLayer) OutDepth() int { return l.outDepth }
func (l *TanhLayer) OutSx() int    { return l.outSx }
func (l *TanhLayer) OutSy() int    { return l.outSy }

func (l *TanhLayer) fromDef(def LayerDef, r *rand.Rand) {
	// computed
	l.outSx = def.InSx
	l.outSy = def.InSy
	l.outDepth = def.InDepth
}

func (l *TanhLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v
	v2 := v.CloneAndZero()

	for i := range v.W {
		v2.W[i] = math.Tanh(v.W[i])
	}

	l.outAct = v2

	return l.outAct
}
func (l *TanhLayer) Backward() {
	v := l.inAct // we need to set dw of this
	v2 := l.outAct

	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data

	for i := range v.W {
		v2wi := v2.W[i]
		v.Dw[i] = (1.0 - v2wi*v2wi) * v2.Dw[i]
	}
}
func (l *TanhLayer) ParamsAndGrads() []ParamsAndGrads { return nil }

func (l *TanhLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerTanh.String(),
	})
}
func (l *TanhLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.outSx = data.OutSx
	l.outSy = data.OutSy

	return nil
}
