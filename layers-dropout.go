package convnet

import (
	"encoding/json"
	"math/rand"
)

// An inefficient dropout layer
// Note this is not most efficient implementation since the layer before
// computed all these activations and now we're just going to drop them :(
// same goes for backward pass. Also, if we wanted to be efficient at test time
// we could equivalently be clever and upscale during train and copy pointers during test
type DropoutLayer struct {
	outSx    int
	outSy    int
	outDepth int
	dropProb float64
	dropped  []bool
	rand     *rand.Rand
	inAct    *Vol
	outAct   *Vol
}

func (l *DropoutLayer) OutDepth() int { return l.outDepth }
func (l *DropoutLayer) OutSx() int    { return l.outSx }
func (l *DropoutLayer) OutSy() int    { return l.outSy }
func (l *DropoutLayer) fromDef(def LayerDef, r *rand.Rand) {
	// computed
	l.outSx = def.InSx
	l.outSy = def.InSy
	l.outDepth = def.InDepth

	l.dropProb = def.DropProb
	if l.dropProb == 0.0 && !def.DropProbZero {
		l.dropProb = 0.5
	}

	l.dropped = make([]bool, l.outSx*l.outSy*l.outDepth)

	l.rand = r
}
func (l *DropoutLayer) ParamsAndGrads() []ParamsAndGrads { return nil }
func (l *DropoutLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v
	v2 := v.Clone()

	if isTraining {
		// do dropout
		for i := range v2.W {
			if l.rand.Float64() < l.dropProb {
				// drop!
				v2.W[i] = 0
				l.dropped[i] = true
			} else {
				l.dropped[i] = false
			}
		}
	} else {
		// scale the activations during prediction
		for i := range v2.W {
			v2.W[i] *= l.dropProb
		}
	}

	l.outAct = v2

	return l.outAct
}
func (l *DropoutLayer) Backward() {
	v := l.inAct // we need to set dw of this
	chainGrad := l.outAct

	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data
	for i := range v.Dw {
		if !l.dropped[i] {
			v.Dw[i] = chainGrad.Dw[i] // copy over the gradient
		}
	}
}
func (l *DropoutLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int     `json:"out_depth"`
		OutSx     int     `json:"out_sx"`
		OutSy     int     `json:"out_sy"`
		LayerType string  `json:"layer_type"`
		DropProb  float64 `json:"drop_prob"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerDropout.String(),
		DropProb:  l.dropProb,
	})
}
func (l *DropoutLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int     `json:"out_depth"`
		OutSx     int     `json:"out_sx"`
		OutSy     int     `json:"out_sy"`
		LayerType string  `json:"layer_type"`
		DropProb  float64 `json:"drop_prob"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth
	l.outSx = data.OutSx
	l.outSy = data.OutSy
	l.dropProb = data.DropProb

	return nil
}
