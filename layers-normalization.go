package convnet

import (
	"encoding/json"
	"math"
	"math/rand"
)

// Local Response Normalization in window, along depths of volumes
type LocalResponseNormalizationLayer struct {
	k        float64
	alpha    float64
	beta     float64
	n        int
	outSx    int
	outSy    int
	outDepth int
	inAct    *Vol
	outAct   *Vol
	s        *Vol
}

func (l *LocalResponseNormalizationLayer) OutDepth() int { return l.outDepth }
func (l *LocalResponseNormalizationLayer) OutSx() int    { return l.outSx }
func (l *LocalResponseNormalizationLayer) OutSy() int    { return l.outSy }
func (l *LocalResponseNormalizationLayer) fromDef(def LayerDef, r *rand.Rand) {
	// required
	l.k = def.K
	l.n = def.N
	l.alpha = def.Alpha
	l.beta = def.Beta

	// computed
	l.outSx = def.InSx
	l.outSy = def.InSy
	l.outDepth = def.InDepth

	// checks
	if l.n%2 == 0 {
		panic("convnet: n should be odd for LRN layer")
	}
}
func (l *LocalResponseNormalizationLayer) ParamsAndGrads() []ParamsAndGrads { return nil }
func (l *LocalResponseNormalizationLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v

	a := v.CloneAndZero()
	l.s = v.CloneAndZero()
	n2 := l.n / 2

	for x := 0; x < v.Sx; x++ {
		for y := 0; y < v.Sy; y++ {
			for i := 0; i < v.Depth; i++ {
				ai := v.Get(x, y, i)

				// normalize in a window of size n
				den := 0.0
				min := i - n2
				if min < 0 {
					min = 0
				}
				max := i + n2
				if max >= v.Depth {
					max = v.Depth - 1
				}
				for j := min; j <= max; j++ {
					aa := v.Get(x, y, j)
					den += aa * aa
				}
				den *= l.alpha / float64(l.n)
				den += l.k
				l.s.Set(x, y, i, den) // will be useful for backprop
				den = math.Pow(den, l.beta)
				a.Set(x, y, i, ai/den)
			}
		}
	}

	l.outAct = a
	return l.outAct
}
func (l *LocalResponseNormalizationLayer) Backward() {
	// evaluate gradient wrt data
	v := l.inAct                     // we need to set dw of this
	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data
	a := l.outAct                    // computed in forward pass

	n2 := l.n / 2
	for x := 0; x < v.Sx; x++ {
		for y := 0; y < v.Sy; y++ {
			for i := 0; i < v.Depth; i++ {
				chainGrad := a.GetGrad(x, y, i)
				s := l.s.Get(x, y, i)
				sb := math.Pow(s, l.beta)
				sb2 := sb * sb

				// normalize in a window of size n
				min := i - n2
				if min < 0 {
					min = 0
				}

				max := i + n2
				if max >= v.Depth {
					max = v.Depth - 1
				}

				for j := min; j <= max; j++ {
					aj := v.Get(x, y, j)
					g := -aj * l.beta * math.Pow(s, l.beta-1) * l.alpha / float64(l.n) * 2 * aj

					if j == i {
						g += sb
					}

					g /= sb2
					g *= chainGrad
					v.AddGrad(x, y, j, g)
				}

			}
		}
	}
}
func (l *LocalResponseNormalizationLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		K         float64 `json:"k"`
		N         int     `json:"n"`
		Alpha     float64 `json:"alpha"`
		Beta      float64 `json:"beta"`
		OutSx     int     `json:"out_sx"`
		OutSy     int     `json:"out_sy"`
		OutDepth  int     `json:"out_depth"`
		LayerType string  `json:"layer_type"`
	}{
		K:         l.k,
		N:         l.n,
		Alpha:     l.alpha, // normalize by size
		Beta:      l.beta,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		OutDepth:  l.outDepth,
		LayerType: LayerLRN.String(),
	})
}
func (l *LocalResponseNormalizationLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		K         float64 `json:"k"`
		N         int     `json:"n"`
		Alpha     float64 `json:"alpha"`
		Beta      float64 `json:"beta"`
		OutSx     int     `json:"out_sx"`
		OutSy     int     `json:"out_sy"`
		OutDepth  int     `json:"out_depth"`
		LayerType string  `json:"layer_type"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.k = data.K
	l.n = data.N
	l.alpha = data.Alpha // normalize by size
	l.beta = data.Beta
	l.outSx = data.OutSx
	l.outSy = data.OutSy
	l.outDepth = data.OutDepth

	return nil
}
