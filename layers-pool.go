package convnet

import (
	"encoding/json"
	"math/rand"
)

type PoolLayer struct {
	sx      int
	sy      int
	inDepth int
	inSx    int
	inSy    int
	outSx   int
	outSy   int
	stride  int
	pad     int
	switchx []int
	switchy []int
	inAct   *Vol
	outAct  *Vol
}

func (l *PoolLayer) OutDepth() int { return l.inDepth }
func (l *PoolLayer) OutSx() int    { return l.outSx }
func (l *PoolLayer) OutSy() int    { return l.outSy }

func (l *PoolLayer) fromDef(def LayerDef, r *rand.Rand) {
	// required
	l.sx = def.Sx // filter size
	l.inDepth = def.InDepth
	l.inSx = def.InSx
	l.inSy = def.InSy

	// optional
	l.sy = def.Sy
	if l.sy == 0 && !def.SyZero {
		l.sy = def.Sx
	}

	l.stride = def.Stride
	if l.stride == 0 && !def.StrideZero {
		l.stride = 2
	}

	l.pad = def.Pad // amount of 0 padding to add around borders of input volume

	// computed
	l.outSx = (l.inSx+l.pad*2-l.sx)/l.stride + 1
	l.outSy = (l.inSy+l.pad*2-l.sy)/l.stride + 1

	// store switches for x,y coordinates for where the max comes from, for each output neuron
	l.switchx = make([]int, l.outSx*l.outSy*l.inDepth)
	l.switchy = make([]int, l.outSx*l.outSy*l.inDepth)
}
func (l *PoolLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.inAct = v

	a := NewVol(l.outSx, l.outSy, l.inDepth, 0.0)

	n := 0 // a counter for switches

	for d := 0; d < l.inDepth; d++ {
		x := -l.pad

		for ax := 0; ax < l.outSx; x, ax = x+l.stride, ax+1 {
			y := -l.pad

			for ay := 0; ay < l.outSy; y, ay = y+l.stride, ay+1 {
				// convolve centered at this particular location
				bestValue := -99999.0 // hopefully small enough ;\
				winx, winy := -1, -1

				for fx := 0; fx < l.sx; fx++ {
					for fy := 0; fy < l.sy; fy++ {
						ox, oy := x+fx, y+fy

						if oy >= 0 && oy < v.Sy && ox >= 0 && ox < v.Sx {
							value := v.Get(ox, oy, d)

							// perform max pooling and store pointers to where
							// the max came from. This will speed up backprop
							// and can help make nice visualizations in future
							if value > bestValue {
								bestValue = value
								winx = ox
								winy = oy
							}
						}
					}
				}

				l.switchx[n] = winx
				l.switchy[n] = winy
				n++

				a.Set(ax, ay, d, bestValue)
			}
		}
	}

	l.outAct = a

	return l.outAct
}
func (l *PoolLayer) Backward() {
	// pooling layers have no parameters, so simply compute
	// gradient wrt data here
	v := l.inAct
	v.Dw = make([]float64, len(v.W)) // zero out gradient wrt data

	n := 0
	for d := 0; d < l.inDepth; d++ {
		x := -l.pad

		for ax := 0; ax < l.outSx; x, ax = x+l.stride, ax+1 {
			y := -l.pad

			for ay := 0; ay < l.outSy; y, ay = y+l.stride, ay+1 {
				chainGrad := l.outAct.GetGrad(ax, ay, d)

				v.AddGrad(l.switchx[n], l.switchy[n], d, chainGrad)

				n++
			}
		}
	}
}
func (l *PoolLayer) ParamsAndGrads() []ParamsAndGrads { return nil }
func (l *PoolLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Sx        int    `json:"sx"`
		Sy        int    `json:"sy"`
		Stride    int    `json:"stride"`
		InDepth   int    `json:"in_depth"`
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		Pad       int    `json:"pad"`
	}{
		Sx:        l.sx,
		Sy:        l.sy,
		Stride:    l.stride,
		InDepth:   l.inDepth,
		OutDepth:  l.inDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerPool.String(),
		Pad:       l.pad,
	})
}
func (l *PoolLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		Sx        int    `json:"sx"`
		Sy        int    `json:"sy"`
		Stride    int    `json:"stride"`
		InDepth   int    `json:"in_depth"`
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		Pad       int    `json:"pad"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outSx = data.OutSx
	l.outSy = data.OutSy
	l.sx = data.Sx
	l.sy = data.Sy
	l.stride = data.Stride
	l.inDepth = data.InDepth
	l.pad = data.Pad

	// need to re-init these appropriately
	l.switchx = make([]int, l.outSx*l.outSy*l.inDepth)
	l.switchy = make([]int, l.outSx*l.outSy*l.inDepth)

	return nil
}
