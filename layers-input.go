package convnet

import (
	"encoding/json"
	"math/rand"
)

type InputLayer struct {
	outDepth int
	outSx    int
	outSy    int

	act *Vol
}

func (l *InputLayer) OutDepth() int { return l.outDepth }
func (l *InputLayer) OutSx() int    { return l.outSx }
func (l *InputLayer) OutSy() int    { return l.outSy }

func (l *InputLayer) fromDef(def LayerDef, r *rand.Rand) {
	// required: depth
	l.outDepth = def.OutDepth

	// optional: default these dimensions to 1
	l.outSx = def.OutSx
	l.outSy = def.OutSy

	if l.outSx == 0 {
		l.outSx = 1
	}

	if l.outSy == 0 {
		l.outSy = 1
	}
}

func (l *InputLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.act = v

	return l.act // simply identity function for now
}

func (l *InputLayer) Backward()                        {}
func (l *InputLayer) ParamsAndGrads() []ParamsAndGrads { return nil }

func (l *InputLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     l.outSx,
		OutSy:     l.outSy,
		LayerType: LayerInput.String(),
	})
}

func (l *InputLayer) UnmarshalJSON(b []byte) error {
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
