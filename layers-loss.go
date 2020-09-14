package convnet

import (
	"encoding/json"
	"math"
	"math/rand"
)

// Layers that implement a loss. Currently these are the layers that
// can initiate a backward() pass. In future we probably want a more
// flexible system that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)
type SoftmaxLayer struct {
	outDepth int
	inAct    *Vol
	outAct   *Vol
	es       []float64
}

var _ LossLayer = (*SoftmaxLayer)(nil)

func (l *SoftmaxLayer) OutSx() int    { return 1 }
func (l *SoftmaxLayer) OutSy() int    { return 1 }
func (l *SoftmaxLayer) OutDepth() int { return l.outDepth }

func (l *SoftmaxLayer) fromDef(def LayerDef, r *rand.Rand) {
	l.outDepth = def.InSx * def.InSy * def.InDepth
}

func (l *SoftmaxLayer) Forward(v *Vol, isTraining bool) *Vol {
	a := NewVol(1, 1, l.outDepth, 0.0)

	// compute max activation
	as := v.W
	amax := v.W[0]
	for i := 1; i < l.outDepth; i++ {
		if as[i] > amax {
			amax = as[i]
		}
	}

	// compute exponentials (carefully to not blow up)
	es := make([]float64, l.outDepth)
	esum := 0.0
	for i := 0; i < l.outDepth; i++ {
		e := math.Exp(as[i] - amax)
		esum += e
		es[i] = e
	}

	// normalize and output to sum to one
	for i := 0; i < l.outDepth; i++ {
		es[i] /= esum
		a.W[i] = es[i]
	}

	l.es = es // save these for backprop
	l.outAct = a

	return l.outAct
}
func (l *SoftmaxLayer) Backward() {}
func (l *SoftmaxLayer) BackwardLoss(y LossData) float64 {
	// compute and accumulate gradient wrt weights and bias of this layer
	x := l.inAct
	// zero out the gradient of input Vol
	x.Dw = make([]float64, len(x.W))

	for i := 0; i < l.outDepth; i++ {
		indicator := 0.0
		if i == y.Dim {
			indicator = 1.0
		}

		mul := -(indicator - l.es[i])
		x.Dw[i] = mul
	}

	// loss is the class negative log likelihood
	return -math.Log(l.es[y.Dim])
}
func (l *SoftmaxLayer) ParamsAndGrads() []ParamsAndGrads { return nil }
func (l *SoftmaxLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		NumInputs int    `json:"num_inputs"`
	}{
		OutDepth:  l.outDepth,
		OutSx:     1,
		OutSy:     1,
		LayerType: LayerSoftmax.String(),
		NumInputs: l.outDepth,
	})
}
func (l *SoftmaxLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		NumInputs int    `json:"num_inputs"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.outDepth = data.OutDepth

	return nil
}

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.
type RegressionLayer struct {
	numInputs int
	act       *Vol
}

var _ LossLayer = (*RegressionLayer)(nil)

func (l *RegressionLayer) OutDepth() int { return l.numInputs }
func (l *RegressionLayer) OutSx() int    { return 1 }
func (l *RegressionLayer) OutSy() int    { return 1 }

func (l *RegressionLayer) fromDef(def LayerDef, r *rand.Rand) {
	// computed
	l.numInputs = def.InSx * def.InSy * def.InDepth
}

func (l *RegressionLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.act = v
	return v // identity function
}

func (l *RegressionLayer) Backward() {}

func (l *RegressionLayer) BackwardLoss(y LossData) float64 {
	// compute and accumulate gradient wrt weights and bias of this layer
	x := l.act
	x.Dw = make([]float64, len(x.W)) // zero out the gradient of input Vol

	i, yi := y.Dim, y.Val
	dy := x.W[i] - yi
	x.Dw[i] = dy

	return 0.5 * dy * dy
}
func (l *RegressionLayer) ParamsAndGrads() []ParamsAndGrads { return nil }

func (l *RegressionLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		NumInputs int    `json:"num_inputs"`
	}{
		OutDepth:  l.numInputs,
		OutSx:     1,
		OutSy:     1,
		LayerType: LayerRegression.String(),
		NumInputs: l.numInputs,
	})
}
func (l *RegressionLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		NumInputs int    `json:"num_inputs"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.numInputs = data.NumInputs

	return nil
}

type SVMLayer struct {
	numInputs int
	act       *Vol
}

var _ LossLayer = (*SVMLayer)(nil)

func (l *SVMLayer) OutDepth() int { return l.numInputs }
func (l *SVMLayer) OutSx() int    { return 1 }
func (l *SVMLayer) OutSy() int    { return 1 }

func (l *SVMLayer) fromDef(def LayerDef, r *rand.Rand) {
	// computed
	l.numInputs = def.InSx * def.InSy * def.InDepth
}

func (l *SVMLayer) Forward(v *Vol, isTraining bool) *Vol {
	l.act = v // nothing to do, output raw scores
	return v
}

func (l *SVMLayer) Backward() {}

func (l *SVMLayer) BackwardLoss(y LossData) float64 {
	// compute and accumulate gradient wrt weights and bias of this layer
	x := l.act
	x.Dw = make([]float64, len(x.W)) // zero out the gradient of input Vol

	// we're using structured loss here, which means that the score
	// of the ground truth should be higher than the score of any other
	// class, by a margin
	yscore := x.W[y.Dim] // score of ground truth
	margin := 1.0
	loss := 0.0

	for i := 0; i < l.numInputs; i++ {
		if y.Dim == i {
			continue
		}

		ydiff := -yscore + x.W[i] + margin
		if ydiff > 0 {
			// violating dimension, apply loss
			x.Dw[i] += 1
			x.Dw[y.Dim] -= 1
			loss += ydiff
		}
	}

	return loss
}
func (l *SVMLayer) ParamsAndGrads() []ParamsAndGrads { return nil }

func (l *SVMLayer) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		NumInputs int    `json:"num_inputs"`
	}{
		OutDepth:  l.numInputs,
		OutSx:     1,
		OutSy:     1,
		LayerType: LayerSVM.String(),
		NumInputs: l.numInputs,
	})
}
func (l *SVMLayer) UnmarshalJSON(b []byte) error {
	var data struct {
		OutDepth  int    `json:"out_depth"`
		OutSx     int    `json:"out_sx"`
		OutSy     int    `json:"out_sy"`
		LayerType string `json:"layer_type"`
		NumInputs int    `json:"num_inputs"`
	}

	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}

	l.numInputs = data.NumInputs

	return nil
}
