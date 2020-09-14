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
type RegressionLayer struct{}

/*
TODO:
var RegressionLayer = function(opt) {
	var opt = opt || {};

	// computed
	this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
	this.out_depth = this.num_inputs;
	this.out_sx = 1;
	this.out_sy = 1;
	this.layer_type = 'regression';
}

RegressionLayer.prototype = {
	forward: function(V, is_training) {
		this.in_act = V;
		this.out_act = V;
		return V; // identity function
	},
	// y is a list here of size num_inputs
	// or it can be a number if only one value is regressed
	// or it can be a struct {dim: i, val: x} where we only want to
	// regress on dimension i and asking it to have value x
	backward: function(y) {

		// compute and accumulate gradient wrt weights and bias of this layer
		var x = this.in_act;
		x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol
		var loss = 0.0;
		if(y instanceof Array || y instanceof Float64Array) {
			for(var i=0;i<this.out_depth;i++) {
				var dy = x.w[i] - y[i];
				x.dw[i] = dy;
				loss += 0.5*dy*dy;
			}
		} else if(typeof y === 'number') {
			// lets hope that only one number is being regressed
			var dy = x.w[0] - y;
			x.dw[0] = dy;
			loss += 0.5*dy*dy;
		} else {
			// assume it is a struct with entries .dim and .val
			// and we pass gradient only along dimension dim to be equal to val
			var i = y.dim;
			var yi = y.val;
			var dy = x.w[i] - yi;
			x.dw[i] = dy;
			loss += 0.5*dy*dy;
		}
		return loss;
	},
	getParamsAndGrads: function() {
		return [];
	},
	toJSON: function() {
		var json = {};
		json.out_depth = this.out_depth;
		json.out_sx = this.out_sx;
		json.out_sy = this.out_sy;
		json.layer_type = this.layer_type;
		json.num_inputs = this.num_inputs;
		return json;
	},
	fromJSON: function(json) {
		this.out_depth = json.out_depth;
		this.out_sx = json.out_sx;
		this.out_sy = json.out_sy;
		this.layer_type = json.layer_type;
		this.num_inputs = json.num_inputs;
	}
}
*/

type SVMLayer struct{}

/*
TODO:
var SVMLayer = function(opt) {
	var opt = opt || {};

	// computed
	this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
	this.out_depth = this.num_inputs;
	this.out_sx = 1;
	this.out_sy = 1;
	this.layer_type = 'svm';
}

SVMLayer.prototype = {
	forward: function(V, is_training) {
		this.in_act = V;
		this.out_act = V; // nothing to do, output raw scores
		return V;
	},
	backward: function(y) {

		// compute and accumulate gradient wrt weights and bias of this layer
		var x = this.in_act;
		x.dw = global.zeros(x.w.length); // zero out the gradient of input Vol

		// we're using structured loss here, which means that the score
		// of the ground truth should be higher than the score of any other
		// class, by a margin
		var yscore = x.w[y]; // score of ground truth
		var margin = 1.0;
		var loss = 0.0;
		for(var i=0;i<this.out_depth;i++) {
			if(y === i) { continue; }
			var ydiff = -yscore + x.w[i] + margin;
			if(ydiff > 0) {
				// violating dimension, apply loss
				x.dw[i] += 1;
				x.dw[y] -= 1;
				loss += ydiff;
			}
		}

		return loss;
	},
	getParamsAndGrads: function() {
		return [];
	},
	toJSON: function() {
		var json = {};
		json.out_depth = this.out_depth;
		json.out_sx = this.out_sx;
		json.out_sy = this.out_sy;
		json.layer_type = this.layer_type;
		json.num_inputs = this.num_inputs;
		return json;
	},
	fromJSON: function(json) {
		this.out_depth = json.out_depth;
		this.out_sx = json.out_sx;
		this.out_sy = json.out_sy;
		this.layer_type = json.layer_type;
		this.num_inputs = json.num_inputs;
	}
}
*/
