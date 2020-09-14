package convnet

import (
	"encoding/json"
	"math"
	"math/rand"
)

// Implements ReLU nonlinearity elementwise
// x -> max(0, x)
// the output is in [0, inf)
type ReluLayer struct{}

/*
	TODO:
	var ReluLayer = function(opt) {
		var opt = opt || {};

		// computed
		this.out_sx = opt.in_sx;
		this.out_sy = opt.in_sy;
		this.out_depth = opt.in_depth;
		this.layer_type = 'relu';
	}
	ReluLayer.prototype = {
		forward: function(V, is_training) {
			this.in_act = V;
			var V2 = V.clone();
			var N = V.w.length;
			var V2w = V2.w;
			for(var i=0;i<N;i++) {
				if(V2w[i] < 0) V2w[i] = 0; // threshold at 0
			}
			this.out_act = V2;
			return this.out_act;
		},
		backward: function() {
			var V = this.in_act; // we need to set dw of this
			var V2 = this.out_act;
			var N = V.w.length;
			V.dw = global.zeros(N); // zero out gradient wrt data
			for(var i=0;i<N;i++) {
				if(V2.w[i] <= 0) V.dw[i] = 0; // threshold
				else V.dw[i] = V2.dw[i];
			}
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
			return json;
		},
		fromJSON: function(json) {
			this.out_depth = json.out_depth;
			this.out_sx = json.out_sx;
			this.out_sy = json.out_sy;
			this.layer_type = json.layer_type;
		}
	}
*/

// Implements Sigmoid nonlinearity elementwise
// x -> 1/(1+e^(-x))
// so the output is between 0 and 1.
type SigmoidLayer struct{}

/*
	TODO:
	var SigmoidLayer = function(opt) {
		var opt = opt || {};

		// computed
		this.out_sx = opt.in_sx;
		this.out_sy = opt.in_sy;
		this.out_depth = opt.in_depth;
		this.layer_type = 'sigmoid';
	}
	SigmoidLayer.prototype = {
		forward: function(V, is_training) {
			this.in_act = V;
			var V2 = V.cloneAndZero();
			var N = V.w.length;
			var V2w = V2.w;
			var Vw = V.w;
			for(var i=0;i<N;i++) {
				V2w[i] = 1.0/(1.0+Math.exp(-Vw[i]));
			}
			this.out_act = V2;
			return this.out_act;
		},
		backward: function() {
			var V = this.in_act; // we need to set dw of this
			var V2 = this.out_act;
			var N = V.w.length;
			V.dw = global.zeros(N); // zero out gradient wrt data
			for(var i=0;i<N;i++) {
				var v2wi = V2.w[i];
				V.dw[i] =  v2wi * (1.0 - v2wi) * V2.dw[i];
			}
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
			return json;
		},
		fromJSON: function(json) {
			this.out_depth = json.out_depth;
			this.out_sx = json.out_sx;
			this.out_sy = json.out_sy;
			this.layer_type = json.layer_type;
		}
	}
*/

// Implements Maxout nonlinearity that computes
// x -> max(x)
// where x is a vector of size group_size. Ideally of course,
// the input size should be exactly divisible by group_size
type MaxoutLayer struct{}

/*
	TODO:
	var MaxoutLayer = function(opt) {
		var opt = opt || {};

		// required
		this.group_size = typeof opt.group_size !== 'undefined' ? opt.group_size : 2;

		// computed
		this.out_sx = opt.in_sx;
		this.out_sy = opt.in_sy;
		this.out_depth = Math.floor(opt.in_depth / this.group_size);
		this.layer_type = 'maxout';

		this.switches = global.zeros(this.out_sx*this.out_sy*this.out_depth); // useful for backprop
	}
	MaxoutLayer.prototype = {
		forward: function(V, is_training) {
			this.in_act = V;
			var N = this.out_depth;
			var V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);

			// optimization branch. If we're operating on 1D arrays we dont have
			// to worry about keeping track of x,y,d coordinates inside
			// input volumes. In convnets we do :(
			if(this.out_sx === 1 && this.out_sy === 1) {
				for(var i=0;i<N;i++) {
					var ix = i * this.group_size; // base index offset
					var a = V.w[ix];
					var ai = 0;
					for(var j=1;j<this.group_size;j++) {
						var a2 = V.w[ix+j];
						if(a2 > a) {
							a = a2;
							ai = j;
						}
					}
					V2.w[i] = a;
					this.switches[i] = ix + ai;
				}
			} else {
				var n=0; // counter for switches
				for(var x=0;x<V.sx;x++) {
					for(var y=0;y<V.sy;y++) {
						for(var i=0;i<N;i++) {
							var ix = i * this.group_size;
							var a = V.get(x, y, ix);
							var ai = 0;
							for(var j=1;j<this.group_size;j++) {
								var a2 = V.get(x, y, ix+j);
								if(a2 > a) {
									a = a2;
									ai = j;
								}
							}
							V2.set(x,y,i,a);
							this.switches[n] = ix + ai;
							n++;
						}
					}
				}

			}
			this.out_act = V2;
			return this.out_act;
		},
		backward: function() {
			var V = this.in_act; // we need to set dw of this
			var V2 = this.out_act;
			var N = this.out_depth;
			V.dw = global.zeros(V.w.length); // zero out gradient wrt data

			// pass the gradient through the appropriate switch
			if(this.out_sx === 1 && this.out_sy === 1) {
				for(var i=0;i<N;i++) {
					var chain_grad = V2.dw[i];
					V.dw[this.switches[i]] = chain_grad;
				}
			} else {
				// bleh okay, lets do this the hard way
				var n=0; // counter for switches
				for(var x=0;x<V2.sx;x++) {
					for(var y=0;y<V2.sy;y++) {
						for(var i=0;i<N;i++) {
							var chain_grad = V2.get_grad(x,y,i);
							V.set_grad(x,y,this.switches[n],chain_grad);
							n++;
						}
					}
				}
			}
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
			json.group_size = this.group_size;
			return json;
		},
		fromJSON: function(json) {
			this.out_depth = json.out_depth;
			this.out_sx = json.out_sx;
			this.out_sy = json.out_sy;
			this.layer_type = json.layer_type;
			this.group_size = json.group_size;
			this.switches = global.zeros(this.group_size);
		}
	}
*/

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
