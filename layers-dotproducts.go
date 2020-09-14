package convnet

import (
	"encoding/json"
	"math/rand"
)

/*
TODO:
(function(global) {
  "use strict";
  var Vol = global.Vol; // convenience

  // This file contains all layers that do dot products with input,
  // but usually in a different connectivity pattern and weight sharing
  // schemes:
  // - FullyConn is fully connected dot products
  // - ConvLayer does convolutions (so weight sharing spatially)
  // putting them together in one file because they are very similar
  var ConvLayer = function(opt) {
    var opt = opt || {};

    // required
    this.out_depth = opt.filters;
    this.sx = opt.sx; // filter size. Should be odd if possible, it's cleaner.
    this.in_depth = opt.in_depth;
    this.in_sx = opt.in_sx;
    this.in_sy = opt.in_sy;

    // optional
    this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
    this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 1; // stride at which we apply filters to input volume
    this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0; // amount of 0 padding to add around borders of input volume
    this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
    this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

    // computed
    // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
    // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
    // final application.
    this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
    this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
    this.layer_type = 'conv';

    // initializations
    var bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
    this.filters = [];
    for(var i=0;i<this.out_depth;i++) { this.filters.push(new Vol(this.sx, this.sy, this.in_depth)); }
    this.biases = new Vol(1, 1, this.out_depth, bias);
  }
  ConvLayer.prototype = {
    forward: function(V, is_training) {
      // optimized code by @mdda that achieves 2x speedup over previous version

      this.in_act = V;
      var A = new Vol(this.out_sx |0, this.out_sy |0, this.out_depth |0, 0.0);

      var V_sx = V.sx |0;
      var V_sy = V.sy |0;
      var xy_stride = this.stride |0;

      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad |0;
        var y = -this.pad |0;
        for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
            var a = 0.0;
            for(var fy=0;fy<f.sy;fy++) {
              var oy = y+fy; // coordinates in the original input array coordinates
              for(var fx=0;fx<f.sx;fx++) {
                var ox = x+fx;
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(var fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    a += f.w[((f.sx * fy)+fx)*f.depth+fd] * V.w[((V_sx * oy)+ox)*V.depth+fd];
                  }
                }
              }
            }
            a += this.biases.w[d];
            A.set(ax, ay, d, a);
          }
        }
      }
      this.out_act = A;
      return this.out_act;
    },
    backward: function() {

      var V = this.in_act;
      V.dw = global.zeros(V.w.length); // zero out gradient wrt bottom data, we're about to fill it

      var V_sx = V.sx |0;
      var V_sy = V.sy |0;
      var xy_stride = this.stride |0;

      for(var d=0;d<this.out_depth;d++) {
        var f = this.filters[d];
        var x = -this.pad |0;
        var y = -this.pad |0;
        for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
          x = -this.pad |0;
          for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

            // convolve centered at this particular location
            var chain_grad = this.out_act.get_grad(ax,ay,d); // gradient from above, from chain rule
            for(var fy=0;fy<f.sy;fy++) {
              var oy = y+fy; // coordinates in the original input array coordinates
              for(var fx=0;fx<f.sx;fx++) {
                var ox = x+fx;
                if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                  for(var fd=0;fd<f.depth;fd++) {
                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                    var ix1 = ((V_sx * oy)+ox)*V.depth+fd;
                    var ix2 = ((f.sx * fy)+fx)*f.depth+fd;
                    f.dw[ix2] += V.w[ix1]*chain_grad;
                    V.dw[ix1] += f.w[ix2]*chain_grad;
                  }
                }
              }
            }
            this.biases.dw[d] += chain_grad;
          }
        }
      }
    },
    getParamsAndGrads: function() {
      var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
    },
    toJSON: function() {
      var json = {};
      json.sx = this.sx; // filter size in x, y dims
      json.sy = this.sy;
      json.stride = this.stride;
      json.in_depth = this.in_depth;
      json.out_depth = this.out_depth;
      json.out_sx = this.out_sx;
      json.out_sy = this.out_sy;
      json.layer_type = this.layer_type;
      json.l1_decay_mul = this.l1_decay_mul;
      json.l2_decay_mul = this.l2_decay_mul;
      json.pad = this.pad;
      json.filters = [];
      for(var i=0;i<this.filters.length;i++) {
        json.filters.push(this.filters[i].toJSON());
      }
      json.biases = this.biases.toJSON();
      return json;
    },
    fromJSON: function(json) {
      this.out_depth = json.out_depth;
      this.out_sx = json.out_sx;
      this.out_sy = json.out_sy;
      this.layer_type = json.layer_type;
      this.sx = json.sx; // filter size in x, y dims
      this.sy = json.sy;
      this.stride = json.stride;
      this.in_depth = json.in_depth; // depth of input volume
      this.filters = [];
      this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
      this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
      this.pad = typeof json.pad !== 'undefined' ? json.pad : 0;
      for(var i=0;i<json.filters.length;i++) {
        var v = new Vol(0,0,0,0);
        v.fromJSON(json.filters[i]);
        this.filters.push(v);
      }
      this.biases = new Vol(0,0,0,0);
      this.biases.fromJSON(json.biases);
    }
  }
*/

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
		sum := 0.0

		for d := 0; d < l.numInputs; d++ {
			sum += v.W[d] * f.W[d] // for efficiency use Vols directly for now
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
			v.Dw[d] += f.W[d] * chainGrad // grad wrt input data
			f.Dw[d] += v.W[d] * chainGrad // grad wrt params
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
