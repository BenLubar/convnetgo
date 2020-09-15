//go:generate stringer -type LayerType -linecomment

package convnet

import (
	"encoding/json"
	"fmt"
	"math/rand"
)

type LayerType int

const (
	LayerInput      LayerType = iota + 1 // input
	LayerRelu                            // relu
	LayerSigmoid                         // sigmoid
	LayerTanh                            // tanh
	LayerDropout                         // dropout
	LayerConv                            // conv
	LayerPool                            // pool
	LayerLRN                             // lrn
	LayerSoftmax                         // softmax
	LayerRegression                      // regression
	LayerFC                              // fc
	LayerMaxout                          // maxout
	LayerSVM                             // svm
)

type LayerDef struct {
	Type           LayerType `json:"type"`
	NumNeurons     int       `json:"num_neurons"`
	NumClasses     int       `json:"num_classes"`
	BiasPref       float64   `json:"bias_pref"`
	BiasPrefZero   bool      `json:"-"`
	Activation     LayerType `json:"activation"`
	GroupSize      int       `json:"group_size"`
	GroupSizeZero  bool      `json:"-"`
	DropProb       float64   `json:"drop_prob"`
	DropProbZero   bool      `json:"-"`
	InSx           int       `json:"in_sx"`
	InSy           int       `json:"in_sy"`
	InDepth        int       `json:"in_depth"`
	OutSx          int       `json:"out_sx"`
	OutSy          int       `json:"out_sy"`
	OutDepth       int       `json:"out_depth"`
	L1DecayMul     float64   `json:"l1_decay_mul"`
	L1DecayMulZero bool      `json:"-"`
	L2DecayMul     float64   `json:"l2_decay_mul"`
	L2DecayMulZero bool      `json:"-"`
	Sx             int       `json:"sx"`
	SxZero         bool      `json:"-"`
	Sy             int       `json:"sy"`
	SyZero         bool      `json:"-"`
	Pad            int       `json:"pad"`
	PadZero        bool      `json:"-"`
	Stride         int       `json:"stride"`
	StrideZero     bool      `json:"-"`
	Filters        int       `json:"filters"`
	K              float64   `json:"k"`
	N              int       `json:"n"`
	Alpha          float64   `json:"alpha"`
	Beta           float64   `json:"beta"`
}

type Layer interface {
	OutSx() int
	OutSy() int
	OutDepth() int

	Forward(v *Vol, isTraining bool) *Vol
	Backward()
	ParamsAndGrads() []ParamsAndGrads

	fromDef(LayerDef, *rand.Rand)
	json.Marshaler
	json.Unmarshaler
}

type LossData struct {
	Dim int
	Val float64
}

type LossLayer interface {
	Layer
	BackwardLoss(y LossData) float64
}

type ParamsAndGrads struct {
	Params     []float64
	Grads      []float64
	L1DecayMul float64
	L2DecayMul float64
}

// Net manages a set of layers
// For now constraints: Simple linear order of layers, first layer input last layer a cost layer
type Net struct {
	Layers []Layer `json:"layers"`
}

// desugar layer_defs for adding activation, dropout layers etc
func desugar(defs []LayerDef) []LayerDef {
	var newDefs []LayerDef
	for _, def := range defs {
		if def.Type == LayerSoftmax || def.Type == LayerSVM {
			// add an fc layer here, there is no reason the user should
			// have to worry about this and we almost always want to
			newDefs = append(newDefs, LayerDef{Type: LayerFC, NumNeurons: def.NumClasses})
		}

		if def.Type == LayerRegression {
			// add an fc layer here, there is no reason the user should
			// have to worry about this and we almost always want to
			newDefs = append(newDefs, LayerDef{Type: LayerFC, NumNeurons: def.NumNeurons})
		}

		if (def.Type == LayerFC || def.Type == LayerConv) && def.BiasPref == 0 && !def.BiasPrefZero {
			def.BiasPref = 0.0
			def.BiasPrefZero = true

			if def.Activation != 0 && def.Activation == LayerRelu {
				// relus like a bit of positive bias to get gradients early
				// otherwise it's technically possible that a relu unit will never turn on (by chance)
				// and will never get any gradient and never contribute any computation. Dead relu.
				def.BiasPref = 0.1
			}
		}

		newDefs = append(newDefs, def)

		if def.Activation != 0 {
			switch def.Activation {
			case LayerRelu:
				newDefs = append(newDefs, LayerDef{Type: LayerRelu})
			case LayerSigmoid:
				newDefs = append(newDefs, LayerDef{Type: LayerSigmoid})
			case LayerTanh:
				newDefs = append(newDefs, LayerDef{Type: LayerTanh})
			case LayerMaxout:
				// create maxout activation, and pass along group size, if provided
				gs := def.GroupSize
				if def.GroupSize == 0 && !def.GroupSizeZero {
					gs = 2
				}
				newDefs = append(newDefs, LayerDef{Type: LayerMaxout, GroupSize: gs, GroupSizeZero: true})
			default:
				panic("convnet: unsupported activation " + def.Activation.String())
			}
		}
		if def.DropProb != 0 && def.Type != LayerDropout {
			newDefs = append(newDefs, LayerDef{Type: LayerDropout, DropProb: def.DropProb})
		}
	}
	return newDefs
}

// takes a list of layer definitions and creates the network layer objects
func (n *Net) MakeLayers(defs []LayerDef, r *rand.Rand) {
	// few checks
	if len(defs) < 2 {
		panic("convnet: at least one input layer and one loss layer are required")
	}
	if defs[0].Type != LayerInput {
		panic("convnet: first layer must be the input layer, to declare size of inputs")
	}

	defs = desugar(defs)

	// create the layers
	n.Layers = make([]Layer, len(defs))
	for i, def := range defs {
		if i > 0 {
			prev := n.Layers[i-1]
			def.InSx = prev.OutSx()
			def.InSy = prev.OutSy()
			def.InDepth = prev.OutDepth()
		}

		switch def.Type {
		case LayerFC:
			n.Layers[i] = &FullyConnLayer{}
		case LayerLRN:
			n.Layers[i] = &LocalResponseNormalizationLayer{}
		case LayerDropout:
			n.Layers[i] = &DropoutLayer{}
		case LayerInput:
			n.Layers[i] = &InputLayer{}
		case LayerSoftmax:
			n.Layers[i] = &SoftmaxLayer{}
		case LayerRegression:
			n.Layers[i] = &RegressionLayer{}
		case LayerConv:
			n.Layers[i] = &ConvLayer{}
		case LayerPool:
			n.Layers[i] = &PoolLayer{}
		case LayerRelu:
			n.Layers[i] = &ReluLayer{}
		case LayerSigmoid:
			n.Layers[i] = &SigmoidLayer{}
		case LayerTanh:
			n.Layers[i] = &TanhLayer{}
		case LayerMaxout:
			n.Layers[i] = &MaxoutLayer{}
		case LayerSVM:
			n.Layers[i] = &SVMLayer{}
		default:
			panic("convnet: unrecognized layer type: " + def.Type.String())
		}

		n.Layers[i].fromDef(def, r)
	}
}

// forward prop the network.
// The trainer class passes is_training = true, but when this function is
// called from outside (not from the trainer), it defaults to prediction mode
func (n *Net) Forward(v *Vol, isTraining bool) *Vol {
	act := n.Layers[0].Forward(v, isTraining)

	for i := 1; i < len(n.Layers); i++ {
		act = n.Layers[i].Forward(act, isTraining)
	}

	return act
}

func (n *Net) CostLoss(v *Vol, y LossData) float64 {
	n.Forward(v, false)

	return n.Layers[len(n.Layers)-1].(LossLayer).BackwardLoss(y)
}

// backprop: compute gradients wrt all parameters
func (n *Net) Backward(y LossData) float64 {
	loss := n.Layers[len(n.Layers)-1].(LossLayer).BackwardLoss(y) // last layer assumed to be loss layer

	// first layer assumed input
	for i := len(n.Layers) - 2; i >= 0; i-- {
		n.Layers[i].Backward()
	}

	return loss
}

// accumulate parameters and gradients for the entire network
func (n *Net) ParamsAndGrads() []ParamsAndGrads {
	var response []ParamsAndGrads

	for _, l := range n.Layers {
		response = append(response, l.ParamsAndGrads()...)
	}

	return response
}

// this is a convenience function for returning the argmax
// prediction, assuming the last layer of the net is a softmax
func (n *Net) Prediction() int {
	s, ok := n.Layers[len(n.Layers)-1].(*SoftmaxLayer)
	if !ok {
		panic("convnet: Net.Prediction assumes softmax as the last layer of the net!")
	}

	p := s.outAct.W
	maxv, maxi := p[0], 0

	for i := 1; i < len(p); i++ {
		if p[i] > maxv {
			maxv, maxi = p[i], i
		}
	}

	return maxi // return index of the class with highest class probability
}
func (n *Net) UnmarshalJSON(b []byte) error {
	var rawData struct {
		Layers []json.RawMessage `json:"layers"`
	}

	if err := json.Unmarshal(b, &rawData); err != nil {
		return err
	}

	n.Layers = make([]Layer, 0, len(rawData.Layers))

	for _, lj := range rawData.Layers {
		var t struct {
			LayerType string `json:"layer_type"`
		}

		if err := json.Unmarshal(lj, &t); err != nil {
			return err
		}

		var l Layer

		switch t.LayerType {
		case "input":
			l = &InputLayer{}
		case "relu":
			l = &ReluLayer{}
		case "sigmoid":
			l = &SigmoidLayer{}
		case "tanh":
			l = &TanhLayer{}
		case "dropout":
			l = &DropoutLayer{}
		case "conv":
			l = &ConvLayer{}
		case "pool":
			l = &PoolLayer{}
		case "lrn":
			l = &LocalResponseNormalizationLayer{}
		case "softmax":
			l = &SoftmaxLayer{}
		case "regression":
			l = &RegressionLayer{}
		case "fc":
			l = &FullyConnLayer{}
		case "maxout":
			l = &MaxoutLayer{}
		case "svm":
			l = &SVMLayer{}
		default:
			return fmt.Errorf("convnet: unknown layer type %q", t.LayerType)
		}

		if err := l.UnmarshalJSON(b); err != nil {
			return err
		}

		n.Layers = append(n.Layers, l)
	}

	return nil
}
