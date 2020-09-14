package convnet_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/BenLubar/convnet"
)

// Simple Fully-Connected Neural Net Classifier.
func createTestNet() (*convnet.Net, *convnet.SGDTrainer) {
	net := &convnet.Net{}

	layerDefs := []convnet.LayerDef{
		{Type: convnet.LayerInput, OutSx: 1, OutSy: 1, OutDepth: 2},
		{Type: convnet.LayerFC, NumNeurons: 5, Activation: convnet.ActivationTanh},
		{Type: convnet.LayerFC, NumNeurons: 5, Activation: convnet.ActivationTanh},
		{Type: convnet.LayerSoftMax, NumClasses: 3},
	}

	net.MakeLayers(layerDefs)

	trainer := convnet.NewSGDTrainer(net, &convnet.NetOptions{LearningRate: 0.0001, Momentum: 0.0, BatchSize: 1, L2Decay: 0.0})

	return net, trainer
}

// it should be possible to initialize.
func TestInitialize(t *testing.T) {
	// tanh are their own layers. Softmax gets its own fully connected layer.
	// this should all get desugared just fine.

	net, _ := createTestNet()

	if len(net.Layers) != 7 {
		t.Errorf("expected 7 layers, but there are %d", len(net.Layers))
	}
}

// it should forward prop volumes to probabilities
func TestForward(t *testing.T) {
	net, _ := createTestNet()

	x := convnet.NewVol1D([]float64{0.2, -0.3})
	pv := net.Forward(x)

	// 3 classes output
	if len(pv.W) != 3 {
		t.Errorf("expected probability_volume.W to have length 3, but length is %d", len(pv.W))
	}
	var total float64
	for i, f := range pv.W {
		if f <= 0 || f >= 1 {
			t.Errorf("expected probability_volume[%d] to be in (0, 1) but it is %f", i, f)
		}

		total += f
	}

	if math.Abs(total-1) > 0.0001 {
		t.Errorf("expected total probability to approximately equal 1, but it is %f", total)
	}
}

// it should increase probabilities for ground truth class when trained
func TestTrain(t *testing.T) {
	net, trainer := createTestNet()
	r := rand.New(rand.NewSource(0))

	// lets test 100 random point and label settings
	// note that this should work since l2 and l1 regularization are off
	// an issue is that if step size is too high, this could technically fail...
	for k := 0; k < 100; k++ {
		x := convnet.NewVol1D([]float64{r.Float64()*2 - 1, r.Float64()*2 - 1})
		pv := net.Forward(x)
		gti := r.Intn(3)
		trainer.Train(x, gti)
		pv2 := net.Forward(x)
		if pv2.W[gti] <= pv.W[gti] {
			t.Errorf("expected trained class probability to increase, but it changed from %f to %f", pv.W[gti], pv2.W[gti])
		}
	}
}

// it should compute correct gradient at data
func TestGradient(t *testing.T) {
	// here we only test the gradient at data, but if this is
	// right then that's comforting, because it is a function
	// of all gradients above, for all layers.

	r := rand.New(rand.NewSource(0))

	net, trainer := createTestNet()

	x := convnet.NewVol1D([]float64{r.Float64()*2 - 1, r.Float64()*2 - 1})
	gti := r.Intn(3)      // ground truth index
	trainer.Train(x, gti) // computes gradients at all layers, and at x

	const delta = 0.000001

	for i := 0; i < len(x.W); i++ {
		gradAnalytic := x.Dw[i]

		xold := x.W[i]
		x.W[i] += delta
		c0 := net.GetCostLoss(x, gti)
		x.W[i] -= 2 * delta
		c1 := net.GetCostLoss(x, gti)
		x.W[i] = xold // reset

		gradNumeric := (c0 - c1) / (2 * delta)
		relError := math.Abs(gradAnalytic-gradNumeric) / math.Abs(gradAnalytic+gradNumeric)
		t.Logf("%d: numeric: %f, analytic: %f => rel error %f", i, gradNumeric, gradAnalytic, relError)
		if relError >= 1e-2 {
			t.Error("rel error too high")
		}
	}
}
