//go:generate stringer -type TrainerMethod -linecomment

package convnet

import "math"

type TrainerMethod int

const (
	MethodSGD        TrainerMethod = iota // sgd
	MethodAdam                            // adam
	MethodADAGrad                         // adagrad
	MethodADADelta                        // adadelta
	MethodWindowGrad                      // windowgrad
	MethodNetsterov                       // netsterov
)

type TrainerOptions struct {
	LearningRate float64
	L1Decay      float64
	L2Decay      float64
	BatchSize    int
	Method       TrainerMethod

	Momentum float64
	Ro       float64 // used in adadelta
	Eps      float64 // used in adam or adadelta
	Beta1    float64 // used in adam
	Beta2    float64 // used in adam
}

var DefaultTrainerOptions = TrainerOptions{
	LearningRate: 0.01,
	L1Decay:      0.0,
	L2Decay:      0.0,
	BatchSize:    1,
	Method:       MethodSGD,

	Momentum: 0.9,
	Ro:       0.95,
	Eps:      1e-8,
	Beta1:    0.9,
	Beta2:    0.999,
}

type Trainer struct {
	Net *Net
	TrainerOptions

	k    int         // iteration counter
	gsum [][]float64 // last iteration gradients (used for momentum calculations)
	xsum [][]float64 // used in adam or adadelta
}

type TrainingResult struct {
	Loss        float64
	CostLoss    float64
	L1DecayLoss float64
	L2DecayLoss float64
}

func NewTrainer(net *Net, opts TrainerOptions) *Trainer {
	return &Trainer{
		Net:            net,
		TrainerOptions: opts,
	}
}

func (t *Trainer) Train(x *Vol, y LossData) TrainingResult {
	t.Net.Forward(x, true) // also set the flag that lets the net know we're just training

	costLoss := t.Net.Backward(y)
	l2DecayLoss := 0.0
	l1DecayLoss := 0.0

	t.k++
	if t.k%t.BatchSize == 0 {
		pglist := t.Net.ParamsAndGrads()

		// initialize lists for accumulators. Will only be done once on first iteration
		if len(t.gsum) == 0 && (t.Method != MethodSGD || t.Momentum > 0.0) {
			// only vanilla sgd doesnt need either lists
			// momentum needs gsum
			// adagrad needs gsum
			// adam and adadelta needs gsum and xsum
			for i := 0; i < len(pglist); i++ {
				t.gsum = append(t.gsum, make([]float64, len(pglist[i].Params)))

				if t.Method == MethodAdam || t.Method == MethodADADelta {
					t.xsum = append(t.xsum, make([]float64, len(pglist[i].Params)))
				} else {
					t.xsum = append(t.xsum, nil) // conserve memory
				}
			}
		} else if len(t.gsum) == 0 {
			// so we can grab them from outside the switch statement later
			t.gsum = make([][]float64, len(pglist))
			t.xsum = make([][]float64, len(pglist))
		}

		// perform an update for all sets of weights
		for i, pg := range pglist {
			p, g := pg.Params, pg.Grads

			// learning rate for some parameters.
			l2Decay := t.L2Decay * pg.L2DecayMul
			l1Decay := t.L1Decay * pg.L1DecayMul

			for j := range p {
				l2DecayLoss += l2Decay * p[j] * p[j] / 2 // accumulate weight decay loss
				l1DecayLoss += l1Decay * math.Abs(p[j])
				l1grad := l1Decay * math.Copysign(1, p[j])
				l2grad := l2Decay * p[j]

				gij := (l2grad + l1grad + g[j]) / float64(t.BatchSize) // raw batch gradient

				gsumi, xsumi := t.gsum[i], t.xsum[i]

				switch t.Method {
				case MethodAdam:
					// adam update
					gsumi[j] = gsumi[j]*t.Beta1 + (1-t.Beta1)*gij                 // update biased first moment estimate
					xsumi[j] = xsumi[j]*t.Beta2 + (1-t.Beta2)*gij*gij             // update biased second moment estimate
					biasCorr1 := gsumi[j] * (1 - math.Pow(t.Beta1, float64(t.k))) // correct bias first moment estimate
					biasCorr2 := xsumi[j] * (1 - math.Pow(t.Beta2, float64(t.k))) // correct bias second moment estimate
					dx := -t.LearningRate * biasCorr1 / (math.Sqrt(biasCorr2) + t.Eps)
					p[j] += dx
				case MethodADAGrad:
					// adagrad update
					gsumi[j] = gsumi[j] + gij*gij
					var dx = -t.LearningRate / math.Sqrt(gsumi[j]+t.Eps) * gij
					p[j] += dx
				case MethodWindowGrad:
					// this is adagrad but with a moving window weighted average
					// so the gradient is not accumulated over the entire history of the run.
					// it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
					gsumi[j] = t.Ro*gsumi[j] + (1-t.Ro)*gij*gij
					dx := -t.LearningRate / math.Sqrt(gsumi[j]+t.Eps) * gij // eps added for better conditioning
					p[j] += dx
				case MethodADADelta:
					gsumi[j] = t.Ro*gsumi[j] + (1-t.Ro)*gij*gij
					dx := -math.Sqrt((xsumi[j]+t.Eps)/(gsumi[j]+t.Eps)) * gij
					xsumi[j] = t.Ro*xsumi[j] + (1-t.Ro)*dx*dx // yes, xsum lags behind gsum by 1.
					p[j] += dx
				case MethodNetsterov:
					dx := gsumi[j]
					gsumi[j] = gsumi[j]*t.Momentum + t.LearningRate*gij
					dx = t.Momentum*dx - (1.0+t.Momentum)*gsumi[j]
					p[j] += dx
				default:
					// assume SGD
					if t.Momentum > 0.0 {
						// momentum update
						dx := t.Momentum*gsumi[j] - t.LearningRate*gij // step
						gsumi[j] = dx                                  // back this up for next iteration of momentum
						p[j] += dx                                     // apply corrected gradient
					} else {
						// vanilla sgd
						p[j] += -t.LearningRate * gij
					}
				}

				g[j] = 0.0 // zero out gradient so that we can begin accumulating anew
			}
		}
	}

	return TrainingResult{
		Loss:        costLoss + l1DecayLoss + l2DecayLoss,
		CostLoss:    costLoss,
		L1DecayLoss: l1DecayLoss,
		L2DecayLoss: l2DecayLoss,
	}
}
