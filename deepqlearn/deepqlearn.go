package deepqlearn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/BenLubar/convnet"
	"github.com/BenLubar/convnet/cnnutil"
)

// An agent is in state0 and does action0
// environment then assigns reward0 and provides new state, state1
// Experience nodes store all this information, which is used in the
// Q-learning update step
type Experience struct {
	State0  []float64
	Action0 int
	Reward0 float64
	State1  []float64
}

type BrainOptions struct {
	// in number of time steps, of temporal memory
	// the ACTUAL input to the net will be (x,a) temporal_window times, and followed by current x
	// so to have no information from previous time step going into value function, set to 0.
	TemporalWindow int
	// size of experience replay memory
	ExperienceSize int
	// number of examples in experience replay memory before we begin learning
	StartLearnThreshold int
	// gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
	Gamma float64
	// number of steps we will learn for
	LearningStepsTotal int
	// how many steps of the above to perform only random actions (in the beginning)?
	LearningStepsBurnin int
	// what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
	EpsilonMin float64
	// what epsilon to use at test time? (i.e. when learning is disabled)
	EpsilonTestTime float64
	// advanced feature. Sometimes a random action should be biased towards some values
	// for example in flappy bird, we may want to choose to not flap more often
	// this better sum to 1 by the way, and be of length this.num_actions
	RandomActionDistribution []float64

	LayerDefs        []convnet.LayerDef
	HiddenLayerSizes []int
	Rand             *rand.Rand

	TDTrainerOptions convnet.TrainerOptions
}

var DefaultBrainOptions = BrainOptions{
	TemporalWindow:           1,
	ExperienceSize:           30000,
	StartLearnThreshold:      int(math.Floor(math.Min(30000*0.1, 1000))),
	Gamma:                    0.8,
	LearningStepsTotal:       100000,
	LearningStepsBurnin:      3000,
	EpsilonMin:               0.05,
	EpsilonTestTime:          0.01,
	RandomActionDistribution: nil,
	TDTrainerOptions: convnet.TrainerOptions{
		LearningRate: 0.01,
		Momentum:     0.0,
		BatchSize:    64,
		L2Decay:      0.01,
	},
}

// A Brain object does all the magic.
// over time it receives some inputs and some rewards
// and its job is to set the outputs to maximize the expected reward
type Brain struct {
	TemporalWindow           int
	ExperienceSize           int
	StartLearnThreshold      int
	Gamma                    float64
	LearningStepsTotal       int
	LearningStepsBurnin      int
	EpsilonMin               float64
	EpsilonTestTime          float64
	RandomActionDistribution []float64

	NetInputs  int
	NumStates  int
	NumActions int
	WindowSize int

	StateWindow  [][]float64
	ActionWindow []int
	RewardWindow []float64
	NetWindow    [][]float64

	Rand       *rand.Rand
	ValueNet   convnet.Net
	TDTrainer  *convnet.Trainer
	Experience []Experience

	Age                 int
	ForwardPasses       int
	Epsilon             float64
	LatestReward        float64
	LastInputArray      []float64
	AverageRewardWindow *cnnutil.Window
	AverageLossWindow   *cnnutil.Window
	Learning            bool
}

func NewBrain(numStates, numActions int, opt BrainOptions) (*Brain, error) {
	b := &Brain{
		TemporalWindow:           opt.TemporalWindow,
		ExperienceSize:           opt.ExperienceSize,
		StartLearnThreshold:      opt.StartLearnThreshold,
		Gamma:                    opt.Gamma,
		LearningStepsTotal:       opt.LearningStepsTotal,
		LearningStepsBurnin:      opt.LearningStepsBurnin,
		EpsilonMin:               opt.EpsilonMin,
		EpsilonTestTime:          opt.EpsilonTestTime,
		RandomActionDistribution: opt.RandomActionDistribution,
	}

	if b.RandomActionDistribution != nil {
		b.RandomActionDistribution = opt.RandomActionDistribution
		if len(b.RandomActionDistribution) != numActions {
			return nil, errors.New("deepqlearn: random_action_distribution should be same length as num_actions")
		}

		sum := 0.0
		for _, a := range b.RandomActionDistribution {
			sum += a
		}

		if math.Abs(sum-1.0) > 0.0001 {
			return nil, errors.New("deepqlearn: random_action_distribution should sum to 1!")
		}
	}

	// states that go into neural net to predict optimal action look as
	// x0,a0,x1,a1,x2,a2,...xt
	// this variable controls the size of that temporal window. Actions are
	// encoded as 1-of-k hot vectors
	b.NetInputs = numStates*b.TemporalWindow + numActions*b.TemporalWindow + numStates
	b.NumStates = numStates
	b.NumActions = numActions

	b.WindowSize = b.TemporalWindow
	if b.WindowSize < 2 {
		// must be at least 2, but if we want more context even more
		b.WindowSize = 2
	}

	b.StateWindow = make([][]float64, b.WindowSize)
	b.ActionWindow = make([]int, b.WindowSize)
	b.RewardWindow = make([]float64, b.WindowSize)
	b.NetWindow = make([][]float64, b.WindowSize)

	// create [state -> value of all possible actions] modeling net for the value function
	layerDefs := opt.LayerDefs
	if layerDefs != nil {
		// this is an advanced usage feature, because size of the input to the network, and number of
		// actions must check out. This is not very pretty Object Oriented programming but I can"t see
		// a way out of it :(

		if len(layerDefs) < 2 {
			return nil, errors.New("deepqlearn: must have at least 2 layers")
		}

		if layerDefs[0].Type != convnet.LayerInput {
			return nil, errors.New("deepqlearn: first layer must be input layer!")
		}

		if layerDefs[len(layerDefs)-1].Type != convnet.LayerRegression {
			return nil, errors.New("deepqlearn: last layer must be input regression!")
		}

		if layerDefs[0].OutDepth*layerDefs[0].OutSx*layerDefs[0].OutSy != b.NetInputs {
			return nil, errors.New("deepqlearn: Number of inputs must be num_states * temporal_window + num_actions * temporal_window + num_states!")
		}

		if layerDefs[len(layerDefs)-1].NumNeurons != b.NumActions {
			return nil, errors.New("deepqlearn: Number of regression neurons should be num_actions!")
		}
	} else {
		// create a very simple neural net by default
		layerDefs = append(layerDefs, convnet.LayerDef{Type: convnet.LayerInput, OutSx: 1, OutSy: 1, OutDepth: b.NetInputs})

		for _, hl := range opt.HiddenLayerSizes {
			// relu by default
			layerDefs = append(layerDefs, convnet.LayerDef{Type: convnet.LayerFC, NumNeurons: hl, Activation: convnet.LayerRelu})
		}

		// value function output
		layerDefs = append(layerDefs, convnet.LayerDef{Type: convnet.LayerRegression, NumNeurons: numActions})
	}

	b.Rand = opt.Rand
	if b.Rand == nil {
		b.Rand = rand.New(rand.NewSource(0))
	}

	b.ValueNet.MakeLayers(layerDefs, b.Rand)

	// and finally we need a Temporal Difference Learning trainer!
	b.TDTrainer = convnet.NewTrainer(&b.ValueNet, opt.TDTrainerOptions)

	// experience replay
	b.Experience = make([]Experience, 0, b.ExperienceSize)

	// various housekeeping variables
	b.Age = 0           // incremented every backward()
	b.ForwardPasses = 0 // incremented every forward()
	b.Epsilon = 1.0     // controls exploration exploitation tradeoff. Should be annealed over time
	b.LatestReward = 0
	b.LastInputArray = nil
	b.AverageRewardWindow = cnnutil.NewWindow(1000, 10)
	b.AverageLossWindow = cnnutil.NewWindow(1000, 10)
	b.Learning = true

	return b, nil
}

// a bit of a helper function. It returns a random action
// we are abstracting this away because in future we may want to
// do more sophisticated things. For example some actions could be more
// or less likely at "rest"/default state.
func (b *Brain) RandomAction() int {
	if b.RandomActionDistribution == nil {
		return b.Rand.Intn(b.NumActions)
	}

	// okay, lets do some fancier sampling:
	p := b.Rand.Float64()
	cumprob := 0.0

	for k := 0; k < b.NumActions; k++ {
		cumprob += b.RandomActionDistribution[k]

		if p < cumprob {
			return k
		}
	}

	// rounding error
	return b.NumActions - 1
}

// compute the value of doing any action in this state
// and return the argmax action and its value
func (b *Brain) Policy(s []float64) (action int, value float64) {
	svol := convnet.NewVol(1, 1, b.NetInputs, 0)
	svol.W = s

	actionValues := b.ValueNet.Forward(svol, false)

	maxval, maxk := actionValues.W[0], 0

	for k := 1; k < b.NumActions; k++ {
		if actionValues.W[k] > maxval {
			maxk, maxval = k, actionValues.W[k]
		}
	}

	return maxk, maxval
}

// return s = (x,a,x,a,x,a,xt) state vector.
// It"s a concatenation of last window_size (x,a) pairs and current state x
func (b *Brain) NetInput(xt []float64) []float64 {
	var w []float64
	w = append(w, xt...) // start with current state

	// and now go backwards and append states and actions from history temporal_window times
	for k := 0; k < b.TemporalWindow; k++ {
		// state
		w = append(w, b.StateWindow[b.WindowSize-1-k]...)

		// action, encoded as 1-of-k indicator vector. We scale it up a bit because
		// we dont want weight regularization to undervalue this information, as it only exists once
		action1ofk := make([]float64, b.NumActions)

		action1ofk[b.ActionWindow[b.WindowSize-1-k]] = float64(b.NumStates)

		w = append(w, action1ofk...)
	}
	return w
}

// compute forward (behavior) pass given the input neuron signals from body
func (b *Brain) Forward(inputArray []float64) int {
	b.ForwardPasses++
	b.LastInputArray = inputArray // back this up

	// create network input
	var (
		netInput []float64
		action   int
	)
	if b.ForwardPasses > b.TemporalWindow {
		// we have enough to actually do something reasonable
		netInput = b.NetInput(inputArray)

		if b.Learning {
			// compute epsilon for the epsilon-greedy policy
			b.Epsilon = math.Min(1.0, math.Max(b.EpsilonMin, 1.0-float64(b.Age-b.LearningStepsBurnin)/float64(b.LearningStepsTotal-b.LearningStepsBurnin)))
		} else {
			b.Epsilon = b.EpsilonTestTime // use test-time value
		}

		rf := b.Rand.Float64()
		if rf < b.Epsilon {
			// choose a random action with epsilon probability
			action = b.RandomAction()
		} else {
			// otherwise use our policy to make decision
			action, _ = b.Policy(netInput)
		}
	} else {
		// pathological case that happens first few iterations
		// before we accumulate window_size inputs
		netInput = nil
		action = b.RandomAction()
	}

	// remember the state and action we took for backward pass
	copy(b.NetWindow, b.NetWindow[1:])
	b.NetWindow[len(b.NetWindow)-1] = netInput
	copy(b.StateWindow, b.StateWindow[1:])
	b.StateWindow[len(b.StateWindow)-1] = inputArray
	copy(b.ActionWindow, b.ActionWindow[1:])
	b.ActionWindow[len(b.ActionWindow)-1] = action

	return action
}

func (b *Brain) Backward(reward float64) {
	b.LatestReward = reward
	b.AverageRewardWindow.Add(reward)
	copy(b.RewardWindow, b.RewardWindow[1:])
	b.RewardWindow[len(b.RewardWindow)-1] = reward

	if !b.Learning {
		return
	}

	// various book-keeping
	b.Age++

	// it is time t+1 and we have to store (s_t, a_t, r_t, s_{t+1}) as new experience
	// (given that an appropriate number of state measurements already exist, of course)
	if b.ForwardPasses > b.TemporalWindow+1 {
		n := b.WindowSize
		e := Experience{
			State0:  b.NetWindow[n-2],
			Action0: b.ActionWindow[n-2],
			Reward0: b.RewardWindow[n-2],
			State1:  b.NetWindow[n-1],
		}

		if len(b.Experience) < b.ExperienceSize {
			b.Experience = append(b.Experience, e)
		} else {
			// replace. finite memory!
			ri := b.Rand.Intn(b.ExperienceSize)
			b.Experience[ri] = e
		}
	}

	// learn based on experience, once we have some samples to go on
	// this is where the magic happens...
	if len(b.Experience) > b.StartLearnThreshold {
		avcost := 0.0

		for k := 0; k < b.TDTrainer.BatchSize; k++ {
			re := b.Rand.Intn(len(b.Experience))
			e := b.Experience[re]

			x := convnet.NewVol(1, 1, b.NetInputs, 0)
			x.W = e.State0

			_, maxact := b.Policy(e.State1)
			r := e.Reward0 + b.Gamma*maxact

			loss := b.TDTrainer.Train(x, convnet.LossData{Dim: e.Action0, Val: r})
			avcost += loss.Loss
		}

		avcost /= float64(b.TDTrainer.BatchSize)
		b.AverageLossWindow.Add(avcost)
	}
}

func (b *Brain) String() string {
	return fmt.Sprintf(`experience replay size: %d
exploration epsilon: %f
age: %d
average Q-learning loss: %f
smooth-ish reward: %f
`, len(b.Experience), b.Epsilon, b.Age, b.AverageLossWindow.Average(), b.AverageRewardWindow.Average())
}
