package convnet

// return max and min of a given non-empty array.
func maxmin(w []float64) (maxi int, maxv float64, mini int, minv, dv float64) {
	if len(w) == 0 {
		return // ... ;s
	}

	maxv, minv = w[0], w[0]
	maxi, mini = 0, 0

	for i, v := range w {
		if v > maxv {
			maxv, maxi = v, i
		}

		if v < minv {
			minv, mini = v, i
		}
	}

	return maxi, maxv, mini, minv, maxv - minv
}
