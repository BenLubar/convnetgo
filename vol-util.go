package convnet

import (
	"image"
	"image/draw"
)

// Volume utilities
// intended for use with data augmentation
// crop is the size of output
// dx,dy are offset wrt incoming volume, of the shift
// fliplr is boolean on whether we also want to flip left<->right
//
// Note: When converting from convnetjs, dx and dy default to a
// random number in [0, v.S[xy] - crop).
func (v *Vol) Augment(crop, dx, dy int, fliplr bool) *Vol {
	// note assumes square outputs of size crop x crop

	// randomly sample a crop in the input volume
	var w *Vol

	if crop != v.Sx || dx != 0 || dy != 0 {
		w = NewVol(crop, crop, v.Depth, 0.0)
		for x := 0; x < crop; x++ {
			for y := 0; y < crop; y++ {
				if x+dx < 0 || x+dx >= v.Sx || y+dy < 0 || y+dy >= v.Sy {
					continue // oob
				}

				for d := 0; d < v.Depth; d++ {
					w.Set(x, y, d, v.Get(x+dx, y+dy, d)) // copy data over
				}
			}
		}
	} else {
		w = v
	}

	if fliplr {
		// flip volume horziontally
		w2 := w.CloneAndZero()

		for x := 0; x < w.Sx; x++ {
			for y := 0; y < w.Sy; y++ {
				for d := 0; d < w.Depth; d++ {
					w2.Set(x, y, d, w.Get(w.Sx-x-1, y, d)) // copy data over
				}
			}
		}

		w = w2 // swap
	}

	return w
}

// returns a Vol of size (W, H, 4). 4 is for RGBA
func ImgToVol(img image.Image, convertGrayscale bool) *Vol {
	// ensure RGBA
	rgba, ok := img.(*image.RGBA)
	if !ok {
		rgba = image.NewRGBA(img.Bounds())
		draw.Draw(rgba, rgba.Rect, img, rgba.Rect.Min, draw.Src)
	}

	// prepare the input: get pixels and normalize them
	p := rgba.Pix
	W := rgba.Rect.Dx()
	H := rgba.Rect.Dy()
	v := NewVol(W, H, 4, 0.0) // input volume (image)

	for y := 0; y < H; y++ {
		j := rgba.Stride * y

		for x := 0; x < W; x++ {
			// normalize image pixels to [-0.5, 0.5]
			v.Set(x, y, 0, float64(p[j+0])/255.0-0.5)
			v.Set(x, y, 1, float64(p[j+1])/255.0-0.5)
			v.Set(x, y, 2, float64(p[j+2])/255.0-0.5)
			v.Set(x, y, 3, float64(p[j+3])/255.0-0.5)

			j += 4
		}
	}

	if convertGrayscale {
		// flatten into depth=1 array
		v1 := NewVol(W, H, 1, 0.0)

		for i := 0; i < W; i++ {
			for j := 0; j < H; j++ {
				v1.Set(i, j, 0, v.Get(i, j, 0))
			}
		}

		v = v1
	}

	return v
}
