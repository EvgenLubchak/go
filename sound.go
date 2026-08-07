package main

import (
	"bytes"
	"math"

	"github.com/hajimehoshi/ebiten/v2/audio"
)

const audioSampleRate = 44100

var (
	audioCtx          *audio.Context
	beatPlayer        *audio.Player
	currentPatternIdx int // який патерн зараз грає
)

func init() {
	audioCtx = audio.NewContext(audioSampleRate)
}

// drumPattern — 16-кроковий секвенсор.
// Кожен елемент масиву = одна шістнадцята нота (1/16 такту).
//
// Відповідність кроків і ударів (4/4):
//
//	Beat 1: steps 0-3
//	Beat 2: steps 4-7
//	Beat 3: steps 8-11
//	Beat 4: steps 12-15
//
// [GO: NAMED STRUCT WITH ARRAY FIELDS]
// [16]bool — фіксований масив, ініціалізується через індекс: {0: true, 4: true}
type drumPattern struct {
	name  string
	kick  [16]bool
	snare [16]bool
	hihat [16]bool
}

// patterns — всі доступні ритмічні схеми.
// Чергуються щорівня: level 1 → pattern 0, level 2 → pattern 1, і т.д.
//
// [GO: COMPOSITE LITERAL]
// []drumPattern{...} — slice з literal-значень структур.
var patterns = []drumPattern{
	{
		// Наш оригінальний мінімалістичний ритм
		name:  "Minimal",
		kick:  [16]bool{0: true, 8: true},
		snare: [16]bool{4: true, 12: true},
		hihat: [16]bool{0: true, 4: true, 8: true, 12: true},
	},
	{
		// Four-on-the-Floor — класичний драйвовий чіптюн
		// Kick:   x - - - x - - - x - - - x - - -
		// Snare:  - - - - x - - - - - - - x - - -
		// Hi-Hat: - - x - - - x - - - x - - - x -
		name:  "Four-on-the-Floor",
		kick:  [16]bool{0: true, 4: true, 8: true, 12: true},
		snare: [16]bool{4: true, 12: true},
		hihat: [16]bool{2: true, 6: true, 10: true, 14: true},
	},
	{
		// Syncopated — складніший з офбітними хетами
		// Kick:   x - - x - - - x x - - - - - - -
		// Snare:  - - - - x - - - - - - - x - - -
		// Hi-Hat: - - x - - x - x - - x - - x - x
		name:  "Syncopated",
		kick:  [16]bool{0: true, 3: true, 7: true, 8: true},
		snare: [16]bool{4: true, 12: true},
		hihat: [16]bool{2: true, 5: true, 7: true, 10: true, 13: true, 15: true},
	},
	{
		// Fast Dance — енергійний NES-стиль з 16-ма хетами
		// Kick:   x - - - x - - x x - - - x - - -
		// Snare:  - - - - x - - - - - - - x - - -
		// Hi-Hat: x x x x x x x x x x x x x x x x
		name:  "Fast Dance",
		kick:  [16]bool{0: true, 4: true, 7: true, 8: true, 12: true},
		snare: [16]bool{4: true, 12: true},
		hihat: [16]bool{0: true, 1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 7: true, 8: true, 9: true, 10: true, 11: true, 12: true, 13: true, 14: true, 15: true},
	},
}

// generateBeatFromPattern генерує один такт PCM (16-bit signed LE, stereo)
// за заданим drum pattern і темпом (BPM залежить від difficulty).
//
// [GO: STEP SEQUENCER LOGIC]
// Ділимо такт на 16 рівних кроків.
// Для кожного кроку — перевіряємо kick/snare/hihat і пишемо звук у буфер.
func generateBeatFromPattern(p drumPattern, difficulty float32) []byte {
	bpm := baseBPM + float64(difficulty-1)*bpmPerDifficulty
	if bpm > maxBPM {
		bpm = maxBPM
	}
	// [ТЕМП] Єдине місце в проєкті, де є РЕАЛЬНИЙ час: sixteenthSec нижче — секунди, а
	// не кадри. Тож при зміні gameTPS ритм треба переганяти разом із грою, інакше
	// половинна швидкість гри й незмінний біт розʼїдуться.
	bpm *= tpsScale()

	// шістнадцята нота = чверть від чвертної ноти
	sixteenthSec := (60.0 / bpm) / 4.0
	totalSamples := int(sixteenthSec * 16 * audioSampleRate)
	buf := make([]byte, totalSamples*4)

	for step := 0; step < 16; step++ {
		stepStart := int(float64(step) * sixteenthSec * audioSampleRate)

		if p.kick[step] {
			writeSamples(buf, stepStart, 80.0, 0.085, 0.85)
		}
		if p.snare[step] {
			writeSamples(buf, stepStart, 300.0, 0.050, 0.50)
		}
		if p.hihat[step] {
			writeSamples(buf, stepStart, 8000.0, 0.020, 0.25)
		}
	}

	return buf
}

// writeSamples записує квадратну хвилю з fade-out у PCM буфер.
//
// Square wave: sin(t) ≥ 0 → +amplitude, sin(t) < 0 → −amplitude.
// Fade-out: лінійне затухання → звук не кліпає на кінці.
func writeSamples(buf []byte, startSample int, freq, durationSec, amplitude float64) {
	count := int(durationSec * audioSampleRate)
	for i := 0; i < count; i++ {
		pos := (startSample + i) * 4
		if pos+3 >= len(buf) {
			break
		}
		fade := 1.0 - float64(i)/float64(count)
		sinVal := math.Sin(2 * math.Pi * freq * float64(i) / audioSampleRate)
		var v float64
		if sinVal >= 0 {
			v = amplitude * fade
		} else {
			v = -amplitude * fade
		}
		s := int16(v * math.MaxInt16)
		lo := byte(s)
		hi := byte(s >> 8)
		buf[pos], buf[pos+1] = lo, hi   // лівий канал
		buf[pos+2], buf[pos+3] = lo, hi // правий канал
	}
}

// tpsScale — у скільки разів поточний темп швидший за базові 120 тіків.
//
// Живе тут, поруч із єдиним споживачем: більше в проєкті реального часу немає, і
// заводити для цього загальний хелпер означало б натякати, що він є.
func tpsScale() float64 { return float64(gameTPS) / 120.0 }

// startBeat зупиняє поточний програвач і запускає наступний патерн у циклі.
// На рестарті — викликати з попереднім скиданням currentPatternIdx = 0.
func startBeat(difficulty float32) {
	if !soundEnabled {
		return
	}

	if beatPlayer != nil {
		_ = beatPlayer.Close() // помилка закриття не критична — ігноруємо
		beatPlayer = nil
	}

	p := patterns[currentPatternIdx%len(patterns)]
	currentPatternIdx++

	pcm := generateBeatFromPattern(p, difficulty)
	loop := audio.NewInfiniteLoop(bytes.NewReader(pcm), int64(len(pcm)))

	var err error
	beatPlayer, err = audioCtx.NewPlayer(loop)
	if err != nil {
		return
	}
	beatPlayer.Play()
}
